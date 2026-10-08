#!/usr/bin/env python3
"""
AVE: Anchor-based Video Editing (zero-shot version, Wan2.2 I2V backbone)

Pipeline:
1. Extract N evenly spaced keyframes from input video (default N = 3, including first & last)
2. Edit keyframes via Gemini multi-turn conversation (consistent style)
3. Generate a motion-focused video prompt from densely sampled frames of the source video
4. Use edited keyframes as anchors to generate output video via WAN I2V

Usage:
    export GEMINI_API_KEY=<your key>
    python video_edit.py --input_video input.mp4 \
        --edit_prompt "Convert the video to a watercolor style" \
        --ckpt_dir ./Wan2.2-I2V-A14B --num_gpus 1 --output edited.mp4
"""

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import requests
from PIL import Image
from google import genai
from google.genai import types


# ============== Google Gemini API ==============

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_RETRIES = 5


API_SLEEP = 5  # seconds to sleep between API calls to avoid rate limits


def _request_with_retry(url, payload, timeout=120):
    """Send POST request with retry on 429/5xx errors."""
    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** (attempt + 1), 60)  # 2, 4, 8, 16, 32 seconds, cap 60
            print(f"    Rate limited ({resp.status_code}), retrying in {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    # Last attempt, let it raise
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def gemini_chat(api_key, prompt, images=None, model="gemini-2.5-flash"):
    """
    Call Google Gemini API for text/vision tasks.

    Args:
        api_key: Google AI Studio API key
        prompt: Text prompt
        images: Optional list of PIL Images to include as context
        model: Gemini model name

    Returns:
        Response text string
    """
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"

    parts = []

    # Add images if provided
    if images:
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({
                "inlineData": {"mimeType": "image/png", "data": b64},
            })

    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.7},
    }

    result = _request_with_retry(url, payload, timeout=120)

    # Extract text from response
    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {result}")
    parts = candidates[0]["content"]["parts"]
    for part in parts:
        if "text" in part:
            return part["text"]

    raise RuntimeError(f"No text in Gemini response: {result}")


def gemini_multiturn_image_edit(api_key, images, prompt,
                                 model="gemini-3-pro-image-preview"):
    """
    Edit images via multi-turn chat using the official google-genai SDK.

    Creates a chat session and sends frames one by one. The SDK manages
    conversation history properly, so Gemini sees all previous edits
    and maintains consistent style across frames.

    Args:
        api_key: Google AI Studio API key
        images: List of PIL Images to edit
        prompt: Editing instruction
        model: Gemini model with image generation support

    Returns:
        List of edited PIL Images (same order as input)
    """
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    edited = []

    for i, img in enumerate(images):
        if i == 0:
            user_text = (
                f"I have {len(images)} keyframes from a video. "
                f"I will send them one by one. Please edit each frame with this instruction:\n\n"
                f"{prompt}\n\n"
                f"Important requirements:\n"
                f"- Apply the edit to the entire image\n"
                f"- Preserve the original composition, layout, and subject positions\n"
                f"- Make the edit look natural and photorealistic\n"
                f"- Keep all details that are not related to the edit unchanged\n\n"
                f"Here is frame 1. Edit it and return exactly 1 edited image."
            )
        else:
            user_text = (
                f"Here is frame {i + 1}. Same as the previous frames, {prompt}. "
                f"Keep the editing style exactly consistent. Return exactly 1 edited image."
            )

        # Send image + text as a single message
        msg_parts = [img, user_text]

        print(f"    Frame {i + 1}/{len(images)}...", end=" ", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                response = chat.send_message(msg_parts)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(2 ** (attempt + 1), 60)
                    print(f"rate limited, retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    raise
        else:
            # Final attempt, let it raise
            response = chat.send_message(msg_parts)

        # Extract edited image from response
        edited_img = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_img = Image.open(BytesIO(part.inline_data.data))
                break

        if edited_img is None:
            raise RuntimeError(f"No image in Gemini response for frame {i + 1}")

        edited.append(edited_img)
        print("done")

        # Sleep between calls to avoid rate limits
        if i < len(images) - 1:
            time.sleep(API_SLEEP)

    return edited


# ============== Video Processing ==============


def get_video_info(video_path):
    """Get video duration in seconds and fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_keyframes(video_path, num_keyframes):
    """
    Extract num_keyframes evenly spaced frames from video, including first and last.

    Returns:
        List of (frame_index_in_source, PIL.Image) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute evenly spaced frame indices including first (0) and last (total-1)
    if num_keyframes == 1:
        indices = [0]
    else:
        indices = [
            round(i * (total_frames - 1) / (num_keyframes - 1))
            for i in range(num_keyframes)
        ]

    keyframes = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx}, using previous frame")
            continue
        # OpenCV BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        keyframes.append((idx, pil_img))

    cap.release()
    return keyframes


def compute_output_frame_num(duration, output_fps=16):
    """
    Compute output frame_num for WAN based on source video duration.
    Must be 4n+1 (e.g., 81, 161, 241...).
    """
    desired_frames = round(duration * output_fps) + 1

    # Round up to nearest 4n+1
    n = math.ceil((desired_frames - 1) / 4)
    frame_num = 4 * n + 1

    return frame_num


def map_keyframes_to_output(num_keyframes, frame_num):
    """
    Map N keyframes evenly to output frame indices [0, frame_num-1].
    All indices are rounded to multiples of 4 (due to VAE temporal stride).
    Returns list of output frame indices.
    """
    if num_keyframes == 1:
        return [0]
    indices = []
    for i in range(num_keyframes):
        raw = i * (frame_num - 1) / (num_keyframes - 1)
        # Round to nearest multiple of 4
        idx = round(raw / 4) * 4
        # Clamp to valid range
        idx = min(idx, frame_num - 1)
        indices.append(idx)
    # Deduplicate while preserving order (in case rounding caused collisions)
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


# ============== Prompt Generation ==============



def extract_dense_frames_for_understanding(video_path, max_frames=16):
    """
    Extract dense frames from video for motion understanding.
    Independent of --num_keyframes, always extracts enough frames
    to accurately capture motion and camera movement.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num = min(max_frames, total_frames)

    indices = [round(i * (total_frames - 1) / (num - 1)) for i in range(num)] if num > 1 else [0]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def generate_video_prompt(api_key, video_edit_prompt, video_frames,
                          model="gemini-2.5-flash"):
    """
    Generate a motion-focused I2V prompt based on dense video frames.

    Uses densely sampled frames from the ORIGINAL video (not just the
    edited anchor keyframes) to accurately understand motion and camera
    movement.
    """
    system_prompt = """You are a video motion descriptor. Given densely sampled frames from a video (in chronological order) AND an editing instruction, write a prompt that describes the EDITED video's motion and dynamics.

Rules:
- CRITICAL: The frames show the ORIGINAL video BEFORE editing. You MUST apply the editing instruction to your description. Replace original subjects/scenes with the edited versions. For example, if the frames show a bird but the edit says "change the bird to a dog", describe a DOG's motion, NOT a bird's.
- Focus on HOW things move: directions, speed, trajectories, gestures
- Describe motion patterns: "walks from left to right", "camera slowly pans", "object moves toward the viewer"
- ALWAYS describe camera movement or viewpoint changes: pan, tilt, zoom, dolly, tracking, static, handheld, etc. Infer from how the scene composition shifts across frames.
- Mention dynamic elements: swaying, flowing, spinning, drifting, etc.
- Do NOT describe static scene appearance, colors, or style in detail (the anchor frames already define the look)
- Keep it to 2-4 sentences
- Write in present tense
- Output ONLY the motion description, nothing else

Examples:
- Edit: "change the woman to a robot" + frames showing a woman walking → "A robot walks forward along the street. Cars pass by in the background from right to left. The camera slowly tracks its movement with a steady dolly shot."
- Edit: "make it a snowy scene" + frames showing a summer park → "People stroll through the snowy park, footprints trailing behind them. The camera slowly pans from left to right. Snowflakes drift gently downward." """

    prompt = (
        f"{system_prompt}\n\n"
        f"Editing instruction: {video_edit_prompt}\n\n"
        f"These are {len(video_frames)} frames from the ORIGINAL video (before editing). "
        f"Describe the motion as it would appear AFTER the edit is applied. "
        f"Remember to replace original subjects with the edited versions:"
    )

    return gemini_chat(api_key, prompt, images=video_frames, model=model).strip()


# ============== Main Pipeline ==============


def run_pipeline(args):
    print("=" * 60)
    print("Video Editing Pipeline")
    if args.resume_from > 1:
        print(f"  (Resuming from step {args.resume_from})")
    print("=" * 60)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = work_dir / "original_keyframes"
    edited_dir = work_dir / "edited_keyframes"
    state_file = work_dir / "pipeline_state.json"

    # Load saved state if resuming
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"  Loaded saved state from {state_file}")

    def save_state():
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    # ---- Step 1: Analyze input video ----
    print("\n[Step 1] Analyzing input video...")
    info = get_video_info(args.input_video)
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  FPS: {info['fps']}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  Total frames: {info['total_frames']}")

    num_keyframes = max(args.num_keyframes, 2)
    print(f"  Keyframes to extract: {num_keyframes}")

    state["num_keyframes"] = num_keyframes
    state["video_info"] = info
    save_state()

    # ---- Step 2: Extract keyframes ----
    if args.resume_from <= 2:
        print(f"\n[Step 2] Extracting {num_keyframes} keyframes...")
        keyframes = extract_keyframes(args.input_video, num_keyframes)
        print(f"  Extracted {len(keyframes)} keyframes")

        orig_dir.mkdir(exist_ok=True)
        for i, (src_idx, img) in enumerate(keyframes):
            save_path = orig_dir / f"keyframe_{i:03d}_srcframe{src_idx}.png"
            img.save(save_path)
            print(f"  Saved: {save_path}")
    else:
        print(f"\n[Step 2] Skipped (resuming). Loading keyframes from {orig_dir}")
        keyframe_files = sorted(orig_dir.glob("keyframe_*.png"))
        if not keyframe_files:
            raise FileNotFoundError(f"No keyframes found in {orig_dir}. Cannot resume, run from step 1.")
        keyframes = [(0, Image.open(f)) for f in keyframe_files]
        num_keyframes = len(keyframes)
        print(f"  Loaded {num_keyframes} keyframes")

    # ---- Step 3: Edit keyframes ----
    image_edit_prompt = args.image_edit_prompt or args.edit_prompt
    print(f"\n  Edit prompt for frames: {image_edit_prompt}")
    edited_dir.mkdir(exist_ok=True)

    if args.resume_from <= 3:
        print(f"\n[Step 3] Editing {len(keyframes)} keyframes (multi-turn conversation)...")

        if args.dry_run:
            edited_frames = [img.copy() for _, img in keyframes]
            print(f"  [DRY RUN] Skipping image editing API calls")
        else:
            source_images = [img for _, img in keyframes]
            edited_frames = gemini_multiturn_image_edit(
                args.api_key, source_images, image_edit_prompt,
                model=args.image_edit_model,
            )
            time.sleep(API_SLEEP)

        for i, edited_img in enumerate(edited_frames):
            save_path = edited_dir / f"edited_{i:03d}.png"
            edited_img.save(save_path)
            print(f"  Saved: {save_path}")
    else:
        print(f"\n[Step 3] Skipped (resuming). Loading edited keyframes from {edited_dir}")
        edited_files = sorted(edited_dir.glob("edited_*.png"))
        if not edited_files:
            raise FileNotFoundError(f"No edited keyframes found in {edited_dir}. Cannot resume, run from step 3.")
        edited_frames = [Image.open(f) for f in edited_files]
        print(f"  Loaded {len(edited_frames)} edited keyframes")

    # ---- Step 4: Generate final video prompt (motion-focused) ----
    if args.video_prompt:
        final_prompt = args.video_prompt
        print(f"\n[Step 4] Using provided video prompt: {final_prompt}")
    elif args.resume_from <= 4:
        print(f"\n[Step 4] Generating motion-focused video prompt...")

        if args.dry_run:
            final_prompt = f"[DRY RUN] A motion description based on: {args.edit_prompt}"
            print(f"  [DRY RUN] Skipping API call")
        else:
            print(f"  Extracting dense frames from original video for motion understanding...")
            dense_frames = extract_dense_frames_for_understanding(args.input_video, max_frames=16)
            print(f"  Sampled {len(dense_frames)} frames for analysis")
            final_prompt = generate_video_prompt(
                args.api_key, args.edit_prompt, dense_frames,
                model=args.text_model,
            )
        print(f"  Final video prompt: {final_prompt}")
    else:
        final_prompt = state.get("video_prompt", "")
        if not final_prompt:
            raise ValueError("No video_prompt in saved state. Provide --video_prompt or run from step 4.")
        print(f"\n[Step 4] Skipped (resuming). Video prompt: {final_prompt}")

    state["video_prompt"] = final_prompt

    # Pre-compute generation metadata (needed for both prepare_only and full run)
    output_frame_num = compute_output_frame_num(info["duration"], output_fps=16)
    output_frame_indices = map_keyframes_to_output(len(edited_frames), output_frame_num)
    aspect = info["width"] / info["height"]
    size = args.size if args.size else ("832*480" if aspect >= 1 else "480*832")
    output_file = args.output or str(work_dir / "edited_video.mp4")

    # Save generation metadata for batch_generate.py
    state["output_frame_num"] = output_frame_num
    state["output_frame_indices"] = output_frame_indices
    state["size"] = size
    state["output_file"] = output_file
    state["num_edited_frames"] = len(edited_frames)
    state["edited_dir"] = str(edited_dir)
    save_state()

    if args.prepare_only:
        print(f"\n[Prepare Only] Saved state to {state_file}")
        print(f"  Video prompt:  {final_prompt}")
        print(f"  Frame num:     {output_frame_num}")
        print(f"  Size:          {size}")
        print(f"  Anchors:       {output_frame_indices}")
        print(f"  Output target: {output_file}")
        return

    # ---- Step 5: Build WAN I2V command ----
    print(f"\n[Step 5] Building WAN I2V generation command...")
    print(f"  Output frame_num: {output_frame_num}")
    print(f"  Anchor mapping:")

    frame_args = []
    for i, frame_idx in enumerate(output_frame_indices):
        img_path = str((edited_dir / f"edited_{i:03d}.png").resolve())
        frame_args.append(f"{frame_idx}:{img_path}")
        print(f"    Frame {frame_idx} <- edited_{i:03d}.png")

    generate_script = str(Path(__file__).resolve().parent / "generate.py")
    if args.num_gpus > 1:
        launcher = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint=localhost:{args.port}",
            generate_script,
        ]
        parallel_args = [
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size", str(args.num_gpus),
        ]
    else:
        # Single GPU: FSDP / sequence parallel are unavailable, so offload
        # the idle expert and T5 to CPU to fit in < 80GB VRAM
        launcher = [sys.executable, generate_script]
        parallel_args = [
            "--offload_model", "True",
            "--convert_model_dtype",
            "--t5_cpu",
        ]
    cmd = launcher + [
        "--size", size,
        "--ckpt_dir", os.path.abspath(args.ckpt_dir),
        "--frame_num", str(output_frame_num),
        "--prompt", final_prompt,
        "--save_file", os.path.abspath(output_file),
        "--base_seed", str(args.seed),
    ] + parallel_args + ["--frame_images"] + frame_args

    print(f"\n  Command:")
    print(f"    {' '.join(cmd)}")

    # ---- Step 6: Run generation ----
    if args.dry_run:
        print(f"\n[Step 6] [DRY RUN] Skipping video generation")
        print(f"  Would generate: {output_file}")
    else:
        print(f"\n[Step 6] Running WAN I2V generation...")
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        if result.returncode != 0:
            print(f"  ERROR: Generation failed with return code {result.returncode}")
            sys.exit(1)
        print(f"  Output saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Input video:       {args.input_video}")
    print(f"  Edit prompt:       {args.edit_prompt}")
    print(f"  Final video prompt:{final_prompt}")
    print(f"  Keyframes:         {num_keyframes}")
    print(f"  Output frames:     {output_frame_num}")
    print(f"  Output file:       {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Video Editing Pipeline via WAN I2V")

    # Required
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--edit_prompt", type=str, required=True,
                        help="Video editing prompt (e.g., 'Make it a snowy winter scene')")

    # API
    parser.add_argument("--api_key", type=str, default=None,
                        help="Gemini API key (default: read from the GEMINI_API_KEY env var)")
    parser.add_argument("--image_edit_model", type=str, default="gemini-3-pro-image-preview",
                        help="Gemini model used to edit keyframes")
    parser.add_argument("--text_model", type=str, default="gemini-2.5-flash",
                        help="Gemini model used to write the motion prompt")

    # Keyframes
    parser.add_argument("--num_keyframes", type=int, default=3,
                        help="Number of keyframes (anchors) to extract, including first & last (min 2)")

    # WAN settings
    parser.add_argument("--ckpt_dir", type=str, default="./Wan2.2-I2V-A14B",
                        help="WAN checkpoint directory")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs")
    parser.add_argument("--port", type=int, default=7559,
                        help="Distributed port")
    parser.add_argument("--size", type=str, default=None,
                        help="Output size (e.g., 832*480). Auto-detected if not set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for video generation (-1 for random)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--work_dir", type=str, default="./video_edit_workdir",
                        help="Working directory for intermediate files")

    # Resume
    parser.add_argument("--resume_from", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help="Resume from step N (1=analyze, 2=extract, 3=image edit, 4=video prompt, 5=build cmd, 6=generate)")
    parser.add_argument("--image_edit_prompt", type=str, default=None,
                        help="Override image edit prompt (default: use --edit_prompt directly)")
    parser.add_argument("--video_prompt", type=str, default=None,
                        help="Override final video prompt (skip step 4 prompt generation)")

    # Stage control
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only run steps 1-4 (extract, edit, prompt). Skip WAN generation.")

    # Debug
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip API calls and generation, just show the pipeline")

    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    # Resolve API key
    if args.api_key is None:
        args.api_key = os.environ.get("GEMINI_API_KEY")
    if args.api_key is None and not args.dry_run:
        parser.error("--api_key or GEMINI_API_KEY environment variable is required")

    run_pipeline(args)


if __name__ == "__main__":
    main()
