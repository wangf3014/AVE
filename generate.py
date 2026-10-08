# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified: reduced to the Wan2.2 I2V-A14B task and extended with multi-anchor
# conditioning (--frame_images).
"""
Anchor-conditioned video generation with Wan2.2 I2V-A14B.

Each anchor image is pinned to a frame index of the output video and the model
generates the frames in between. A single anchor at frame 0 is equivalent to
the original Wan2.2 image-to-video behavior.

Examples:
    # Single GPU
    python generate.py --ckpt_dir ./Wan2.2-I2V-A14B --size 832*480 --frame_num 81 \
        --frame_images 0:start.png 40:mid.png 80:end.png \
        --prompt "..." --offload_model True --convert_model_dtype --t5_cpu

    # Multi GPU (8x)
    torchrun --nproc_per_node=8 generate.py --ckpt_dir ./Wan2.2-I2V-A14B \
        --frame_images 0:start.png 80:end.png --prompt "..." \
        --dit_fsdp --t5_fsdp --ulysses_size 8
"""
import argparse
import logging
import os
import random
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool

TASK = "i2v-A14B"


def _parse_frame_images(frame_args):
    """Parse ['0:a.png', '40:b.png'] into {0: 'a.png', 40: 'b.png'}."""
    frame_paths = {}
    for frame_arg in frame_args:
        if ':' not in frame_arg:
            raise ValueError(
                f"Invalid --frame_images entry '{frame_arg}'. Expected 'frame_id:image_path'.")
        frame_id, image_path = frame_arg.split(':', 1)
        frame_paths[int(frame_id)] = image_path
    return frame_paths


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.prompt, "Please specify --prompt."
    assert args.image is not None or args.frame_images is not None, \
        "Please specify --image or --frame_images."
    assert args.size in SUPPORTED_SIZES[TASK], \
        f"Unsupported size {args.size}, supported sizes are: {', '.join(SUPPORTED_SIZES[TASK])}"

    cfg = WAN_CONFIGS[TASK]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    assert (args.frame_num - 1) % 4 == 0, "--frame_num must be 4n+1 (e.g. 49, 65, 81)."

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from one or more anchor frames with Wan2.2 I2V-A14B")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Path to the Wan2.2-I2V-A14B checkpoint directory.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="Target area (width*height). The aspect ratio follows the first anchor image.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="Number of output frames, must be 4n+1 (default: 81).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt describing the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single conditioning image placed at frame 0 (ignored if --frame_images is set).")
    parser.add_argument(
        "--frame_images",
        type=str,
        nargs='+',
        default=None,
        help="Anchor frames as 'frame_id:image_path'. Frame ids should be multiples of 4 "
        "(VAE temporal stride). Example: --frame_images 0:start.png 40:mid.png 80:end.png")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Output video path.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Random seed (-1 for a random seed).")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")

    # Memory / parallelism
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Offload the model to CPU after each forward to reduce GPU memory "
        "(default: True on a single GPU, False with multiple GPUs).")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Convert model parameters to bf16 (recommended on a single GPU).")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Place the T5 text encoder on CPU.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence parallel size (set to the number of GPUs).")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Use FSDP for T5 (multi-GPU only).")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Use FSDP for the DiT (multi-GPU only).")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel is not supported in non-distributed environments."

    cfg = WAN_CONFIGS[TASK]
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        init_distributed_group()

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    if args.frame_images is not None:
        frame_paths = _parse_frame_images(args.frame_images)
    else:
        frame_paths = {0: args.image}
    frame_images = {}
    for frame_id, image_path in sorted(frame_paths.items()):
        frame_images[frame_id] = Image.open(image_path).convert("RGB")
        logging.info(f"  Anchor frame {frame_id}: {image_path}")

    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logging.info("Generating video ...")
    video = wan_i2v.generate(
        args.prompt,
        None,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        frame_images=frame_images)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = f"{TASK}_{args.size.replace('*', 'x')}_{formatted_prompt}_{formatted_time}.mp4"

        save_dir = os.path.dirname(os.path.abspath(args.save_file))
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
