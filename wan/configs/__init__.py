# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B

WAN_CONFIGS = {
    'i2v-A14B': i2v_A14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
}
