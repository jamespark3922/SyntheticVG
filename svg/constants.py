CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Tokenizer Constants
BOX_START="<box>"
BOX_END="</box>"
POINT_START="<point>"
POINT_END="</point>"
PHRASE_START="<phrase>"
PHRASE_END="</phrase>"

# Qwen2 Constants
BOX_START_QWEN2="<|box_start|>"
BOX_END_QWEN2="<|box_end|>"
PHRASE_START_QWEN2="<|object_ref_start|>"
PHRASE_END_QWEN2="<|object_ref_end|>"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

PATCH_SIZE = 16