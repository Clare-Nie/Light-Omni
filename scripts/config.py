DATASET_CONVERSATION_PATH = "data.json"
INPUT_DATA_PATH = "input"
RAW_DIALOGUE_DATA_PATH = "raw"

START_TIME = "2026-01-01 08:00:00" # default
BACKGROUND_INTERVAL = 30 # seconds
MIN_BACKGROUND_BEFORE_SPEECH = 20
FRAME_DURATION = 1.0
VIDEO_OUPUT_MODE = "image"

class RetrieverConfig:
    RETRIEVE_RAW_DATA = True
    TOP_K_SEMANTIC = 12
    TOP_K_EPISODIC = 4


class FaceConfig:
    PROFILE_PATH = "profiles"
    PROFILE_DATA_PATH = "profiles/data"
    PROFILE_AVATAR_PATH = "profiles/avatars"
    PROFILE_META_JSON = "profiles.json"
    PROFILE_INDEX_FILE = "faces.index"
    PROFILE_VECTORS_FILE = "vectors.npy"

DATA_QUESTION_NUM = 5
USE_DOUBAO = False



