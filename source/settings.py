BACKGROUND_PATH = 'background/background.jpg'
BACKGROUND_DIR = 'background'
OUTPUT_VIDEO_PATH = 'output_synopsis.mp4'
OUTPUT_FRAMES_DIR = 'frames'
OBJECT_IMAGE_PATH = "data_image"
OBJECT_COORDINATE_PATH = "data_coordinate"
DATA_OF_FRAME_DIR = "data_for_frame"


RESNET_RESIZE = 200
RESNET_CENTERCROP = 140
RESNET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
RESNET_NORMALIZE_STD = [0.229, 0.224, 0.225]
MAX_OBJECT = 10000



COSINE_SIMILARITY_THRESHOLD = 0.65
TEMPLATE_MATCHING_THRESHOLD = 0.04
COSINE_SIMILARITY_THRESHOLD_CHECKED = 0.75
MISSING_OBJECT_THRESHOLD = 8

TEMPLATE_CROP_RATIO = [0.15, 0.85]
OSCILLATE_RANGE_DICT = {"car": 120, "truck": 120, "person": 100, "default": 100}