import os

class ModelConfig:
    DEBUG = False
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 88

    # classes
    CLASSES = ['pig']
    NUM_CLASSES_TO_TRAIN = len(CLASSES)

    # training
    EPOCHS = 3 if DEBUG else 100  # 100
    BATCH_SIZE = 16

    BASE_MODEL = 'yolov9t'  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9c, yolov9e
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
    EXP_NAME = f'{EPOCHS}_epochs'

    OPTIMIZER = 'auto'  # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    LR = 1e-3
    LR_FACTOR = 0.01
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.01
    PATIENCE = 20
    PROFILE = False
    LABEL_SMOOTHING = 0.0
    DEVICE = 0

    # paths
    CUSTOM_DATASET_DIR = r"C:\Users\Administrator\Desktop\dataset"
    OUTPUT_PATH = "../output/"
    OUTPUT_DIR = "output"


class TrackingConfig:
    MODEL_WEIGHTS = os.path.join("..", "model_weight", "yolov9t_100_epochs.pt")

    SEQUENCE_LENGTH = 1000
    # detections (YOLO)
    CONFIDENCE = 0.1
    IOU = 0.5

    # heatmap (Supervision)
    HEATMAP_ALPHA = 0.5
    RADIUS = 30

    # tracking (Supervision, bytetrack)
    TRACK_THRESH = 0.35
    TRACK_SECONDS = 5
    MATCH_THRESH = 0.9999

    # paths: video file path, webcam is 0
    VIDEO_FILE = os.path.join("..", "video_demo", "pigs_video.mp4")
    OUTPUT_PATH = os.path.join("..", "output", "output.mp4")
    VIDEO_FILE_ZONE = os.path.join("..", "video_demo", "pigs_video.mp4")
    ZONE_CONFIGURATION_PATH = os.path.join("..", "data", "vertical-zone-config.json")


    # flags to turn functions on/off
    ENABLE_REALTIME_STREAMING = True
    ENABLE_HEATMAP = False
    ENABLE_COUNTING = False
    ENABLE_TRACE_ANNOTATOR = False

    DEVICE = 0

    CLASSES = [0]

    ZONE_COORDINATES = (100, 100, 300, 300)

    DELAY = 1
    SCALE = 250