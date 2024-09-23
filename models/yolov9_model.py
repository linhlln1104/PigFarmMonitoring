from ultralytics import YOLO
import os
from utils.data_utils import get_image_properties
from torch.amp import GradScaler
def train_model(ModelConfig):

    model = YOLO(ModelConfig.BASE_MODEL_WEIGHTS)

    example_image_path = os.path.join(ModelConfig.CUSTOM_DATASET_DIR, 'train', 'images',
                                      '20240830_051510_19082024_frame_0002_jpg.rf.73eabd629907809420c0ec3fbd94f3c5.jpg')
    img_properties = get_image_properties(example_image_path)
    # Train the model
    model.train(
        data=os.path.join(ModelConfig.OUTPUT_DIR, 'data.yaml'),
        task='detect',
        imgsz=(img_properties['height'], img_properties['width']),
        epochs=ModelConfig.EPOCHS,
        batch=ModelConfig.BATCH_SIZE,
        optimizer=ModelConfig.OPTIMIZER,
        lr0=ModelConfig.LR,
        lrf=ModelConfig.LR_FACTOR,
        weight_decay=ModelConfig.WEIGHT_DECAY,
        dropout=ModelConfig.DROPOUT,
        fraction=ModelConfig.FRACTION,
        patience=ModelConfig.PATIENCE,
        profile=ModelConfig.PROFILE,
        label_smoothing=ModelConfig.LABEL_SMOOTHING,
        name=f'{ModelConfig.BASE_MODEL}_{ModelConfig.EXP_NAME}',
        seed=ModelConfig.SEED,
        val=True,
        amp=True,
        exist_ok=True,
        resume=False,
        device=ModelConfig.DEVICE,
        verbose=False,
    )

    # Export the model
    model.export(
        format='onnx',  # openvino, onnx, engine, tflite
        imgsz=(img_properties['height'], img_properties['width']),
        half=False,
        int8=False,
        simplify=False,
        nms=False,
    )