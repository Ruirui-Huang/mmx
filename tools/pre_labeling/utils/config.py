from enum import Enum

class TASK_TYPE(Enum):
    DET = 'det'
    SEG = 'seg'
    POSE = 'pose'

class ModelType(Enum):
    YOLOV5 = 'yolov5'
    YOLOX = 'yolox'
    PPYOLOE = 'ppyoloe'
    PPYOLOEP = 'ppyoloep'
    YOLOV6 = 'yolov6'
    YOLOV7 = 'yolov7'
    RTMDET = 'rtmdet'
    YOLOV8 = 'yolov8'
