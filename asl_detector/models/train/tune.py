import optuna
from asl_detector.models.mobilenetv2 import MobileNetV2Classifier
from asl_detector.data.dataloader import load_data

### Phase 1: Tune the learning rate + Dropout rate for the classifier head, with the backbone frozen.
def objective(trial):
    





### Phase 2: Unfreeze the last N layers of the backbone and tune N, learning rate, and dropout rate together.