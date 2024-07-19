import sys
sys.path.append('/grphome/fslg_census/nobackup/archive/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.trainer import start_training
from torchvision.transforms import v2

if __name__ == '__main__':
    image_size = (64, 64)
    normal_transforms = [v2.PILToTensor(), v2.Resize(size=image_size, antialias=True)]
    augment_transforms = None # [v2.ColorJitter(brightness=.5, contrast=.5), v2.ElasticTransform(alpha=50.0, sigma=10.0), v2.RandomPerspective(p=0.5)]
    path_to_config_file = '/grphome/fslg_census/nobackup/archive/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch/src/train_example.yaml'

    start_training(path_to_config_file, normal_transforms, augment_transforms)