import torch
from torchvision.transforms import v2
from torchvision.datasets.folder import default_loader
import sys
import os
sys.path.append('/grphome/fslg_census/compute/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.model import select_model 


def get_images_to_tensor(image_paths, transforms):

    list_of_tensors = []

    for image_path in image_paths:
        list_of_tensors.append(transforms(default_loader(image_path)))

    return torch.stack(list_of_tensors)


def main():

    path_to_model_weights = '/home/jroubido/fsl_groups/fslg_census/compute/projects/US_Census/Census_Linking_Code/branches/Jackson/RLL_US_census_linking/1930/gender_classification/weights/resnet50/resnet50_v4.pt'
    

    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_weights_and_metadata = torch.load(path_to_model_weights, map_location='cpu')
    model_state_dict = model_weights_and_metadata['model_state_dict']
    metadata_dict = model_weights_and_metadata['metadata']

    int_to_class_map = metadata_dict['int_to_class_map']
    model_name = metadata_dict['model_architecture_parameters']['model_name'] 
    number_of_output_classes = metadata_dict['model_architecture_parameters']['output_classes']


    model = select_model(model_name, number_of_output_classes, device)
    
    model.load_state_dict(model_state_dict)

    model.eval()

    dir_of_images = '/grphome/fslg_census/compute/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch/tests/resources/gender_snippets'

    image_paths = [os.path.join(dir_of_images, image_file) for image_file in os.listdir(dir_of_images)]

    transforms = v2.Compose([v2.PILToTensor(), v2.Resize(size=(64, 64))])

    batch_of_images = get_images_to_tensor(image_paths, transforms).to(torch.float32)
    print(batch_of_images.dtype)

    output = model(batch_of_images)

    predicted_classes = torch.argmax(output, dim=1)

    for class_, image_path in zip(predicted_classes, image_paths):
        print('image_path: ', image_path.split('/')[-1])
        print('predicted letter: ', int_to_class_map[class_.item()])

    print('yeet')

    
if __name__ == '__main__':
    main()