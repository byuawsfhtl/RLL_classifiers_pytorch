import torch
import torchvision.models as networks
from custom_exception import CustomException

def select_model(model_name: str, output_classes: int, device: str):
    '''
    This function takes in a model name as input and returns a model with other specified parameters.

    Args:
        model_name: The name of a given model. 
        output_classes: The number of output classes for a given model. 
    '''

    if model_name == 'vgg11':
        return networks.vgg11(num_classes=output_classes).to(device)
    elif model_name == 'vgg16':
        return networks.vgg16(num_classes=output_classes).to(device)
    elif model_name == 'vgg19':
        return networks.vgg19(num_classes=output_classes).to(device)
    elif model_name == 'resnet18':
        return networks.resnet18(num_classes=output_classes).to(device)
    elif model_name == 'resnet34':
        return networks.resnet34(num_classes=output_classes).to(device)
    elif model_name == 'resnet50':
        return networks.resnet50(num_classes=output_classes).to(device)
    elif model_name == 'resnet101':
        return networks.resnet101(num_classes=output_classes).to(device)
    elif model_name == 'resnet152':
        return networks.resnet152(num_classes=output_classes).to(device)
    elif model_name == 'densenet121':
        return networks.densenet121(num_classes=output_classes).to(device)
    elif model_name == 'densenet161':
        return networks.densenet161(num_classes=output_classes).to(device)
    elif model_name == 'densenet201':
        return networks.densenet201(num_classes=output_classes).to(device)


def load_model(model_name: str, output_classes: int, int_to_class_map: dict, path_to_weights: str, device: str):
    '''
    This function will load a model architecture with a defined number of output classes and will load existing weights
    for this given model. 

    Args:
        model_name: The name of a given model. 
        output_classes: The number of output classes for a given model. 
        path_to_weights: The file path to weights for the given model. 
    '''
    
    model = select_model(model_name, output_classes)
    model_data = torch.load(path_to_weights, map_location='cpu')
    
    if 'model_state_dict' in model_data:
        model_state_dict = model_data['model_state_dict']
        metadata_dict = model_data['metadata']
        if int_to_class_map == metadata_dict['int_to_class_map'] and model_name == metadata_dict['model_architecture_parameters']['model_name']:
            model.load_state_dict(model_state_dict)
        else:
            raise CustomException("The number of output classes, the type of output classes or the model architecture doesn't match the model you're attempting to perform transfer learning with.")
    else:
        model.load_state_dict(model_data)

    return model.to(device)