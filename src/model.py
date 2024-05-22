import torch
import torchvision.models as networks

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


def load_model(model_name: str, output_classes: int, path_to_weights: str):
    '''
    This function will load a model architecture with a defined number of output classes and will load existing weights
    for this given model. 

    Args:
        model_name: The name of a given model. 
        output_classes: The number of output classes for a given model. 
        path_to_weights: The file path to weights for the given model. 
    '''
    
    model = select_model(model_name, output_classes)
    model.load_state_dict(torch.load(path_to_weights, map_location='cpu'))
    return model