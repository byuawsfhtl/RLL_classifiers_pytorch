import torch
import pandas as pd
import os
import yaml
import sys
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassAccuracy
sys.path.append('/grphome/fslg_census/compute/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.model import select_model 
from src.get_data import get_dataset


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None


def select_optimizer(model, optimizer_name: str, learning_rate: float, weight_decay=0):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_step(data_loader, model, optimizer, loss_objective, accuracy_objective):
    loss_over_step = 0
    classification_accuracy = 0
    number_batches = len(data_loader)

    for images, labels in data_loader:
        images = images.to(torch.float32)
        predictions = model(images)
        loss = loss_objective(predictions, labels)

        loss_over_step += loss.item()
        
        predicted_classes = torch.argmax(predictions, dim=1)
        classification_accuracy += accuracy_objective(predicted_classes, labels).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    average_loss = loss_over_step / number_batches
    average_accuracy = classification_accuracy / number_batches

    return average_loss, average_accuracy
    

def validation_step(data_loader, model, loss_objective, accuracy_objective):
    loss_over_step = 0
    classification_accuracy = 0
    number_batches = len(data_loader)

    for images, labels in data_loader:
        images = images.to(torch.float32)
        predictions = model(images)
        loss = loss_objective(predictions, labels)
        loss_over_step += loss.item()
        
        predicted_classes = torch.argmax(predictions, dim=1)
        classification_accuracy += accuracy_objective(predicted_classes, labels).item()

    average_loss = loss_over_step / number_batches
    average_accuracy = classification_accuracy / number_batches

    return average_loss, average_accuracy
    

def train(train_dataloader, val_dataloader, model, model_name: str, optimizer, loss_objective, accuracy_objective, telemetry: dict, file_paths: dict, model_hyper_parameters: dict, other_parameters: dict):
    epochs = model_hyper_parameters['epochs']
    version = other_parameters['version']
    early_stopping = other_parameters['early_stopping']
    track_val_every_n_epochs = other_parameters['track_val_every_n_epochs']
    model_weights_dir = file_paths['model_weights_directory']
    telemetry_dir_path = file_paths['telemetry_dir_path']

    for epoch in range(epochs):
        train_loss, train_accuracy, val_loss, val_accuracy = None, None, None, None
        train_loss, train_accuracy = train_step(train_dataloader, model, optimizer, loss_objective, accuracy_objective)
        if epoch % track_val_every_n_epochs == 0:
            val_loss, val_accuracy = validation_step(val_dataloader, model, loss_objective, accuracy_objective)
            if early_stopping:
                if len(telemetry) > 2:
                    if telemetry[-2]['val_loss_over_epochs'] < telemetry[-1]['val_loss_over_epochs']:
                        break
                    else:
                        new_model_name = f'{model_name}_v{version}.pt'
                        save_model_weights(model, model_weights_dir, new_model_name)
            else:
                new_model_name = f'{model_name}_v{version}.pt'
                save_model_weights(model, model_weights_dir, new_model_name)
        telemetry.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
        break
    telemetry_filename = f'{model_name}_v{version}.tsv'
    save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename)


def save_model_weights(model, directory_path, model_weights_name):
    model_path = os.path.join(directory_path, model_weights_name)
    torch.save(model.state_dict(), model_path)


def save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename):
    df_telemetry = pd.DataFrame(telemetry, columns=list(telemetry[0].keys()), index=None)
    full_telemetry_filepath = os.path.join(telemetry_dir_path, telemetry_filename)
    df_telemetry.to_csv(full_telemetry_filepath, index=False, sep='\t')


def create_telemetry_dict():
    telemetry = {'train_loss_over_epochs': [], 'val_loss_over_epochs': []}
    return telemetry


def main():
    config = load_config(sys.argv[1])
    
    batch_size = config['dataset_params']['batch_size']
    inference = config['dataset_params']['inference']
    random_split = config['dataset_params']['random_split']
    stratified_split = config['dataset_params']['stratified_split']
    test_size = config['dataset_params']['test_size']
    model_name = config['model_architecture_parameters']['model_name'] 
    output_classes = config['model_architecture_parameters']['output_classes']
    optimizer_name = config['model_hyper_parameters']['optimizer_name']
    learning_rate = config['model_hyper_parameters']['learning_rate']
    weight_decay = config['model_hyper_parameters']['weight_decay']
    
    file_paths = config['paths']
    model_hyper_parameters = config['model_hyper_parameters']
    other_parameters = config['other_parameters']
    # device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_to_images_and_labels = config['paths']['image_paths_and_labels']
    df = pd.read_csv(path_to_images_and_labels, sep='\t')
    df.columns = ['path', 'label']

    transforms = v2.Compose([v2.PILToTensor(), v2.Resize(size=(64, 256))])

    train_dataloader, val_dataloader = get_dataset(df, inference, transforms, batch_size, random_split, stratified_split, test_size)

    model = select_model(model_name, output_classes)

    config['paths']['model_weights_directory'] = os.path.join(config['paths']['model_weights_directory'], model_name)
    if not os.path.isdir(config['paths']['model_weights_directory']):
        os.makedirs(config['paths']['model_weights_directory'])

    optimizer = select_optimizer(model, optimizer_name, learning_rate, weight_decay=weight_decay)

    loss_objective = torch.nn.CrossEntropyLoss()
    accuracy_objective = MulticlassAccuracy(num_classes=output_classes)

    telemetry = []

    train(train_dataloader, val_dataloader, model, model_name, optimizer, loss_objective, accuracy_objective, telemetry, file_paths, model_hyper_parameters, other_parameters)


if __name__ == '__main__':
    main()