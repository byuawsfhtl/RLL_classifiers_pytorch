import torch
import pandas as pd
import os
import yaml
import sys
import csv
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
sys.path.append('/grphome/fslg_census/compute/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.model import select_model 
from src.get_data import DatasetCreator, Augmenter


class CustomException(Exception):
    """A custom exception class. Used to identify error with our scripts."""

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


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


def train_step(data_loader, augmenter, model, device, optimizer, loss_objective, accuracy_objective):
    loss_over_step = 0
    classification_accuracy = 0
    number_batches = len(data_loader)

    model.train()

    for images, labels in data_loader:
        images, labels = augmenter.augment_images(images, labels)
        images = images.to(torch.float32).to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_objective(predictions, labels)

        loss_over_step += loss.item()
        
        predicted_classes = torch.argmax(predictions, dim=1)
        classification_accuracy += accuracy_objective(predicted_classes.cpu(), labels.cpu()).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = loss_over_step / number_batches
    average_accuracy = classification_accuracy / number_batches

    return round(average_loss, 4), round(average_accuracy, 4)
    

def validation_step(data_loader, model, device, loss_objective, accuracy_objective):
    loss_over_step = 0
    classification_accuracy = 0
    number_batches = len(data_loader)

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(torch.float32).to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_objective(predictions, labels)

            loss_over_step += loss.item()
            
            predicted_classes = torch.argmax(predictions, dim=1)
            classification_accuracy += accuracy_objective(predicted_classes.cpu(), labels.cpu()).item()

    average_loss = loss_over_step / number_batches
    average_accuracy = classification_accuracy / number_batches

    return round(average_loss, 4), round(average_accuracy, 4)
    

def train(train_dataloader, val_dataloader, augmenter, model, device, model_name: str, optimizer, loss_objective, accuracy_objective, config: dict):
    
    epochs = config['model_hyper_parameters']['epochs']
    version = config['other_parameters']['version']
    early_stopping = config['other_parameters']['early_stopping']
    track_val_every_n_epochs = config['other_parameters']['track_val_every_n_epochs']
    model_weights_dir = config['paths']['model_weights_directory']
    metrics_dir_path = config['paths']['metrics_dir_path']
    metrics_filename = f'{model_name}_v{version}.tsv'
    
    metrics = []
    val_loss_scores = []

    initialize_metrics_file(metrics_dir_path, metrics_filename)

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(train_dataloader, augmenter, model, device, optimizer, loss_objective, accuracy_objective)

        if epoch % track_val_every_n_epochs == 0:
            val_loss, val_accuracy = validation_step(val_dataloader, model, device, loss_objective, accuracy_objective)
            
            metrics.append([epoch, train_loss, train_accuracy, val_loss, val_accuracy])
            val_loss_scores.append(val_loss)

            if early_stopping:
                if len(val_loss_scores) > 2:
                    if (val_loss_scores[-2] < val_loss_scores[-1]) and (val_loss_scores[-3] < val_loss_scores[-2]):
                        break
                    else:
                        new_model_name = f'{model_name}_v{version}.pt'
                        save_model_weights(model, config, model_weights_dir, new_model_name)
                        metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
                else:
                    new_model_name = f'{model_name}_v{version}.pt'
                    save_model_weights(model, config, model_weights_dir, new_model_name)
                    metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
            else:
                new_model_name = f'{model_name}_v{version}.pt'
                save_model_weights(model, config, model_weights_dir, new_model_name)
                metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
        else:
            metrics.append([epoch, train_loss, train_accuracy, None, None])


def save_model_weights(model, metadata_info, directory_path, model_weights_name):

    '''# Save the model state dictionary and metadata
        save_path = 'model_with_metadata.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)
    '''
    model_path = os.path.join(directory_path, model_weights_name)
    torch.save({'model_state_dict': model.state_dict(), 'metadata': metadata_info}, model_path)


def save_out_metrics(metrics_dir_path, metrics_filename, metrics):
    path_to_metric_file = os.path.join(metrics_dir_path, metrics_filename)

    with open(path_to_metric_file, 'a', newline='') as csv_out:
        writer = csv.writer(csv_out, delimiter='\t')
        writer.writerows(metrics)

    return []


def initialize_metrics_file(metrics_dir_path, metrics_filename):
    path_to_metric_file = os.path.join(metrics_dir_path, metrics_filename)

    with open(path_to_metric_file, 'w', newline='') as csv_out:
        writer = csv.writer(csv_out, delimiter='\t')
        writer.writerow(['Epoch', 'Train_loss', 'Train_accuracy', 'Test_loss', 'Test_accuracy'])


def main(config_file: str, normal_transforms, augment_transforms):
    config = load_config(config_file)
    
    batch_size = config['dataset_params']['batch_size']
    hold_images_in_RAM = config['dataset_params']['hold_images_in_RAM']
    inference = config['dataset_params']['inference']
    random_split = config['dataset_params']['random_split']
    stratified_split = config['dataset_params']['stratified_split']
    test_size = config['dataset_params']['test_size']
    model_name = config['model_architecture_parameters']['model_name'] 
    output_classes = config['model_architecture_parameters']['output_classes']
    optimizer_name = config['model_hyper_parameters']['optimizer_name']
    learning_rate = config['model_hyper_parameters']['learning_rate']
    weight_decay = config['model_hyper_parameters']['weight_decay']

    path_to_images_and_labels = config['paths']['image_paths_and_labels']
    df = pd.read_csv(path_to_images_and_labels, sep='\t')
    df.columns = ['path', 'label']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device, flush=True)

    dataset_creator = DatasetCreator()
    int_to_class_map = dataset_creator.get_int_to_class_map(df)
    config['int_to_class_map'] = int_to_class_map

    train_dataloader, val_dataloader = dataset_creator.get_dataset(df, inference, normal_transforms, batch_size, hold_images_in_RAM, random_split, stratified_split, test_size)

    augmenter = Augmenter(augment_transforms)

    model = select_model(model_name, output_classes, device)

    if config['paths']['transfer_learn_weights_file']:
        model_data = torch.load(config['paths']['transfer_learn_weights_file'], map_location='cpu')
        if 'model_state_dict' in model_data:
            model_state_dict = model_data['model_state_dict']
            metadata_dict = model_data['metadata']
            if int_to_class_map == metadata_dict['int_to_class_map'] and model_name == metadata_dict['model_architecture_parameters']['model_name']:
                model.load_state_dict(model_state_dict)
            else:
                raise CustomException("The number of output classes, the type of output classes or the model architecture doesn't match the model you're attempting to perform transfer learning with.")

        else:
            model.load_state_dict(model_data)


    config['paths']['model_weights_directory'] = os.path.join(config['paths']['model_weights_directory'], model_name)
    if not os.path.isdir(config['paths']['model_weights_directory']):
        os.makedirs(config['paths']['model_weights_directory'])

    optimizer = select_optimizer(model, optimizer_name, learning_rate, weight_decay=weight_decay)

    loss_objective = torch.nn.CrossEntropyLoss()
    accuracy_objective = MulticlassAccuracy(num_classes=output_classes)

    train(train_dataloader, val_dataloader, augmenter, model, device, model_name, optimizer, loss_objective, accuracy_objective, config)


if __name__ == '__main__':
    main()