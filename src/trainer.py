import torch
import pandas as pd
import os
import yaml
import sys
import csv
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
sys.path.append('/grphome/fslg_census/nobackup/archive/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.model import select_model, load_model
from src.get_data import DatasetCreator, Augmenter
from src.custom_exception import CustomException


def load_config(config_file: str):
    '''
    This function loads a .yaml file for the script to use. 
    '''
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None


def select_optimizer(model, optimizer_name: str, learning_rate: float, weight_decay=0):
    '''
    This function can be extended. It currently supports return the adam optimizer or classic gradient descent. 
    '''
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise CustomException("Invalid optimizer selected. See select_optimizer function for more details.")


def train_step(data_loader, augmenter, model, device, optimizer, loss_objective, accuracy_objective):
    '''
    This function defines a step in the training process. 

    Args:
        data_loader: A Pytorch Dataloader object. It holds the images for the model to train on
        augmenter: An augmenter object (custom class), it performs transformations to the image data to help enrich the training process
        model: A pytorch model that we are training 
        device: This defines where the model and data should be: 'cpu' or 'cuda'
        optimizer: An optimizer object that will perform gradient descent (or another optimization technique) to help the model learn
        loss_objective: Our loss function to define how our model is performing and to help the model learn
        accuracy_objective: A function that scores the model's accuracy
    '''
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
    early_stopping = config['other_parameters']['early_stopping']
    track_val_every_n_epochs = config['other_parameters']['track_val_every_n_epochs']
    model_weights_dir = config['paths']['model_weights_directory']
    metrics_dir_path = config['paths']['metrics_dir_path']

    version = 1
    model_file_name = f'{model_name}_v{version}.pt'
    model_path = os.path.join(model_weights_dir, model_file_name)

    if not os.path.exists(metrics_dir_path):
        os.makedirs(model_weights_dir)

    while os.path.exists(model_path):
        version += 1 
        model_file_name = f'{model_name}_v{version}.pt'
        model_path = os.path.join(model_weights_dir, model_file_name)
       
        
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
            config['epochs_trained_so_far'] = epoch

            if early_stopping:
                if len(val_loss_scores) > 2:
                    if (val_loss_scores[-2] < val_loss_scores[-1]) and (val_loss_scores[-3] < val_loss_scores[-2]):
                        break
                    else:
                        save_model_weights(model, config, model_path)
                        metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
                else:
                    save_model_weights(model, config, model_path)
                    metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
            else:
                save_model_weights(model, config, model_path)
                metrics = save_out_metrics(metrics_dir_path, metrics_filename, metrics)
        else:
            metrics.append([epoch, train_loss, train_accuracy, None, None])

def save_model_weights(model, metadata_info, model_path):

    '''# Save the model state dictionary and metadata
        save_path = 'model_with_metadata.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)
    '''
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


def start_training(config_file: str, normal_transforms, augment_transforms):
    config = load_config(config_file)
    
    batch_size = config['dataset_params']['batch_size']
    hold_images_in_RAM = config['dataset_params']['hold_images_in_RAM']
    inference = config['dataset_params']['inference']
    random_split = config['dataset_params']['random_split']
    random_state = config['dataset_params']['random_state']
    stratified_split = config['dataset_params']['stratified_split']
    test_size = config['dataset_params']['test_size']
    model_name = config['model_architecture_parameters']['model_name'] 
    output_classes = config['model_architecture_parameters']['output_classes']
    optimizer_name = config['model_hyper_parameters']['optimizer_name']
    learning_rate = config['model_hyper_parameters']['learning_rate']
    weight_decay = config['model_hyper_parameters']['weight_decay']
    device = config['model_hyper_parameters']['device']

    path_to_images_and_labels = config['paths']['image_paths_and_labels']
    df = pd.read_csv(path_to_images_and_labels, sep='\t')
    df.columns = ['path', 'label']

    print('Using device: ', device, flush=True)

    dataset_creator = DatasetCreator()
    int_to_class_map = dataset_creator.get_int_to_class_map(df)
    config['int_to_class_map'] = int_to_class_map

    train_dataloader, val_dataloader = dataset_creator.get_dataloader(df, inference, normal_transforms, batch_size, hold_images_in_RAM, random_split, random_state, stratified_split, test_size)

    augmenter = Augmenter(augment_transforms)

    if config['paths']['transfer_learn_weights_file']:
        model = load_model(model_name, output_classes, int_to_class_map, config['paths']['transfer_learn_weights_file'], device)
    else:
        model = select_model(model_name, output_classes, device)

    if not os.path.isdir(config['paths']['model_weights_directory']):
        os.makedirs(config['paths']['model_weights_directory'])

    optimizer = select_optimizer(model, optimizer_name, learning_rate, weight_decay=weight_decay)

    loss_objective = torch.nn.CrossEntropyLoss()
    accuracy_objective = MulticlassAccuracy(num_classes=output_classes)

    train(train_dataloader, val_dataloader, augmenter, model, device, model_name, optimizer, loss_objective, accuracy_objective, config)
