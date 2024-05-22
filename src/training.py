import torch
import pandas as pd
import os
import yaml
import sys
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
sys.path.append('/grphome/fslg_census/compute/machine_learning_models/classification_models/branches/main/RLL_classifiers_pytorch')
from src.model import select_model 
from src.get_data import DatasetCreator, Augmenter


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
    

def train(train_dataloader, val_dataloader, augmenter, model, device, model_name: str, optimizer, loss_objective, accuracy_objective, telemetry: dict, file_paths: dict, model_hyper_parameters: dict, other_parameters: dict):
    epochs = model_hyper_parameters['epochs']
    version = other_parameters['version']
    early_stopping = other_parameters['early_stopping']
    track_val_every_n_epochs = other_parameters['track_val_every_n_epochs']
    model_weights_dir = file_paths['model_weights_directory']
    telemetry_dir_path = file_paths['telemetry_dir_path']
    telemetry_filename = f'{model_name}_v{version}.tsv'

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(train_dataloader, augmenter, model, device, optimizer, loss_objective, accuracy_objective)
        telemetry['train'].append((epoch, train_loss, train_accuracy))

        if epoch % track_val_every_n_epochs == 0:
            val_loss, val_accuracy = validation_step(val_dataloader, model, device, loss_objective, accuracy_objective)
            telemetry['val'].append((epoch, val_loss, val_accuracy))

            if early_stopping:
                if len(telemetry['val']) > 2:
                    if (telemetry['val'][-2][1] < telemetry['val'][-1][1]) and (telemetry['val'][-3][1] < telemetry['val'][-2][1]):
                        break
                    else:
                        new_model_name = f'{model_name}_v{version}.pt'
                        save_model_weights(model, model_weights_dir, new_model_name)
                        save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename)
                else:
                    new_model_name = f'{model_name}_v{version}.pt'
                    save_model_weights(model, model_weights_dir, new_model_name)
                    save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename)
            else:
                new_model_name = f'{model_name}_v{version}.pt'
                save_model_weights(model, model_weights_dir, new_model_name)
                save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename)


def save_model_weights(model, directory_path, model_weights_name):
    model_path = os.path.join(directory_path, model_weights_name)
    torch.save(model.state_dict(), model_path)


def save_out_telemetry(telemetry, telemetry_dir_path, telemetry_filename):
    list_of_lists = []

    for (epoch, train_loss, train_accuracy) in telemetry['train']:
        if len(telemetry['val']) > 0:
            if epoch == telemetry['val'][0][0]:
                list_of_lists.append([epoch, train_loss, train_accuracy, telemetry['val'][0][1], telemetry['val'][0][2]])
                telemetry['val'].pop(0)
            else:
                list_of_lists.append([epoch, train_loss, train_accuracy, None, None])
        else:
            list_of_lists.append([epoch, train_loss, train_accuracy, None, None])

    df_telemetry = pd.DataFrame(list_of_lists, columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'], index=None)
    full_telemetry_filepath = os.path.join(telemetry_dir_path, telemetry_filename)
    df_telemetry.to_csv(full_telemetry_filepath, index=False, sep='\t')


def create_telemetry_dict():
    telemetry = {'train': [], 'val': []}
    return telemetry


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
    
    file_paths = config['paths']
    model_hyper_parameters = config['model_hyper_parameters']
    other_parameters = config['other_parameters']

    path_to_images_and_labels = config['paths']['image_paths_and_labels']
    df = pd.read_csv(path_to_images_and_labels, sep='\t')
    df.columns = ['path', 'label']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device, flush=True)

    train_dataloader, val_dataloader = DatasetCreator().get_dataset(df, inference, normal_transforms, batch_size, hold_images_in_RAM, random_split, stratified_split, test_size)

    augmenter = Augmenter(augment_transforms)

    model = select_model(model_name, output_classes, device)

    config['paths']['model_weights_directory'] = os.path.join(config['paths']['model_weights_directory'], model_name)
    if not os.path.isdir(config['paths']['model_weights_directory']):
        os.makedirs(config['paths']['model_weights_directory'])

    optimizer = select_optimizer(model, optimizer_name, learning_rate, weight_decay=weight_decay)

    loss_objective = torch.nn.CrossEntropyLoss()
    accuracy_objective = MulticlassAccuracy(num_classes=output_classes)

    telemetry = create_telemetry_dict()

    train(train_dataloader, val_dataloader, augmenter, model, device, model_name, optimizer, loss_objective, accuracy_objective, telemetry, file_paths, model_hyper_parameters, other_parameters)


if __name__ == '__main__':
    main()