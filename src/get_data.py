import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2


class ImageDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        
        if self.labels:
            return image, self.labels[index]
        else:
            return image


class ImageDatasetInRAM(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        if self.transform:
            self.images = [self.transform(default_loader(path)) for path in self.paths]
        else:
            self.images = [default_loader(path) for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.labels:
            return self.images[index], self.labels[index]
        else:
            return self.images[index]


class DatasetCreator():

    def __init__(self):
        pass

    def convert_classes_to_int(self, df: pd.DataFrame):
        unique_classes = df['label'].unique()
        class_to_int = {}

        for i, class_ in enumerate(sorted(unique_classes)):
            class_to_int[class_] = i

        df['class_index'] = df['label'].map(lambda x: class_to_int.get(x))


    def stratified_sampling(self, labels_and_paths: pd.DataFrame, random_state: int, test_size: float):
        training_paths, training_labels, testing_paths, testing_labels = [], [], [], []
                
        grouped = labels_and_paths.groupby('class_index')

        for field, group_data in grouped:
            group_data = group_data.sample(frac=1)
            train_paths, test_paths, train_labels, test_labels = train_test_split(group_data['path'], group_data['class_index'], test_size=test_size, random_state=random_state)
            training_paths += train_paths.tolist()
            training_labels += train_labels.tolist()
            testing_paths += test_paths.tolist()
            testing_labels += test_labels.tolist()

        return training_paths, testing_paths, training_labels, testing_labels


    def get_train_and_test_sets(self, labels_and_paths: pd.DataFrame, random_split: bool, stratified_split: bool, test_size: float):
        if random_split:
            if stratified_split:
                return self.stratified_sampling(labels_and_paths, None, test_size)
            else:
                labels_and_paths = labels_and_paths.sample(frac=1)
                training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['label'], test_size=test_size)
                return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()
        else:
            if stratified_split:
                return self.stratified_sampling(labels_and_paths, 42, test_size)
            else:
                labels_and_paths = labels_and_paths.sample(frac=1)
                training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['label'], test_size=test_size, random_state=42)
                return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()


    def get_dataset(self, labels_and_paths: pd.DataFrame, inference: bool, transforms, batch_size: int, hold_images_in_RAM: bool = False, random_split: bool = None, stratified_split: bool = None, test_size: float = None):
        self.convert_classes_to_int(labels_and_paths)

        if inference:
            paths, labels = labels_and_paths['path'], labels_and_paths['class_index']
            dataset = ImageDataset(paths, labels, transforms)
            dataloader = DataLoader(dataset, batch_size = batch_size)
            return dataloader
        else:
            training_paths, testing_paths, training_labels, testing_labels = self.get_train_and_test_sets(labels_and_paths, random_split, stratified_split, test_size)
            
            if hold_images_in_RAM:
                train_dataset = ImageDatasetInRAM(training_paths, training_labels, transforms)
                test_dataset = ImageDatasetInRAM(testing_paths, testing_labels, transforms)
                train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
                return train_dataloader, test_dataloader
            else:
                train_dataset = ImageDataset(training_paths, training_labels, transforms)
                test_dataset = ImageDataset(testing_paths, testing_labels, transforms)
                train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
                return train_dataloader, test_dataloader


class Augmenter():
    '''
        This class is meant to be used with the TrainDataset class. After an image has been read into memory, this script will create augmented copies of that image
        to help train the model. See the function doc strings for more information.  
    '''

    def __init__(self, transforms: list):
        '''
            Args:
                transforms: This is a list of torchvision transforms from the v2 library. Ie: [v2.Resize(size=(64, 64)), v2.ElasticTransform(alpha=50.0, sigma=10.0)]
                augmentation_ratio: This is an int value that should be greater than 0. Ie: 3 would mean that for each image we would make 3 augmented copies of the image to train the model with. 
        '''
        self.transforms = transforms
        if self.transforms:
            self.transforms_length = len(transforms)
        else:
            self.transforms_length = 0

    def augment_images(self, images, labels):
        '''
            This is the main function of the Augmenter class. It takes in a tensor of shape: B x C x H x W and returns a
            2 tensors: the augmented image copies and copies of the original image to use as targets. The returned tensors are of size (self.augmentation_ratio*B x C x H x W).

            Args:
                images: A tensor object that is a batch of images. 
        '''
        if self.transforms_length > 0:
            augmented_images = self.random_apply_augment_transforms(images)
            normal_and_augmented_images = torch.cat([augmented_images, images], dim=0)
            normal_and_augmented_labels = torch.cat([labels, labels], dim=0)

            return normal_and_augmented_images, normal_and_augmented_labels

        else:
            return images, labels

    def random_apply_augment_transforms(self, images):
        '''
            This function performs the transforms on our batch of images. The transforms are applied randomly to facilitate model generalization. 

            Args:
                images: A tensor object that is a batch of images. 
        '''
        number_of_transforms = random.randint(1, self.transforms_length)
        sampled_transforms = random.sample(self.transforms, number_of_transforms)
        transforms_ = v2.Compose(sampled_transforms)
        return transforms_(images)