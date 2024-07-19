import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2


class ImageDataset(Dataset):
    '''
    This is a vanilla dataset class that inherits from pytorch's Dataset class. It pulls images from disk to feed to the model. 
    '''
    def __init__(self, paths: list, labels: list = None, transforms: list = None):
        '''
        Args:
            paths: A python list that contains paths to images to train with. 
            labels: A list of integers that correspond to a given class in our dataset. 
            transforms: A list of torchvision transforms. 
        '''
        self.paths = paths
        self.labels = labels
        self.transforms = v2.Compose(transforms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.labels:
            return image, self.labels[index]
        else:
            return image


class ImageDatasetInRAM(Dataset):
    '''
    This is a vanilla dataset class that inherits from pytorch's Dataset class. It pulls holds all images in RAM to feed to the model. 
    '''
    def __init__(self, paths: list, labels: list = None, transforms: list = None):
        '''
        Args:
            paths: A python list that contains paths to images to train with. 
            labels: A list of integers that correspond to a given class in our dataset. 
            transforms: A list of torchvision transforms. 
        '''
        self.paths = paths
        self.labels = labels
        self.transforms = v2.Compose(transforms)
        if self.transforms:
            self.images = [self.transforms(default_loader(path)) for path in self.paths]
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
    '''
    This class is made to create a dataset. Its main function is get_dataloader(). 
    '''

    def convert_classes_to_int(self, df: pd.DataFrame):
        '''
        This function takes in a dataframe that contains two columns: paths and labels. 
        It then creates a third column of integers where each integer corresponds to a label in the dataset.

        Args:
            df: needed columns: paths, labels 
        '''
        class_to_int = self.get_class_to_int_map(df)

        df['class_index'] = df['label'].map(lambda x: class_to_int.get(x))


    def get_class_to_int_map(self, df: pd.DataFrame):
        '''
        This function finds all the unique labels in a dataset and creates a corresponding integer for the class
        so that the classifier can learn what images go to which class. 

        Args:
            df: needed columns: paths, labels
        '''
        unique_classes = df['label'].unique()
        class_to_int = {}

        for i, class_ in enumerate(sorted(unique_classes)):
            class_to_int[class_] = i

        return class_to_int


    def get_int_to_class_map(self, df: pd.DataFrame):
        '''
        This function creates a dictionary that maps integer values to a given class. It's quite similar to get_class_to_int_map()

        Args:
            df: needed columns: paths, labels
        '''
        unique_classes = df['label'].unique()
        class_to_int = {}

        for i, class_ in enumerate(sorted(unique_classes)):
            class_to_int[i] = class_

        return class_to_int


    def stratified_sampling(self, labels_and_paths: pd.DataFrame, randomize: bool, test_size: float):
        '''
        This function performs a stratified sampling technique for the train and test sets.

        Args:
            labels_and_paths: This is a pandas dataframe. Needed columns: paths, labels, class_index (this is obtained prior to entering this function)
            random_state: This is a seed for the random function generator
            test_size: This is a value between 0 and 1. If you want an 80/20 split for training and testing, then the value would be .2
        '''
        training_paths, training_labels, testing_paths, testing_labels = [], [], [], []
                
        grouped = labels_and_paths.groupby('class_index')

        for field, group_data in grouped:
            group_data = group_data.sample(frac=1)
            train_paths, test_paths, train_labels, test_labels = train_test_split(group_data['path'], group_data['class_index'], test_size=test_size, shuffle=randomize)
            training_paths += train_paths.tolist()
            training_labels += train_labels.tolist()
            testing_paths += test_paths.tolist()
            testing_labels += test_labels.tolist()

        return training_paths, testing_paths, training_labels, testing_labels


    def get_train_and_test_sets(self, labels_and_paths: pd.DataFrame, random_split: bool, stratified_split: bool, test_size: float):
        '''
        This function gets paths and labels (class_index) for training and testing. 

        Args:
            labels_and_paths: Pandas Dataframe. Needed columns: paths, labels, class_index (this is obtained prior to entering this function)
            random_split: This is a flag that determines if we want our dataset to be shuffled before creating our train and test sets
            stratified_split: This is a flag that determines if we want our train and test sets to have a consistent proportion of data for each label. 
            test_size: This is a value between 0 and 1. If you want an 80/20 split for training and testing, then the value would be .2
        '''
        if random_split:
            if stratified_split:
                return self.stratified_sampling(labels_and_paths, True, test_size)
            else:
                labels_and_paths = labels_and_paths.sample(frac=1)
                training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['class_index'], test_size=test_size)
                return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()
        else:
            if stratified_split:
                return self.stratified_sampling(labels_and_paths, False, test_size)
            else:
                training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['class_index'], test_size=test_size, shuffle=False)
                return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()


    def get_dataloader(self, labels_and_paths: pd.DataFrame, inference: bool, transforms: list, batch_size: int, hold_images_in_RAM: bool = False, random_split: bool = None, stratified_split: bool = None, test_size: float = None):
        '''
        This is the main function of the DatasetCreator class. It gets the datasets that the user will need for training. 

        Args:
            labels_and_paths: Pandas Dataframe. Needed columns: paths, labels, class_index (this is obtained prior to entering this function)
            inference: Flag that determines whether the dataset is being created for inference or training 
            transforms: A list of torchvision transforms for when data is loaded in via the dataset
            batch_size: An integer value that represents how many images will be drawn from the dataloader
            hold_images_in_RAM: Flag that determines if our images should all be held in RAM or if they'll be read from disk. (RAM is typically faster, ideal for smaller datasets.) 
            random_split: This is a flag that determines if we want our dataset to be shuffled before creating our train and test sets
            stratified_split: This is a flag that determines if we want our train and test sets to have a consistent proportion of data for each label. 
            test_size: This is a value between 0 and 1. If you want an 80/20 split for training and testing, then the value would be .2
        '''
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
        2 tensors: the augmented image copies and copies of the original image to use as targets. The returned tensors are of size (2B x C x H x W) if self.transforms is not None and 
        (B x C x H x W) if self.transforms is None.

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