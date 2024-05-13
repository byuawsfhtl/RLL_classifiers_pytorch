import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader


def convert_classes_to_int(df: pd.DataFrame):
    unique_classes = df['label'].unique()
    class_to_int = {}

    for i, class_ in enumerate(sorted(unique_classes)):
        class_to_int[class_] = i

    df['class_index'] = df['label'].map(lambda x: class_to_int.get(x))


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


def stratified_sampling(labels_and_paths: pd.DataFrame, random_state: int, test_size: float):
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


def get_train_and_test_sets(labels_and_paths: pd.DataFrame, random_split: bool, stratified_split: bool, test_size: float):
    if random_split:
        if stratified_split:
            return stratified_sampling(labels_and_paths, None, test_size)
        else:
            labels_and_paths = labels_and_paths.sample(frac=1)
            training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['label'], test_size=test_size)
            return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()
    else:
        if stratified_split:
            return stratified_sampling(labels_and_paths, 42, test_size)
        else:
            labels_and_paths = labels_and_paths.sample(frac=1)
            training_paths, testing_paths, training_labels, testing_labels = train_test_split(labels_and_paths['path'], labels_and_paths['label'], test_size=test_size, random_state=42)
            return training_paths.tolist(), testing_paths.tolist(), training_labels.tolist(), testing_labels.tolist()


def get_dataset(labels_and_paths: pd.DataFrame, inference: bool, transforms, batch_size: int, random_split: bool = None, stratified_split: bool = None, test_size: float = None):
    convert_classes_to_int(labels_and_paths)

    if inference:
        paths, labels = labels_and_paths['path'], labels_and_paths['class_index']
        dataset = ImageDataset(paths, labels, transforms)
        dataloader = DataLoader(dataset, batch_size = batch_size)
        return dataloader
    else:
        training_paths, testing_paths, training_labels, testing_labels = get_train_and_test_sets(labels_and_paths, random_split, stratified_split, test_size)
        train_dataset = ImageDataset(training_paths, training_labels, transforms)
        test_dataset = ImageDataset(testing_paths, testing_labels, transforms)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
        return train_dataloader, test_dataloader