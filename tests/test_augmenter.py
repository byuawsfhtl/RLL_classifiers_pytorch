from torch.utils.data import DataLoader
import sys
sys.path.append('/home/jroubido/fsl_groups/fslg_census/nobackup/archive/machine_learning_models/classification_models/branches/jackson/RLL_classifiers_pytorch/src')
from get_data import ImageDatasetInRAM, ImageDataset, Augmenter
import os
from torchvision.transforms import v2
import time
import random



if __name__ == '__main__':
    path_to_images = '/home/jroubido/fsl_groups/fslg_census/nobackup/archive/data/us_census/1930/raw/decompressed/15thcensus64uscers_jp2_gender_snippets'
    print('Getting paths and labels.', flush=True)
    list_of_images = [os.path.join(path_to_images, image_file) for image_file in os.listdir(path_to_images)]
    random.shuffle(list_of_images)
    list_of_images = list_of_images[:int(len(list_of_images)/10)]
    labels = [i for i in range(len(list_of_images))]
    print('list of images: ', len(list_of_images))

    normal_transforms = [v2.PILToTensor(), v2.Resize(size=(64, 64), antialias=True)]
    augment_transforms = [v2.Resize(size=(64, 64)), v2.ElasticTransform(alpha=50.0, sigma=10.0)]
    augmenter = Augmenter(augment_transforms)

    print('Creating dataset objects.', flush=True)
    dataset_not_in_ram = ImageDataset(list_of_images, labels, normal_transforms)
    print('Finished creating dataset objects.', flush=True)

    dataloader1 = DataLoader(dataset_not_in_ram, batch_size = 10, num_workers=2, prefetch_factor=4)
    

    print('Iterating through images in dl1.', flush=True)
    start_time = time.time()
    for images, labels in dataloader1:
        # print(len(batch))
        new_images, labels = augmenter.augment_images(images, labels)
        print('yeet')
    end_time = time.time()
    print('Time to get through all images with dataloader1: ', ((end_time-start_time)), flush=True)
