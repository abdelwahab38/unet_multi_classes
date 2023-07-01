import os
from glob import glob
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

#from lib.utils import rgb2mask
LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0], 3:[0,0,255]}
class CoreDataset(Dataset):
    
    def __init__(self, path, transform=None):
        self.path_images = natsort.natsorted(glob("E:\images_multi_classes"))
        self.path_masks = natsort.natsorted(glob("E:\labels_multi_classes"))
        self.transform = transform
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.path_images[idx])
        mask = Image.open(self.path_masks[idx])
        
        sample = {'image':image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
# add image normalization transform at some point
   
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels
        mask = rgb2mask(np.array(mask))
        sample = {'image': torch.from_numpy(image).permute(2,0,1).float(),
                  'mask': torch.from_numpy(mask).long(), 
                 }
        
        return sample
    
def make_datasets(path_images,path_masks):
    #dataset = CoreDataset(path, transform = transforms.Compose([ToTensor()]))
    #val_len = int(val_ratio*len(dataset))
    #lengths = [len(dataset)-val_len, val_len]
    #train_dataset, val_dataset = random_split(dataset, lengths)
    train_dataset = CoreDataset(path_images, transform = transforms.Compose([ToTensor()]))
    val_dataset = CoreDataset(path_masks, transform = transforms.Compose([ToTensor()]))
    return train_dataset, val_dataset


def make_dataloaders(path, val_ratio, params):
    train_dataset, val_dataset = make_datasets(path, val_ratio)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    
    return train_loader, val_loader
def rgb2mask(rgb):
    
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k,v in LABEL_TO_COLOR.items():
        mask[np.all(rgb==v, axis=2)] = k
        
    return mask