import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from skimage import io

import numpy as np
import os
import pickle


def get_imagenet_train_val_loaders(batch_size=128,
                                   val_perc=0.1,
                                   data_root=None,
                                   num_workers=1,
                                   pin_memory=True,
                                   shuffle=True,
                                   random_seed=None):
    '''  Returns iterators through train/val CIFAR10 datasets.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param batch_size: int
        How many samples per batch to load.
    :param val_perc: float
        Percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the train/validation indices
    :param random_seed: int
        Fix seed for reproducibility.
    :return:
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

    traindir = os.path.join(data_root, 'train')

    train_dataset = datasets.ImageFolder(traindir, transform_train)
    val_dataset = datasets.ImageFolder(traindir, transform_test)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_perc * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              drop_last=True, pin_memory=pin_memory, )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                            drop_last=True, pin_memory=pin_memory, )

    return train_loader, val_loader


def get_imagenet_test_loader(batch_size=128, data_root=None, num_workers=1, pin_memory=True, shuffle=False):
    ''' Returns iterator through CIFAR10 test dataset

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    :param batch_size: int
        How many samples per batch to load.
    :param data_root: str
        Path to the directory with the dataset.
    :param num_workers: int
        Number of subprocesses to use when loading the dataset.
    :param pin_memory: bool
        Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    :param shuffle: bool
        Whether to shuffle the dataset after every epoch.
    :return:
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

    test_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             drop_last=True, pin_memory=pin_memory)
    return test_loader



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
            

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

    
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, 
    

def get_training_dataloader(data_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(data_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader