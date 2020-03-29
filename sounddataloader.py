import torchvision.datasets as dset
import librosa
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


NUMPY_EXTENSION = '.wav'


def default_loader(path):
    y, _ = librosa.load(path, sr=8000)
    # if y.shape[0] < int(8000 * 5.3):
    #     add_zeros = np.zeros(int(8000 * 5.3) - y.shape[0])
    #     y = np.concatenate([y, add_zeros])
    return y


class SoundDataLoader(dset.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(SoundDataLoader, self).__init__(root, loader, NUMPY_EXTENSION if is_valid_file is None else None,
                                              transform=transform,
                                              target_transform=target_transform,
                                              is_valid_file=is_valid_file)
        self.imgs = self.samples


def convert_to_mfcc(dataset):
    output = list()
    for file, target in tqdm(dataset):
        # mfcc = librosa.feature.mfcc(file, n_mfcc=40).mean(1)
        mfcc = librosa.feature.melspectrogram(file).mean(1)
        mfcc = normalize(mfcc)
        output.append([mfcc, target])
    return output


def normalize(data):
    if data.max() - data.min() == 0:
        return data
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data.astype(float)


def get_loader(root, batch_size):
    dataset = SoundDataLoader(root=root)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_dataset = convert_to_mfcc(train_dataset)
    val_dataset = convert_to_mfcc(val_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)

    return train_loader, val_loader