import os, sys, random
import tarfile
from pathlib import Path
from time import gmtime, strftime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import tempfile
import shutil
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image

if sys.version_info[0] == 2:
    from urllib import urlretrieve
    import cPickle as pickle

else:
    from urllib.request import urlretrieve
    import pickle
    
    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)




def read_alphabets(alphabet_directory_path):
    """
    Reads all the characters from a given alphabet_directory
    Args:
      alphabet_directory_path (str): path to directory with files
    Returns:
      datax (np.array): array of path names of images
      datay (np.array): array of labels
    """
    datax = []
    datay = []
    # ----- Временная директория ----- # 
    temp_dir = tempfile.mkdtemp()  
    

    for char_dir in sorted(os.listdir(alphabet_directory_path)):
        char_path = os.path.join(alphabet_directory_path, char_dir)
        if not os.path.isdir(char_path):
            continue
            
        for img_file in sorted(os.listdir(char_path)):
            if not img_file.endswith('.png'):
                continue
                
            img_path = os.path.join(char_path, img_file)
            img = Image.open(img_path)
            

            original_path = os.path.join(temp_dir, f"{char_dir}_{img_file}_0.png")
            img.save(original_path)
            datax.append(original_path)
            datay.append(char_dir + '_0')
            
            # ----- ЦИКЛ С ПОВОРОТАМИ ----- #
            for rot in [1, 2, 3]:
                rotated_img = img.rotate(rot * 90)
                rotated_path = os.path.join(temp_dir, f"{char_dir}_{img_file}_{rot}.png")
                rotated_img.save(rotated_path)
                datax.append(rotated_path)
                datay.append(char_dir + f'_{rot}')

    return np.array(datax), np.array(datay)


def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    
    results = [read_alphabets(base_directory + '/' + directory + '/') for directory in os.listdir(base_directory)]
    
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.concatenate([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay



def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support + n_querry, for n_way classes
    Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
    Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support + n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append([cv2.resize(cv2.imread(fname), (28, 28))
                                  for fname in sample_cls])
        
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.permute(0, 1, 4, 2, 3)
    return ({
        'images': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })

def display_sample(sample):
    """
    Displays sample in a grid
    Args:
      sample (torch.Tensor): sample of images to display
    """
    #need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    #make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))
    
    
if __name__ == "__main__":
    urllib.request.urlretrieve('https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip', 'images_background.zip')
    shutil.unpack_archive('images_background.zip', '.')
    urllib.request.urlretrieve('https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip', 'images_evaluation.zip')
    shutil.unpack_archive('images_evaluation.zip', '.')
