import torch
#Pytorch framework
from torch import nn
#Neural network
from tqdm.auto import tqdm
#Progress bars
from torchvision import transforms
#Computer vision, cropping/changing images
from torchvision.datasets import MNIST 
#Training dataset
from torchvision.utils import make_grid
#Creating a grid from a batch of data
from torch.utils.data import DataLoader
#Loading data
import matplotlib.pyplot as plt
#Creating graphs
torch.manual_seed(0) # Set for testing purposes, please do not change!
