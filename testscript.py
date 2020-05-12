import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from models import *
from datasets import *
from datasets import BirdDataset
import numpy as np
import matplotlib.pyplot as plt
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
def viewimage(indx):
  transforms_ = [
      transforms.ToTensor()
  ]
  birb = BirdDataset("data/birds",transforms_=transforms_)
  item = birb.__getitem__(indx)
  view(item['A'])
  view(item['B'])


def view(img):
  img = img.numpy()
  img = img.transpose(1, 2, 0)
  plt.imshow(img)
  plt.show()
