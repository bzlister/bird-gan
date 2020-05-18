# Importing the libraries
from fastai.vision import *
from efficientnet_pytorch import EfficientNet
import torch


# Define the path
path = Path('C:/Users/bzlis/Documents/CS 747/bird-gan/data/birds/')
path.ls()
# Create the data using fastai's Datablock API
src = (ImageList.from_folder(path)
                .split_by_folder(train='train', valid='val')
                .label_from_folder()
                .add_test_folder('test')
                .transform(get_transforms(do_flip=True, max_rotate=10.0, max_zoom=1, max_lighting=0.2, max_warp=None, p_affine=1, p_lighting=1), size=256))

data = src.databunch(bs=32).normalize()

# Replace the fully connected layer at the end to fit our task
# Pre-trained model based on adversarial training
arch = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
arch._fc = nn.Linear(1280   , 190)
arch = arch.to(torch.device('cuda'))
# Define custom loss function
loss_func = LabelSmoothingCrossEntropy()

# Define the model
learn = Learner(data, arch, loss_func=loss_func, metrics=accuracy, model_dir='/kaggle/working')
# Train the model using 1 Cycle policy
learn.fit_one_cycle(1, slice(1e-3))
learn.freeze()
learn.export()
