# Importing the libraries
from fastai.vision import *
from efficientnet_pytorch import EfficientNet

# Define the path
path = Path('C:/Users/bzlis/Documents/CS 747/final project/data/birds/')
path.ls()
# Create the data using fastai's Datablock API
src = (ImageList.from_folder(path)
                .split_by_folder(train='train', valid='val')
                .label_from_folder()
                .add_test_folder('test')
                .transform(get_transforms(), size=224))

data = src.databunch(bs=32).normalize(imagenet_stats)

# Replace the fully connected layer at the end to fit our task
# Pre-trained model based on adversarial training
arch = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
arch._fc = nn.Linear(1280   , data.c)
# Define custom loss function
loss_func = LabelSmoothingCrossEntropy()

# Define the model
learn = Learner(data, arch, loss_func=loss_func, metrics=accuracy, model_dir='/kaggle/working')
# Train the model using 1 Cycle policy
learn.fit_one_cycle(3, slice(1e-3))


torch.save(learn.model.state_dict(), 'model.pth')
torch.save(learn.model.state_dict(), 'model.ph')
# Let's see the result
learn.show_results()
