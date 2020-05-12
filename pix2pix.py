import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from fastai.vision import ImageList, LabelSmoothingCrossEntropy, Path, get_transforms, imagenet_stats, Learner, accuracy
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="birds", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda')
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_crossentropy = torch.nn.CrossEntropyLoss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = -0.5
    lambda_clf = 300

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()
    path = Path('C:/Users/bzlis/Documents/CS 747/bird-gan/data/birds/')
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
    clf = learn.model

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_crossentropy = criterion_crossentropy.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()

        clf.load_state_dict(torch.load('model.pth'))
        clf.to(torch.device("cuda"))
        print("Models on GPU")
    else:
        clf.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        clf.cuda()
        print("Models on CPU")

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        BirdDataset("data/%s" % opt.dataset_name, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        BirdDataset("data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        input = Variable(imgs["img"].type(Tensor))
        output = generator(input)
        img_sample = torch.cat((input.data, output.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            input = Variable(batch["img"].type(Tensor))
            label = batch["label"].to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((input.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()


            gan_output = generator(input)
            '''
            gan_output_cuda = gan_output.cuda()
            pred_class = clf(gan_output_cuda).to('cpu')
            '''
            pred_class = clf(gan_output.to(device))

            pred_fake = discriminator(gan_output, input)
            # GAN loss
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(gan_output, input)
            # Classification loss
            loss_clf = criterion_crossentropy(pred_class, label)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_clf * loss_clf

            loss_G.backward(retain_graph=True)

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            #pred_real = discriminator(, real_A)
            #loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = loss_fake
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
