import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt
import pytorch_ssim

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

import pyramid
from loss_pyramid import EdgeSaliencyLoss
from fastai.vision import load_learner
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="birds", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
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

    lambda_clf = 10
    lambda_pixel = 5
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    #criterion_pixelwise = torch.nn.L1Loss()
    criterion_pixelwise = pytorch_ssim.SSIM()
    criterion_crossentropy = torch.nn.CrossEntropyLoss()
    criterion_hist = torch.nn.MSELoss()
    softmax = torch.nn.Softmax()

    learn = load_learner('data/birds/')
    learn.train_bn = False
    clf = learn.model
    for param in clf.parameters():
        param.requires_grad = False
    classes = ['ALBATROSS', 'ALEXANDRINE PARAKEET', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART', 'ANHINGA', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ARARIPE MANAKIN', 'BALD EAGLE', 'BALTIMORE ORIOLE', 'BANANAQUIT', 'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW', 'BAY-BREASTED WARBLER', 'BELTED KINGFISHER', 'BIRD OF PARADISE', 'BLACK FRANCOLIN', 'BLACK SKIMMER', 'BLACK SWAN', 'BLACK THROATED WARBLER', 'BLACK VULTURE', 'BLACK-CAPPED CHICKADEE', 'BLACK-NECKED GREBE', 'BLACK-THROATED SPARROW', 'BLACKBURNIAM WARBLER', 'BLUE GROUSE', 'BLUE HERON', 'BOBOLINK', 'BROWN NOODY', 'BROWN THRASHER', 'CACTUS WREN', 'CALIFORNIA CONDOR', 'CALIFORNIA GULL', 'CALIFORNIA QUAIL', 'CANARY', 'CAPE MAY WARBLER', 'CARMINE BEE-EATER', 'CASPIAN TERN', 'CASSOWARY', 'CHARA DE COLLAR', 'CHIPPING SPARROW', 'CINNAMON TEAL', 'COCK OF THE  ROCK', 'COCKATOO', 'COMMON GRACKLE', 'COMMON HOUSE MARTIN', 'COMMON LOON', 'COMMON POORWILL', 'COMMON STARLING', 'COUCHS KINGBIRD', 'CRESTED AUKLET', 'CRESTED CARACARA', 'CROW', 'CROWNED PIGEON', 'CUBAN TODY', 'CURL CRESTED ARACURI', 'D-ARNAUDS BARBET', 'DARK EYED JUNCO', 'DOWNY WOODPECKER', 'EASTERN BLUEBIRD', 'EASTERN MEADOWLARK', 'EASTERN ROSELLA', 'EASTERN TOWEE', 'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMPEROR PENGUIN', 'EMU', 'EURASIAN MAGPIE', 'EVENING GROSBEAK', 'FLAME TANAGER', 'FLAMINGO', 'FRIGATE', 'GILA WOODPECKER', 'GLOSSY IBIS', 'GOLD WING WARBLER', 'GOLDEN CHLOROPHONIA', 'GOLDEN EAGLE', 'GOLDEN PHEASANT', 'GOULDIAN FINCH', 'GRAY CATBIRD', 'GRAY PARTRIDGE', 'GREEN JAY', 'GREY PLOVER', 'GUINEAFOWL', 'HAWAIIAN GOOSE', 'HOODED MERGANSER', 'HOOPOES', 'HORNBILL', 'HOUSE FINCH', 'HOUSE SPARROW', 'HYACINTH MACAW', 'IMPERIAL SHAQ', 'INCA TERN', 'INDIGO BUNTING', 'JABIRU', 'JAVAN MAGPIE', 'KILLDEAR', 'KING VULTURE', 'LARK BUNTING', 'LILAC ROLLER', 'LONG-EARED OWL', 'MALEO', 'MALLARD DUCK', 'MANDRIN DUCK', 'MARABOU STORK', 'MASKED BOOBY', 'MIKADO  PHEASANT', 'MOURNING DOVE', 'MYNA', 'NICOBAR PIGEON', 'NORTHERN CARDINAL', 'NORTHERN FLICKER', 'NORTHERN GANNET', 'NORTHERN GOSHAWK', 'NORTHERN JACANA', 'NORTHERN MOCKINGBIRD', 'NORTHERN RED BISHOP', 'OCELLATED TURKEY', 'OSPREY', 'OSTRICH', 'PAINTED BUNTIG', 'PARADISE TANAGER', 'PARUS MAJOR', 'PEACOCK', 'PELICAN', 'PEREGRINE FALCON', 'PINK ROBIN', 'PUFFIN', 'PURPLE FINCH', 'PURPLE GALLINULE', 'PURPLE MARTIN', 'PURPLE SWAMPHEN', 'QUETZAL', 'RAINBOW LORIKEET', 'RAZORBILL', 'RED BISHOP WEAVER', 'RED FACED CORMORANT', 'RED HEADED DUCK', 'RED HEADED WOODPECKER', 'RED HONEY CREEPER', 'RED THROATED BEE EATER', 'RED WINGED BLACKBIRD', 'RED WISKERED BULBUL', 'RING-BILLED GULL', 'RING-NECKED PHEASANT', 'ROADRUNNER', 'ROBIN', 'ROCK DOVE', 'ROSY FACED LOVEBIRD', 'ROUGH LEG BUZZARD', 'RUBY THROATED HUMMINGBIRD', 'RUFOUS KINGFISHER', 'RUFUOS MOTMOT', 'SAND MARTIN', 'SCARLET IBIS', 'SCARLET MACAW', 'SHOEBILL', 'SNOWY EGRET', 'SNOWY OWL', 'SORA', 'SPANGLED COTINGA', 'SPLENDID WREN', 'SPOONBILL', 'STEAMER DUCK', 'STORK BILLED KINGFISHER', 'STRAWBERRY FINCH', 'TAIWAN MAGPIE', 'TEAL DUCK', 'TIT MOUSE', 'TOUCHAN', 'TRUMPTER SWAN', 'TURKEY VULTURE', 'TURQUOISE MOTMOT', 'VARIED THRUSH', 'VENEZUELIAN TROUPIAL', 'VERMILION FLYCATHER', 'VIOLET GREEN SWALLOW', 'WHITE CHEEKED TURACO', 'WHITE NECKED RAVEN', 'WHITE TAILED TROPIC', 'WILD TURKEY', 'WILSONS BIRD OF PARADISE', 'WOOD DUCK', 'YELLOW HEADED BLACKBIRD']

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_crossentropy = criterion_crossentropy.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()#criterion_pixelwise.cuda()
        criterion_hist = criterion_hist.cuda()
        softmax = softmax.cuda()
        device = torch.device('cuda')

        #pyra = pyra.to(device)
        print("Models on GPU")
    else:
        device = torch.device('cpu')
        print("Models on CPU")

    clf.to(device)
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        correct_label = imgs["label"]
        output = generator(input)
        pred = clf(output.to(device))
        incorrect_label = (pred == pred.max()).nonzero()[0][1].item()
        img_sample = torch.cat((input.data, output.data), -2)
        save_image(img_sample, "images/%s/%s ACTUAL=%s PREDICTED=%s.png" % (opt.dataset_name, batches_done, classes[correct_label], classes[incorrect_label]), nrow=5, normalize=True)


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

            pred_class = clf(gan_output.to(device))

            pred_fake = discriminator(gan_output)
            # GAN loss
            loss_GAN = criterion_GAN(pred_fake, valid)


            loss_clf = 1/criterion_crossentropy(pred_class,label).detach()

            '''
            r_out, g_out, b_out = gan_output[:,0,:,:], gan_output[:,1,:,:], gan_output[:,2,:,:]
            r_in, g_in, b_in = input[:,0,:,:], input[:,1,:,:], input[:,2,:,:]
            loss_pixelwise = 1-criterion_pixelwise((0.299*r_out + 0.587*g_out + 0.114*b_out).reshape(-1,1,256,256), (0.299*r_in + 0.587*g_in + 0.114*b_in).reshape(-1,1,256,256))
            loss_hist = criterion_hist(r_out.histc(bins=256), r_in.histc(bins=256)).detach() + criterion_hist(g_out.histc(bins=256), g_in.histc(bins=256)).detach() + criterion_hist(b_out.histc(bins=256), b_in.histc(bins=256)).detach()
            '''
            loss_pixelwise = 1 - criterion_pixelwise(gan_output, input)

            # Total loss
            loss_G = loss_GAN + lambda_pixel*loss_pixelwise + lambda_clf*loss_clf
            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(input)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            loss_fake = criterion_GAN(discriminator(gan_output.detach()), fake)
            loss_D = 0.5*(loss_fake + loss_real)
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
                "\r[Batch %d/%d] [D loss: %f] [G loss: %f, clf: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_clf.item()*lambda_clf,
                    loss_pixelwise.item()*lambda_pixel,
                    loss_GAN.item(),
                    time_left,
                )
            )
            '''
            sys.stdout.write(
                "\r[Batch %d/%d] [D loss: %f] [G loss: %f, clf_right: %f, clf_wrong: %f, adv: %f] ETA: %s"
                % (
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_clf_correct.item()*lambda_correct,
                    loss_clf_incorrect.item()*lambda_incorrect,
                    loss_GAN.item(),
                    time_left,
                )
            )
            '''

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
