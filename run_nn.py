import torch
import numpy as np
from nn_models.pcae import PCAE
from train_nn import train
from nn_util import make_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

synset_dict = {'airplane': 2691156}

model = PCAE().to(device)

dataloader = make_dataloader(synset_dict['airplane'], device, batchsize=1)

train(model, dataloader, 10)

