import torch
import argparse
import time
import numpy as np

from nn_models.pcae import PCAE
from train_nn import train_one_epoch, sample_from_model, make_dataloader
from data_util import plot_pc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--eps',       type=int, default=10,    help="Number of training epochs")
parser.add_argument('--eps_save',  type=int, default=10,    help="Save model every n epochs")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--sample',    type=int, default=0,     help="Number of samples to generate")
parser.add_argument('--batch',     type=int, default=64,    help="Batch size")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

synset_dict = {'airplane': 2691156}

model = PCAE().to(device)

print("Loading trainig data")
dataloader = make_dataloader(synset_dict['airplane'], device, batchsize=1)

if args.train:
    print("Training")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for ep in range(args.eps):
        start = time.time()
        loss, size = train_one_epoch(model, dataloader, optimizer)
        end = time.time()

        epoch_time = end - start
        samples_sec = args.batch * size / epoch_time
        print("Epoch {}: {} \t [{:.2f} samp/s]".format(ep, loss, samples_sec))

        if args.sample > 0:
            samples = sample_from_model(model, args.sample)
            for sample in samples:
                plot_pc(sample)
            