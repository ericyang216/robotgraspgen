import torch
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from nn_models.pcae import PCAE
from train_nn import * #train_one_epoch, sample_from_model, make_dataloader
from data_util import plot_pc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--eps',       type=int, default=100,    help="Number of training epochs")
parser.add_argument('--eps_save',  type=int, default=100,    help="Save model every n epochs")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--eval',      type=int, default=0,     help="Flag for eval")
parser.add_argument('--sample',    type=int, default=0,     help="Number of samples to generate")
parser.add_argument('--batch',     type=int, default=512,   help="Batch size")
parser.add_argument('--lr',        type=float, default=1e-3,   help="Learning rate")
parser.add_argument('--checkpoint',type=str, default=None,  help="Load from checkpoint")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

synset_dict = {'airplane': 2691156}

model = PCAE().to(device)

print("Loading trainig data")
dataloader = make_dataloader(synset_dict['airplane'], device, batchsize=args.batch, n=512)

if args.checkpoint:
    print("Loading checkpoint: {}".format(args.checkpoint))
    load_checkpoint(model, args.checkpoint)

if args.train:
    print("Training")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_loss = []

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

        epoch_loss.append(loss)

        if ep % args.eps_save == 0 and ep != 0:
            save_path = 'checkpoints/pcae_airplane_512_cd_{}'.format(ep)
            print("Saving checkpoint: {}".format(save_path))
            save_checkpoint(model, save_path)


    plt.plot(range(0,args.eps), epoch_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
if args.eval:
    # for testing
    eval_data = make_dataloader(synset_dict['airplane'], device, batchsize=1, n=512)

    for [x] in eval_data:
        x_np = x.cpu().detach().numpy()[0]
        x_hat = model(x).cpu().detach().numpy()[0]
        print(x_np.shape)
        print(x_hat.shape)

        plot_pc(x_np, CXYZ=True)
        plot_pc(x_hat, CXYZ=True)

    