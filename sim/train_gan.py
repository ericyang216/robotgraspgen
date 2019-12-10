import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn.gan import GANModel
from nn.model_utils import AngleLoss, DistanceLoss

Z_DIM = 4 # 32, 256, 512
H_DIM = 32 # 128, 256, 512

NUM_SAMPLES = 7183
TRAIN_SAMPLES = 7000
VAL_SAMPLES = NUM_SAMPLES - TRAIN_SAMPLES

LOAD_MODEL = False
SAVE_MODEL = True
roi_H = 128
roi_W = 128

BATCH = 128
LR = 1e-3
ANGLE_LOSS_K = 1 #100.
EPOCHS = 500
SAVE_EVERY = 100

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

def load_dataset(start, end):
    DATA_DIR = "./data/cube2/depth"
    LABEL_DIR = "./data/cube2/label"

    num_images = end - start

    images = np.empty(shape=(num_images,1,128,128)).astype('float32')
    labels = np.empty(shape=(num_images,5)).astype('float32')
    
    for i in range(num_images):
        images[i] = np.load(os.path.join(DATA_DIR, "%s.npy" % i))
        label = np.load(os.path.join(LABEL_DIR, "%s.npy" % i))[:5]
        labels[i] = label
    return images, labels

def train_one_epoch(model, dataloader, d_optimizer, g_optimizer):
    total_loss = 0 #torch.tensor(0.0, device=device, requires_grad=False)

    for img, label in dataloader:
        d_loss, g_loss = model.loss(img, label)
        
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        total_loss += g_loss.cpu().item()

    return total_loss / len(dataloader)

def train(model, dataset, evalset):
    train_loss = []
    eval_loss = []

    d_optimizer = torch.optim.Adam(model.D.parameters(), lr=LR, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(model.G.parameters(), lr=LR, betas=(0.5, 0.999))
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    print("Total Training Batches: {}".format(len(dataloader)))

    for i in range(1, EPOCHS+1):
        model.train()
        train_loss_epoch = train_one_epoch(model, dataloader, d_optimizer, g_optimizer)
        train_loss.append(train_loss_epoch)

        eval_loss_epoch = evaluate(model, evalset)
        eval_loss.append(eval_loss_epoch)

        print("Epoch {}: Train {}, Eval {}".format(i, train_loss_epoch, eval_loss_epoch))

        # Save model checkpoint and training losses
        if SAVE_MODEL and (i % SAVE_EVERY == 0):
            # timestamp = int(time.time())
            save(model, './checkpoints2/{}_{}.pt'.format(model.name,i))
            np.save('./checkpoints2/{}_train_loss.npy'.format(model.name), np.array(train_loss))
            np.save('./checkpoints2/{}_eval_loss.npy'.format(model.name), np.array(eval_loss))

    print("Train Losses:")
    for i, loss in enumerate(train_loss):
        print("{}: {}".format(i, loss))

def save(model, model_path):
    torch.save(model.state_dict(), model_path)
    print("Model saved: {}".format(model_path))

def load(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

def evaluate(model, dataset):
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=BATCH)
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)

        model.eval()
        for img, label in dataloader:

            output = model(img)

            loss = DistanceLoss(output, label) + ANGLE_LOSS_K * AngleLoss(output, label, norm=True)

            total_loss += loss

    return total_loss.cpu().item() / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_dim', type=int, default=H_DIM, help='size of hidden layers in FC')
    parser.add_argument('--z_dim', type=int, default=Z_DIM, help='size of latent variable in VAE')
    args = parser.parse_args()
    
    print("=== Creating Model ===")
    model = GANModel(args.z_dim, args.h_dim)
    model.to(device)

    print("=== Loading Testing Data ===")
    # Load test data
    evalset = None
    images, labels = load_dataset(TRAIN_SAMPLES, NUM_SAMPLES)
    eval_images = torch.from_numpy(images).to(device=device, dtype=dtype)
    eval_labels = torch.from_numpy(labels).to(device=device, dtype=dtype)
    evalset = TensorDataset(eval_images, eval_labels)

    if LOAD_MODEL:
        print("=== Loading Model ===")
        load(model, MODEL_PATH)

    else:
        print("=== Loading Training Data ===")
        # Load Files
        images, labels = load_dataset(0, TRAIN_SAMPLES)
        
        # Convert to NCHW dimensions
        # images = np.transpose(images, (0,3,1,2))

        # Transfer to GPU, float32
        images = torch.from_numpy(images).to(device=device, dtype=dtype)
        labels = torch.from_numpy(labels).to(device=device, dtype=dtype)
        
        # Create dataset from tensors
        dataset = TensorDataset(images, labels)

        print("=== Training Model ===")
        train(model, dataset, evalset)

    print("=== Evaluate Model ===")
    eval_loss = evaluate(model, evalset)
    print("Evaluation Loss: {}".format(eval_loss))
