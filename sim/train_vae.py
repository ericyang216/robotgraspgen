import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn.vae import VAEModel
from nn.model_utils import AngleLoss, DistanceLoss

# from Q4_helper import load_dataset, load_training_dataset, load_testing_dataset

LOAD_MODEL = False
SAVE_MODEL = True
roi_H = 128
roi_W = 128

BATCH = 64
LR = 1e-4
ANGLE_LOSS_K = 100.
EPOCHS = 1000
SAVE_EVERY = 100

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dtype = torch.float32

def load_dataset(start, end):
    DATA_DIR = "./data/cube/depth"
    LABEL_DIR = "./data/cube/label"

    num_images = end - start

    images = np.empty(shape=(num_images,1,128,128)).astype('float32')
    labels = np.empty(shape=(num_images,5)).astype('float32')
    
    for i in range(num_images):
        images[i] = np.load(os.path.join(DATA_DIR, "%s.npy" % i))
        label = np.load(os.path.join(LABEL_DIR, "%s.npy" % i))[:5]
        labels[i] = label
    return images, labels

def train_one_epoch(model, dataloader, optimizer):
    total_loss = torch.tensor(0.0, device=device, requires_grad=False)

    for img, label in dataloader:
        optimizer.zero_grad()

        nelbo, kl, rec = model.loss(img, label)
        
        loss = nelbo
        
        loss.backward()

        optimizer.step()

        total_loss += loss

    return total_loss.cpu().item() / len(dataloader)

def train(model, dataset, evalset):
    train_loss = []
    eval_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    print("Total Training Batches: {}".format(len(dataloader)))

    for i in range(EPOCHS):
        model.train()
        train_loss_epoch = train_one_epoch(model, dataloader, optimizer)
        train_loss.append(train_loss_epoch)

        eval_loss_epoch = evaluate(model, evalset)
        eval_loss.append(eval_loss_epoch)

        print("Epoch {}: Train {}, Eval {}".format(i, train_loss[i], eval_loss[i]))

        # Save model checkpoint and training losses
        if SAVE_MODEL and (i % SAVE_EVERY == 0):
            timestamp = int(time.time())
            save(model, './checkpoints/vae_{}_{}'.format(timestamp, i))
            np.save('./checkpoints/train_loss_{}_{}'.format(timestamp, i), np.array(train_loss))
            np.save('./checkpoints/eval_loss_{}_{}'.format(timestamp, i), np.array(eval_loss))

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
   
    print("=== Creating Model ===")
    model = VAEModel()
    model.to(device)

    print("=== Loading Testing Data ===")
    # Load test data
    evalset = None
    images, labels = load_dataset(3000, 4000)
    eval_images = torch.from_numpy(images).to(device=device, dtype=dtype)
    eval_labels = torch.from_numpy(labels).to(device=device, dtype=dtype)
    evalset = TensorDataset(eval_images, eval_labels)

    if LOAD_MODEL:
        print("=== Loading Model ===")
        load(model, MODEL_PATH)

    else:
        print("=== Loading Training Data ===")
        # Load Files
        images, labels = load_dataset(0, 3000)
        
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
