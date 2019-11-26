import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Normalize(nn.Module):
    def forward(self, input):
        sq = input * input
        norm = torch.sqrt(torch.sum(sq, dim=0))
        return input / norm

class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(4096, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU())
                    # nn.Linear(256, 5))

        self.pos_net = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 3))

        self.ang_net = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2),
                                     Normalize())
    def forward(self, x):
        features = self.net(x)
        position = self.pos_net(features)
        angle = self.ang_net(features)
        return torch.cat([position, angle], dim=1)

    def loss(self, a, b):
        mse_loss = torch.mean(torch.sum(torch.abs(a[:, 0:3] - b[:, 0:3]), dim=1), dim=0)
        cos_loss = 1. - ((a[:, 4] * b[:, 4]) + (a[:, 3] * b[:, 3]))
        cos_loss = torch.mean(cos_loss, dim=0)

        return mse_loss, cos_loss

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

        output = model(img)
        mse_loss, cos_loss = model.loss(output, label)
        loss = mse_loss + ANGLE_LOSS_K * cos_loss
        
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
            save(model, './checkpoints/cnn_{}_{}'.format(timestamp, i))
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
            mse_loss, cos_loss = model.loss(output, label)
            loss = mse_loss + ANGLE_LOSS_K * cos_loss

            total_loss += loss

    return total_loss.cpu().item() / len(dataloader)

if __name__ == "__main__":
   
    print("=== Creating Model ===")
    model = model()
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
