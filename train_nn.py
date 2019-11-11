import torch
import glob
import datetime
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def train_one_epoch(model, dataloader, optimizer):
    total_loss = 0
    n = 0
    for [x] in dataloader:
        optimizer.zero_grad()

        loss = model.loss_chamfer(x)

        loss.backward()
        optimizer.step()

        total_loss += loss
        n += 1

    return total_loss / n, n

def sample_from_model(model, n):
    x_samples = model.sample_x(n)
    return x_samples.cpu().detach().numpy()

def save_checkpoint(model, path=None):
    if path is None:
        path = './checkpoints/{}-{}'.format(model.name, datatime.timestamp().total_time())
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    return model

def gather_pc_data(synset_id, n=256):
    # Get all point clouds from sysnets
    MODEL_PATH = './shapenet/{:08}/*/models/model_normalized_{}.npy'.format(
                    synset_id, n)

    data = None
    model_files = glob.glob(MODEL_PATH)
    for i, pc_file in enumerate(model_files):
        if data is None:
            data = np.zeros((len(model_files), n, 3)).astype(np.float32)

        data[i] = np.load(pc_file).astype(np.float32)

    return data

def make_dataloader(synset_id, device, batchsize=1, n=256):
    data_np = gather_pc_data(synset_id, n)
    data_tensor = torch.from_numpy(data_np).to(device)
    data_tensor.transpose_(2, 1)
    return DataLoader(TensorDataset(data_tensor), batch_size=batchsize)