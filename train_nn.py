import torch
import glob
import datetime
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def train_one_epoch(model, dataloader, optimizer):
    total_loss = 0
    n = 0
    for (x, y) in dataloader:
        optimizer.zero_grad()

        x_hat = model.forward(x)
        loss = model.loss_chamfer(x_hat, y)

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

# def gather_pc_data(target_object, partial_n=128, complete_n=1024):
#     # Get all point clouds from sysnets

#     PC_PATH = './ycb/ycb/{}/clouds/partial_*_{}.npy'.format(target_object, partial_n)
#     GT_PATH = './ycb/ycb/{}/clouds/cloud_{}_*.npy'.format(target_object, complete_n)

#     pc_files = glob.glob(PC_PATH)
#     gt_files = glob.glob(GT_PATH)
    
#     data = np.zeros((len(pc_files), complete_n, 3)).astype(np.float32)
#     gt   = np.zeros((len(pc_files), complete_n, 3)).astype(np.float32)

#     for i, pc_file in enumerate(pc_files):
#         # Get a ground truth cloud
#         idx = np.random.randint(0, len(gt_files))
#         gt[i] = np.load(gt_files[idx]).astype(np.float32)

#         # Subsample partial cloud
#         partial = np.load(pc_file).astype(np.float32)
#         idx = np.random.randint(0, partial.shape[0], size=complete_n)
#         data[i] = partial[idx, :]

#     return data, gt

# def make_dataloader(target_object, device, batchsize=1, partial_n=128, complete_n=1024):
#     data_np, gt_np = gather_pc_data(target_object, partial_n, complete_n)
   
#     data_tensor = torch.from_numpy(data_np).to(device)
#     gt_tensor = torch.from_numpy(gt_np).to(device)
    
#     data_tensor.transpose_(2, 1)
#     gt_tensor.transpose_(2, 1)

#     return DataLoader(TensorDataset(data_tensor, gt_tensor), batch_size=batchsize)

# def gather_pc_data(synset_id, n=256):
#     # Get all point clouds from sysnets
#     MODEL_PATH = './shapenet/{:08}/*/models/model_normalized_{}.npy'.format(
#                     synset_id, n)

#     data = None
#     model_files = glob.glob(MODEL_PATH)
#     for i, pc_file in enumerate(model_files):
#         if data is None:
#             data = np.zeros((len(model_files), n, 3)).astype(np.float32)

#         data[i] = np.load(pc_file).astype(np.float32)

#     return data

# def make_dataloader(synset_id, device, batchsize=1, n=256):
#     data_np = gather_pc_data(synset_id, n)
#     data_tensor = torch.from_numpy(data_np).to(device)
#     data_tensor.transpose_(2, 1)
#     return DataLoader(TensorDataset(data_tensor), batch_size=batchsize)