import torch
import glob
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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