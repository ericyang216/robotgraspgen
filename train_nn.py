import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset


def train(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        for [x] in dataloader:
            optimizer.zero_grad()

            loss = model.loss_chamfer(x)
            loss.backward()
            optimizer.step()
            # pbar.update(1)

            # pbar.set_postfix(loss='{:.2e}'.format(loss))
        print("Epoch {} loss: {}".format(ep, loss))