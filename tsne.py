import argparse
import os
import torch
import numpy as np
from torch import nn
from models.coconet import coco_decoder
from convert import piano_roll2d_to_midi, convert_3d_to_2d, convert_2d_to_3d
from dataset import MidiDataset
from visualization import visualize, pad_piano_roll
import matplotlib.pyplot as plt
from tsnecuda import TSNE
from sklearn.manifold import TSNE
from maestro_dataset import MaestroDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

def extract_latent(x, oh_x, model, device, iter=500):
    B = oh_x.shape[0]
    T = oh_x.shape[2]
    P = oh_x.shape[3]
    latent = torch.zeros(B, 1, T, P)
    latent = latent.normal_(std=0.01).to(device)
    model.eval()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([{
                "params": latent,
                "lr": 1e-3,
                }])

    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    oh_x = oh_x.to(device)
    x = x.to(device)
    for i in range(iter):
        optimizer.zero_grad()
        masks = []
        for j in range(B):
            masks.append(torch.from_numpy(get_concat_mask(T, P)).unsqueeze(0))
        masks = torch.cat(masks).to(device)
        # incomplete_x = oh_x * mask
        # incomplete_x = incomplete_x.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        pred = model(oh_x, masks, latent)
        loss = criterion(pred.reshape(-1, P), x.reshape(-1).long())
        loss += 1e-4 * torch.mean(latent.pow(2))
        print(f"iter: {i}/{iter} loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    return latent

def get_concat_mask(T, P):
    temp = []
    for i in range(4):
        mask = get_mask(T, P)
        temp.append(mask)

    return np.concatenate(temp, axis=0)

def get_mask(T, P):
    # mask size is 1 x data_len x P
    mask_idx = np.random.choice(T * P, size=np.random.choice(T * P) + 1, replace=False)
    # print(len(mask_idx))
    mask = np.zeros(T * P, dtype=np.float32)
    mask[mask_idx] = 1.
    mask = mask.reshape((1, T, P))

    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='coconet')
    # parser.add_argument('--target_midi', type=str, required=True)
    # parser.add_argument('--style_midi', type=str, default=None)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--weights", type=str, default="./weights/on_new_dataset2.pth")
    parser.add_argument('--input_channels', type=int, default=9)
    parser.add_argument('--time_steps', type=int, default=128)

    opts = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = coco_decoder(9, hidden_channels=opts.time_steps).to(device)
    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model.eval()

    # mds = MidiDataset('./data/jsb/jsb-chorales-16th.pkl', timestep_len=opts.time_steps)
    mds = MaestroDataset('./data/maestro-v3.0.0', timestep_len=opts.time_steps)
    mds_loader = DataLoader(dataset=mds,
               batch_size=opts.bs,
               shuffle=True)

    latents = []
    labels = []
    print(len(mds_loader))

    j = 0
    with tqdm(total=(len(mds) - len(mds) % opts.bs)) as t:
        for i, v in enumerate(mds_loader):
            test_midi2d, test_midi3d, _, _, label = v
            latent = extract_latent(test_midi2d, test_midi3d, model, device=device, iter=250)
            latents.append(latent.reshape(test_midi2d.shape[0], -1))
            labels.append(label)
            t.update(test_midi2d.shape[0])

            if i != 0 and i % 25 == 0:
                torch.save(torch.cat(latents), f"latents{j}.pt")
                torch.save(torch.cat(labels), f"labels{j}.pt")
                j += 1


    latents = torch.cat(latents)
    labels = torch.cat(labels)
    torch.save(latents, "final_latents.pt")
    torch.save(labels, "final_labels.pt")

    # latents = torch.load("latents0.pt").detach()
    # labels = torch.load("labels0.pt").detach()


    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
    x = tsne.fit_transform(latents.detach().cpu()[:])
    plt.scatter(x[:, 0], x[:, 1], c=labels[:])
    plt.show()
