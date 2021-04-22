import argparse
import os
import torch
import numpy as np
from torch import nn
from models.coconet import coco_decoder
from convert import piano_roll2d_to_midi, convert_3d_to_2d, convert_2d_to_3d
from dataset import MidiDataset
from visualization import visualize_hehexd, pad_piano_roll
import matplotlib.pyplot as plt
from tsnecuda import TSNE
from sklearn.manifold import TSNE

def extract_latent(x, oh_x, model, iter=500):
    T = oh_x.shape[1]
    P = oh_x.shape[2]
    latent = torch.zeros(T, P)
    latent = latent.normal_(std=0.01)
    model.eval()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([{
                "params": latent,
                "lr": 1e-3,
                }])

    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(iter):
        optimizer.zero_grad()
        mask = torch.from_numpy(get_concat_mask(T, P))
        incomplete_x = oh_x * mask
        incomplete_x = incomplete_x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        pred = model(incomplete_x, mask, latent)
        loss = criterion(pred.reshape(-1, P), x.reshape(-1).long())
        loss += 1e-4 * torch.mean(latent.pow(2))
        print(f"loss: {loss.item()}")
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
    parser.add_argument("--weights", type=str, default="./weights/experiment1.pth")
    parser.add_argument('--input_channels', type=int, default=9)

    # opts = parser.parse_args()
    #
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")
    #
    # model = coco_decoder(9)
    # model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    # model.eval()
    #
    # mds = MidiDataset('./data/jsb/jsb-chorales-16th.pkl')
    #
    # latents = []
    # for i, v in enumerate(mds):
    #     test_midi2d, test_midi3d, _, _ = v
    #     latent = extract_latent(test_midi2d, test_midi3d, model)
    #     latents.append(latent)

    # latents = torch.cat(latents)
    # torch.save(latents, "latents.pt")
    # X = torch.zeros(16, 784)
    latents = torch.load("latents.pt").detach()

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    x = tsne.fit_transform(latents)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=0.3
    # )
