import argparse
import os
import torch
import numpy as np
from torch import nn
from models.coconet import coco_decoder
from convert import piano_roll2d_to_midi, convert_3d_to_2d, convert_2d_to_3d
from dataset import MidiDataset
from visualization import *
from PIL import Image
import matplotlib.pyplot as plt


def extract_latent(x, oh_x, model, iter=1):
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

    opts = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = coco_decoder(9)
    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model.eval()

    md = MidiDataset('../jsb/jsb-chorales-16th.pkl')

    values = [0, 0.2,0.4, 0.6, 0.8, 1]
    test_midi2d, test_midi3d, mask, _ = md[0]
    mask = mask.unsqueeze(0)
    latent_a = extract_latent(test_midi2d, test_midi3d, model)
    latent_b = extract_latent(test_midi2d, test_midi3d, model)
    latent_c = extract_latent(test_midi2d, test_midi3d, model)

    # first row

    for r in range(1):
        for c in range(len(values)):
            beta = values[r]
            alpha = values[c]
            inter_latent = beta * (alpha * latent_a + (1 - alpha) * latent_b) + (1 - beta) * latent_c
            test_midi = model(test_midi3d.unsqueeze(0), mask, inter_latent)
            test_midi = torch.round(test_midi.squeeze(0).detach())
            padded_midi2 = pad_piano_roll(test_midi, md.get_min_midi_pitch(), md.get_max_midi_pitch())
            img = visualize_together(padded_midi2, md.get_min_midi_pitch(), md.get_max_midi_pitch())
            img_arr = get_img_from_fig(img)
            if c == 0:
                col_plot = img_arr
            else:
                col_plot = np.hstack((col_plot, img_arr))
        first_row = col_plot

    # rest
    for r in range(1,len(values)):
        for c in range(len(values)):
            beta = values[r]
            alpha = values[c]
            inter_latent = beta * (alpha * latent_a + (1 - alpha) * latent_b) + (1 - beta) * latent_c
            test_midi = model(test_midi3d.unsqueeze(0), mask, inter_latent)
            test_midi = torch.round(test_midi.squeeze(0).detach())
            padded_midi2 = pad_piano_roll(test_midi, md.get_min_midi_pitch(), md.get_max_midi_pitch())
            img = visualize_together(padded_midi2, md.get_min_midi_pitch(), md.get_max_midi_pitch())
            img_arr = get_img_from_fig(img)
            if c == 0:
                col_plot = img_arr
            else:
                col_plot = np.hstack((col_plot, img_arr))
        first_row = np.vstack((first_row, col_plot))

    test = Image.fromarray(first_row, 'RGB')
    test.show()


