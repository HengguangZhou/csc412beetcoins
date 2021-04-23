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

def get_concat_mask(T, P):
    temp = []
    for i in range(4):
        mask = get_mask(T, P)
        temp.append(mask)

    return np.concatenate(temp, axis=0)

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
    parser.add_argument('--target_midi', type=str, required=True)
    parser.add_argument('--style_midi', type=str, default=None)
    parser.add_argument("--weights", type=str, default="./weights/experiment1.pth")
    parser.add_argument('--input_channels', type=int, default=9)
    parser.add_argument('--time_steps', type=int, default=128)

    opts = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = coco_decoder(opts.input_channels, hidden_channels=opts.time_steps)

    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model.eval()

    midis = MidiDataset(opts.target_midi, timestep_len=opts.time_steps, fold='train')

    test_midi2d, test_midi3d, _, _ = midis[20]
    style_midi2d, style_midi3d, _, _ = midis[99]
    style_midi2d2, style_midi3d2, _, _ = midis[177]
    T = test_midi3d.shape[1]
    P = test_midi3d.shape[2]
    mask = torch.ones(test_midi3d.shape)
    mask[1:3, :, :] = 0
    mask = mask.unsqueeze(0)


    print(midis.get_min_midi_pitch())
    latent1 = extract_latent(style_midi2d, style_midi3d, model)
    # latent2 = extract_latent(style_midi2d2, style_midi3d2, model)
    pred = model(test_midi3d.unsqueeze(0), mask, latent1, testing=True)

    pred = torch.round(torch.nn.Softmax(dim=-1)(pred).squeeze(0).detach())
    mido_result = piano_roll2d_to_midi(convert_3d_to_2d(pred.numpy(), midis.get_min_midi_pitch(), timestep_size=128))
    mido_result.save('result.mid')
    piano_roll2d_to_midi(convert_3d_to_2d(style_midi3d.numpy(), midis.get_min_midi_pitch(), timestep_size=128)).save('style1.mid')
    piano_roll2d_to_midi(convert_3d_to_2d(style_midi3d2.numpy(), midis.get_min_midi_pitch(), timestep_size=128)).save('style2.mid')

    padded_midi2 = pad_piano_roll(pred, midis.get_min_midi_pitch(), midis.get_max_midi_pitch())
    visualize_hehexd(padded_midi2)
    plt.show()