import argparse
import os
import torch
import numpy as np
from torch import nn
from models.coconet import coco_decoder

def extract_latent(x):
    pass
    #TODO

def reconstruct(x, latent):
    pass
    #TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siamese')
    parser.add_argument('--target_midi', type=str, required=True)
    parser.add_argument('--style_midi', type=str,required=True)
    parser.add_argument("--weights", type=str, default="checkpoints/")
    parser.add_argument('--input_channels', type=int, default=3)

    opts = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'coconet':
        model = coco_decoder(opts.input_channels)
    else:
        model = coco_decoder(opts.input_channels)

    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model.eval()

    #TODO: take in midi file, use decoder to generate output and convert output back into midi file,
    #TODO: then generate midi visualization and playable audio
    #TODO: Check utils.py