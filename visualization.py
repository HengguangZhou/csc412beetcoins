import os
from midi2audio import FluidSynth
import pretty_midi
from IPython.display import Audio, display
from dataset import MidiDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def visualize_hehexd(midi_data_hehexd):
    """
    Visualize each instrument's piano roll
    :param midi_data_hehexd: 3d tensor of size 4x128xP representing the encoding of the piano roll of the midi file
    :return: idk
    """
    soprano = midi_data_hehexd[0].transpose(0, 1)
    alto = midi_data_hehexd[1].transpose(0, 1)
    tenor = midi_data_hehexd[2].transpose(0, 1)
    bass = midi_data_hehexd[3].transpose(0, 1)

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(soprano, cmap='hot', interpolation='nearest')
    axs[0].set_title('soprano')
    axs[0].invert_yaxis()
    axs[1].imshow(alto, cmap='hot', interpolation='nearest')
    axs[1].set_title('alto')
    axs[1].invert_yaxis()
    axs[2].imshow(tenor, cmap='hot', interpolation='nearest')
    axs[2].set_title('tenor')
    axs[2].invert_yaxis()
    axs[3].imshow(bass, cmap='hot', interpolation='nearest')
    axs[3].set_title('bass')
    axs[3].invert_yaxis()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    return fig, axs


def pad_piano_roll(piano_roll, min_pitch, max_pitch):
    """
    Pad the piano roll so the pitches are at the right index
    :param piano_roll: piano roll hehexd
    :return: padded piano roll hehexd
    """
    out = F.pad(piano_roll, (min_pitch, 127-max_pitch))

    return out


if __name__ == '__main__':
    md = MidiDataset('../data/JSB-Chorales-dataset/jsb-chorales-16th.pkl')
    test_midi = md[0]
    print(test_midi[0].shape)

    padded_midi = pad_piano_roll(test_midi[0], md.get_min_midi_pitch(), md.get_max_midi_pitch())
    visualize_hehexd(padded_midi)
    plt.show()