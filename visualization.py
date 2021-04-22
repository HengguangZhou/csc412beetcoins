import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from midi2audio import FluidSynth
import pretty_midi
from IPython.display import Audio, display
from dataset import MidiDataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from convert import *

# This function is adapted from
# https://github.com/kevindonoghue/coconet-pytorch/blob/master/coconet.ipynb?fbclid=IwAR3XEObsWdDMqocQX5L_lAHACqQG8wc1WURNY1XeUAAhQZpgV42qc0l_7fM
def visualize_hehexd(midi_data):
    """
    Visualize each instrument's piano roll
    :param midi_data_hehexd: 3d tensor of size 4x128xP representing the encoding of the piano roll of the midi file
    :return: idk
    """
    soprano = midi_data[0].transpose(0, 1)
    alto = midi_data[1].transpose(0, 1)
    tenor = midi_data[2].transpose(0, 1)
    bass = midi_data[3].transpose(0, 1)

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

def visualize_together(midi_data, min_pitch, max_pitch):
    soprano = midi_data[0].transpose(0, 1)
    alto = midi_data[1].transpose(0, 1)
    tenor = midi_data[2].transpose(0, 1)
    bass = midi_data[3].transpose(0, 1)
    plt.imshow(soprano, cmap='plasma', interpolation='nearest', origin="lower")
    plt.imshow(alto, cmap='Reds', interpolation='nearest', alpha=0.5, origin="lower")
    plt.imshow(tenor, cmap='Greens', interpolation='nearest', alpha=0.5, origin="lower")
    plt.imshow(bass, cmap='Blues', interpolation='nearest', alpha=0.5, origin="lower")
    plt.ylim(min_pitch - 1, max_pitch + 1)


def pad_piano_roll(piano_roll, min_pitch, max_pitch):
    """
    Pad the piano roll so the pitches are at the right index
    :param piano_roll: piano roll hehexd
    :return: padded piano roll hehexd
    """
    out = F.pad(piano_roll, (min_pitch, 127-max_pitch))

    return out


if __name__ == '__main__':
    md = MidiDataset('../jsb/jsb-chorales-16th.pkl')
    test_midi = md[10]
    print(test_midi[0].shape)


    # padded_midi = pad_piano_roll(test_midi[0], md.get_min_midi_pitch(), md.get_max_midi_pitch())
    # visualize_hehexd(padded_midi)
    # plt.show()

    test_midi2d = convert_3d_to_2d(test_midi[1], md.get_min_midi_pitch())
    test_midi3d = convert_2d_to_3d(torch.transpose(test_midi2d, 0, 1), md.get_min_midi_pitch(), md.get_pitch_range())
    padded_midi2 = pad_piano_roll(test_midi3d, md.get_min_midi_pitch(), md.get_max_midi_pitch())
    visualize_together(padded_midi2,md.get_min_midi_pitch(), md.get_max_midi_pitch())
    plt.show()
