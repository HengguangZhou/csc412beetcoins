import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms, utils
import random
import pretty_midi


class MidiDataset(Dataset):
    def __init__(self, data_path, size=50000, transform=None):
        """
        :param data_path: Root directory of the image data
        """
        self.data_path = data_path
        self.size = size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), ])
        else:
            self.transform = transform
        self.midi_datas = self.load_midi_files(os.path.join(data_path, '.38 Special'))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        pass
        # img1, img2, np_label = self.get_image_pair()
        # real_image = self.transform(img1)
        # fake_image = self.transform(img2)
        # label = torch.from_numpy(np_label)

        # return real_image, fake_image, label

    def load_midi_files(self, path):
        midis = []

        for filename in glob.glob(os.path.join(path, '*.mid')):
            midi = {}
            # midis.append(filename)
            # print(filename)
            midi_data = pretty_midi.PrettyMIDI(filename)
            for i in midi_data.instruments:
                midi[i.name] = i.get_piano_roll()
            # print(midi_data.instruments[0].notes)
            midis.append(midi)
        print(midis)
        return midis

    def get_max_keys(self, notes):
        max_pitch = notes[0].pitch
        for i in notes:
            if i.pitch > max_pitch:
                max_pitch = i.pitch

        return max_pitch


if __name__ == '__main__':
    md = MidiDataset('../clean_midi/')
