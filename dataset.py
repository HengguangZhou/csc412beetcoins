import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms, utils
import random
import pretty_midi
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt


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
        # self.fake_images = self.load_midi_files(os.path.join(data_path, 'fake'))

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
            # midis.append(filename)
            # print(filename)
            midi_data = pretty_midi.PrettyMIDI(filename)
            for i in midi_data.instruments:
                one_hot_enc = self.convert_to_one_hot(i)
                # print(one_hot_enc)
                midis.append(one_hot_enc)
            # print(midi_data.instruments[0].notes)
        return midis

    def convert_to_one_hot(self, instrument):
        notes = instrument.notes
        # print(notes)
        num_notes = len(notes)
        max_notes = 1200
        max_keys = self.get_max_keys(notes)
        # print(num_notes)
        if num_notes > max_notes:
            num_notes = max_notes
        one_hot_enc = np.zeros((num_notes, max_keys))
        for i in range(num_notes):
            one_hot_enc[i, notes[i].pitch - 1] = 1

        return one_hot_enc

    def get_max_keys(self, notes):
        max_pitch = notes[0].pitch
        for i in notes:
            if i.pitch > max_pitch:
                max_pitch = i.pitch

        return max_pitch


if __name__ == '__main__':
    md = MidiDataset('../data/clean_midi.tar/clean_midi/')
