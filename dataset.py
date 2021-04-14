import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class MidiDataset(Dataset):
    def __init__(self, data_path, fold='train'):
        """
        :param data_path: file name of the pickle data to load midi info from
        :param fold: either 'train', 'validate', or 'test'
        """
        self.data_path = data_path
        self.fold = fold

        with open(data_path, 'rb') as p:
            data = pickle.load(p, encoding="latin1")

        self.midi_datas = self.get_midis_array(data[self.fold])
        self.processed_midi_datas = self.get_masked_data(self.midi_datas)

    def __len__(self):
        return len(self.processed_midi_datas)

    def __getitem__(self, idx):
        return torch.from_numpy(self.processed_midi_datas[idx])

    def get_midis_array(self, data):
        midis = []
        for seq in data:
            midi = np.zeros((4, 128, 128))
            crop_data = seq[0:128]
            for ts_idx in range(0, len(crop_data)):
                ts = crop_data[ts_idx]
                for i in range(0, len(ts)):
                    midi[i, ts_idx, int(ts[i])] = 1
            midis.append(midi)

        return midis

    def get_masked_data(self, midis):
        all_masked_data = []
        for midi in midis:
            temp = []
            for i in midi:
                mask = self.get_mask()
                masked_data = np.multiply(i, mask)
                temp.append(np.concatenate((mask, masked_data), axis=0))
            all_masked_data.append(np.concatenate(temp, axis=0))

        return all_masked_data

    def get_mask(self):
        mask_idx = np.random.choice(128 * 128, size=np.random.choice(128 * 128) + 1, replace=False)
        mask = np.zeros(128 * 128, dtype=np.float32)
        mask[mask_idx] = 1.
        mask = mask.reshape((1, 128, 128))

        return mask


if __name__ == '__main__':
    md = MidiDataset('../data/JSB-Chorales-dataset/jsb-chorales-16th.pkl')
    itr = enumerate(md)

    for idx, data in itr:
        print(data)  # Tensor of 8x128x128
        print(data.shape)  # Should be 8x128x128
