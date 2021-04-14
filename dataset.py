import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class MidiDataset(Dataset):
    def __init__(self, data_path, fold='train'):
        """
        :param data_path: file name of the pickle data to load midi info from
        :param fold: either 'train', 'valid', or 'test'
        """
        self.data_path = data_path
        self.fold = fold
        self.max_midi_pitch = -np.inf
        self.min_midi_pitch = np.inf
        with open(data_path, 'rb') as p:
            data = pickle.load(p, encoding="latin1")
        self.get_max_min_pitch(data[self.fold])
        self.pitch_range = self.max_midi_pitch - self.min_midi_pitch + 1
        self.midi_datas = self.get_midis_array(data[self.fold])
        self.midi_masks = self.get_concat_mask(self.midi_datas)
        # self.processed_midi_datas = self.get_masked_data(self.midi_datas)

    def __len__(self):
        return len(self.midi_datas)

    def __getitem__(self, idx):
        original_data = torch.from_numpy(self.midi_datas[idx])
        mask = torch.from_numpy(self.midi_masks[idx])
        return original_data, mask

    def get_max_min_pitch(self, data):
        for seq in data:
            crop_data = seq[0:128]
            for ts_idx in range(0, len(crop_data)):
                ts = crop_data[ts_idx]
                if ts:
                    if max(ts) > self.max_midi_pitch:
                        self.max_midi_pitch = int(max(ts))
                    if min(ts) < self.min_midi_pitch:
                        self.min_midi_pitch = int(min(ts))

    def get_midis_array(self, data):
        midis = []
        for seq in data:
            midi = np.zeros((4, 128, self.pitch_range))
            crop_data = seq[0:128]
            for ts_idx in range(0, len(crop_data)):
                ts = crop_data[ts_idx]
                for i in range(0, len(ts)):
                    # change max and min pitch
                    pitch_idx = int(ts[i]) - self.min_midi_pitch
                    midi[i, ts_idx, pitch_idx] = 1
            midis.append(midi)

        return midis

    def get_concat_mask(self, midis):
        all_masks = []
        for midi in midis: # for each file we want a mask of 4x128xP
            temp = []
            for i in range(len(midi)):
                mask = self.get_mask()
                temp.append(mask)
            all_masks.append(np.concatenate(temp, axis=0))

        return all_masks

    # def get_masked_data(self, midis):
    #     all_masked_data = []
    #     for midi in midis:
    #         temp = []
    #         for i in midi:
    #             mask = self.get_mask()
    #             masked_data = np.multiply(i, mask)
    #             temp.append(np.concatenate((mask, masked_data), axis=0))
    #         all_masked_data.append(np.concatenate(temp, axis=0))
    #
    #     return all_masked_data

    def get_mask(self):
        # mask size is 1 x 128 x P
        mask_idx = np.random.choice(128 * self.pitch_range, size=np.random.choice(128 * self.pitch_range) + 1, replace=False)
        mask = np.zeros(128 * self.pitch_range, dtype=np.float32)
        mask[mask_idx] = 1.
        mask = mask.reshape((1, 128, self.pitch_range))

        return mask


if __name__ == '__main__':
    md = MidiDataset('../jsb/jsb-chorales-16th.pkl')
    itr = enumerate(md)

    for idx, data in itr:
        original_data, mask = data
        # P = pitch range = max_pitch - min_pitch + 1
        # print(original_data)  # Tensor of 4x128xP
        # print(mask)  # Tensor of 8x128xP
        print(original_data.shape)  # Should be 4x128xP
        print(mask.shape)  # Should be 8x128xP
        break
