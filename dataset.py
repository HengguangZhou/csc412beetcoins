import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class MidiDataset(Dataset):
    def __init__(self, data_path, fold='train', timestep_len = 32, all_comb = False):
        """
        :param data_path: file name of the pickle data to load midi info from
        :param fold: either 'train', 'valid', or 'test'
        """
        self.all_comb = all_comb
        self.data_path = data_path
        self.timestep_len = timestep_len
        self.fold = fold
        self.max_midi_pitch = -np.inf
        self.min_midi_pitch = np.inf
        with open(data_path, 'rb') as p:
            data = pickle.load(p, encoding="latin1")
        self.get_max_min_pitch(data[self.fold])
        self.pitch_range = self.max_midi_pitch - self.min_midi_pitch + 1
        self.oh_midi_datas, self.midi_datas = self.get_midis_array(self.separate_data(data[self.fold]))

        # self.midi_masks = self.get_concat_mask(self.midi_datas)
        # self.processed_midi_datas = self.get_masked_data(self.midi_datas)

    def __len__(self):
        return len(self.midi_datas)

    def __getitem__(self, idx):
        oh_data = torch.from_numpy(self.oh_midi_datas[idx]).float()
        data = torch.from_numpy(self.midi_datas[idx]).float()
        mask = torch.from_numpy(get_concat_mask(oh_data))
        return data, oh_data, mask, idx

    def get_max_min_pitch(self, data):
        for seq in data:
            for ts_idx in range(0, len(seq)):
                ts = seq[ts_idx]
                if ts:
                    if max(ts) > self.max_midi_pitch:
                        self.max_midi_pitch = int(max(ts))
                    if min(ts) < self.min_midi_pitch:
                        self.min_midi_pitch = int(min(ts))

    def separate_data(self, data):
        datas = []
        for seq in data:
            seq_len = len(seq)
            if self.all_comb:
                i = 0
                while i + self.timestep_len <= seq_len:
                    datas.append(seq[i:i+self.timestep_len])
                    i += 1
            else:
                if seq_len > self.timestep_len:
                    starting_idx = np.random.randint(0, seq_len - self.timestep_len + 1)
                    timestep_seq = seq[starting_idx:starting_idx+self.timestep_len]
                else:
                    timestep_seq = seq
                    while len(timestep_seq) < self.timestep_len:
                        timestep_seq.append((0, 0, 0, 0))  # Pad the sequence to desired timestep length with 0
                datas.append(timestep_seq)
        return datas


    def get_midis_array(self, data):
        oh_midis = []
        midis = []
        for seq in data:
            midi = np.zeros((4, self.timestep_len))
            oh_midi = np.zeros((4, self.timestep_len, self.pitch_range))
            crop_data = seq[0:self.timestep_len]
            for ts_idx in range(0, len(crop_data)):
                ts = crop_data[ts_idx]
                for i in range(0, len(ts)):
                    # change max and min pitch
                    pitch_idx = int(ts[i]) - self.min_midi_pitch
                    midi[i, ts_idx] = pitch_idx
                    oh_midi[i, ts_idx, pitch_idx] = 1
            oh_midis.append(oh_midi)
            midis.append(midi)
        return oh_midis, midis


    # def get_concat_mask(self, midis):
    #     all_masks = []
    #     for midi in midis: # for each file we want a mask of 4x32xP
    #         temp = []
    #         for i in range(len(midi)):
    #             mask = self.get_mask()
    #             temp.append(mask)
    #         all_masks.append(np.concatenate(temp, axis=0))
    #
    #     return all_masks

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

    # def get_mask(self):
    #     # mask size is 1 x data_len x P
    #     mask_idx = np.random.choice(self.timestep_len * self.pitch_range, size=np.random.choice(self.timestep_len * self.pitch_range) + 1, replace=False)
    #     mask = np.zeros(self.timestep_len * self.pitch_range, dtype=np.float32)
    #     mask[mask_idx] = 1.
    #     mask = mask.reshape((1, self.timestep_len, self.pitch_range))
    #
    #     return mask

    def get_min_midi_pitch(self):
        return self.min_midi_pitch

    def get_max_midi_pitch(self):
        return self.max_midi_pitch

    def get_pitch_range(self):
        return self.pitch_range

    def get_timeset_len(self):
        return self.timestep_len


def get_concat_mask(midi):
    # want a mask of 4x32xP
    temp = []
    for i in range(len(midi)):
        temp.append(get_mask(midi))

    return np.concatenate(temp, axis=0)


def get_mask(midi):
    # mask size is 1 x data_len x P
    ins, timestep_len, pitch_range = midi.shape
    mask_idx = np.random.choice(timestep_len * pitch_range,
                                size=np.random.choice(timestep_len * pitch_range) + 1, replace=False)
    mask = np.zeros(timestep_len * pitch_range, dtype=np.float32)
    mask[mask_idx] = 1.
    mask = mask.reshape((1, timestep_len, pitch_range))

    return mask


if __name__ == '__main__':
    md = MidiDataset('../jsb/jsb-chorales-16th.pkl')
    itr = enumerate(md)

    for idx, data in itr:
        data, oh_data, mask, data_idx = data
        # P = pitch range = max_pitch - min_pitch + 1
        # print(original_data)  # Tensor of 4x128xP
        # print(mask)  # Tensor of 8xtsxP
        print(data.shape)  # Should be 4xtsxP
        print(mask.shape)  # Should be 8xtsxP
        break
