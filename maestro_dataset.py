import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from convert import convert_3d_to_2d
from dataset import get_concat_mask, get_mask
import pickle
import json
import os


class MaestroDataset(Dataset):
    def __init__(self, file_dir, timestep_len):
        self.file_dir = file_dir
        self.timestep_len = timestep_len
        self.midi_lst, self.composer_lst = self.process_data_from_csv()
        self.min_pitch = 21
        self.max_pitch = 108
        self.pitch_range = self.max_pitch - self.min_pitch + 1
        self.composer_map = self.map_composer_name_to_key()
        # This is the array that contains tuples of three elements: 3d_x, 2d_x, and composer label
        if os.path.exists('data.pickle') and os.path.getsize('data.pickle') > 0:
            with open('data.pickle', 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                self.piano_roll_by_composer = pickle.load(f)
        else:
            self.piano_roll_by_composer = self.separate_midi_into_classes()
            with open('data.pickle', 'wb') as f:
                pickle.dump(self.piano_roll_by_composer, f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.piano_roll_by_composer)

    def __getitem__(self, idx):
        info = self.piano_roll_by_composer[idx]
        oh_data = torch.from_numpy(info[0]).float()
        data = info[1]
        mask = torch.from_numpy(get_concat_mask(oh_data))
        composer_name = info[2]
        label = self.composer_map[composer_name]

        return data, oh_data, mask, idx, label

    def find_min_max_pitch(self): # Used to find min and max pitch in the dataset
        min_pitch = 127
        max_pitch = 0
        keep_iter = True
        for i in self.midi_lst:
            if keep_iter:
                pm = pretty_midi.PrettyMIDI(self.file_dir + '/' + i)
                for ins in pm.instruments:
                    all_notes = ins.notes
                    for note in all_notes:
                        if note.pitch > max_pitch:
                            max_pitch = note.pitch
                        if note.pitch < min_pitch:
                            min_pitch = note.pitch
                        if min_pitch == 0 and max_pitch == 127:
                            keep_iter = False
            else:
                break
        print("min_pitch: {}".format(min_pitch))
        print("max_pitch: {}".format(max_pitch))

        return min_pitch, max_pitch

    def process_data_from_csv(self):
        csv = pd.read_csv(self.file_dir+'/maestro-v3.0.0.csv')
        midi_name_lst = csv.midi_filename
        composer_name_lst = csv.canonical_composer
        return midi_name_lst, composer_name_lst

    def map_composer_name_to_key(self):
        composer_map = {}
        index = 0
        for name in self.composer_lst:
            if name not in composer_map:
                composer_map[name] = index
                index += 1

        return composer_map

    def separate_midi_into_classes(self):
        lst = []
        for midi_idx in range(0, len(self.midi_lst)):
            midi_name = self.midi_lst[midi_idx]
            composer_name = self.composer_lst[midi_idx]
            sep_piano_roll3d = self.separate_piano_roll_by_ts(midi_name, composer_name)
            # from the list of 3d piano roll, obtain the 2d piano roll
            for i in sep_piano_roll3d:
                piano_roll2d = convert_3d_to_2d(i, 0, self.timestep_len)  # Output is already a tensor
                lst.append((i, piano_roll2d, composer_name))

        return lst

    def separate_piano_roll_by_ts(self, midi_name, composer_name):
        pm = pretty_midi.PrettyMIDI(self.file_dir + '/' + midi_name)
        temp = []
        for ins in pm.instruments:
            ins_p = ins.get_piano_roll(fs=5)
            ins_p[ins_p > 0] = 1.0
            total_ts = ins_p.shape[1]
            # cut off the extra length at the end to make it multiple of self.timestep_len if needed
            if total_ts % self.timestep_len != 0:
                ins_p = ins_p[:, 0:total_ts - (total_ts % self.timestep_len)]
            temp.append(ins_p[self.min_pitch:self.max_pitch+1, :].transpose())

        if len(temp) <= 4:
            final_piano_roll = np.stack(temp, axis=0)
            diff = 4 - len(temp)
            if diff > 0:  # Pad extra layers of zeros for the final piano roll if less than 4 levels
                final_piano_roll = np.concatenate([final_piano_roll,
                                                   np.zeros((diff, final_piano_roll.shape[1], self.pitch_range))], axis=0)
        else:  # More than 4 tracks, so choose 4 tracks only
            final_piano_roll = np.stack(temp[0:4], axis=0)

        # Split the whole piano roll into chunks of desired length
        ins_p_lst = np.split(final_piano_roll, final_piano_roll.shape[1] // self.timestep_len, axis=1)

        return ins_p_lst


if __name__ == '__main__':
    md = MaestroDataset('../data/maestro-v3.0.0', 128)
    a = md[0]
    # print(a)
    # a = [np.zeros((2, 2)), np.ones((2, 2)), np.ones((2,2))+np.ones((2,2)), np.ones((2,2))+np.ones((2, 2))+np.ones((2,2))]
    # np.save("test", a)
    # with open('test.pickle', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
    # with open('data.pickle', 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     b = pickle.load(f)
    # # print(os.path.exists('test.pickle'))
    # print("load done")

    # b = [np.zeros((2,2)), np.ones((2,2)), np.ones((2,2))+np.ones((2,2)), np.ones((2,2))+np.ones((2,2))+np.ones((2,2))]
    #
    # with open('data.json', 'wb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     json.dump(b, f)
