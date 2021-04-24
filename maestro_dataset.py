import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from convert import convert_3d_to_2d, midi_to_piano_roll
from dataset import get_concat_mask, get_mask
import time
import random


class MaestroDataset(Dataset):
    def __init__(self, file_dir, timestep_len):
        self.file_dir = file_dir
        self.timestep_len = timestep_len
        self.midi_lst, self.composer_lst = self.process_data_from_csv()
        self.min_pitch = 21
        self.max_pitch = 108
        self.pitch_range = self.max_pitch - self.min_pitch + 1
        self.composer_map = self.map_composer_name_to_key()
        self.midi_to_composer = self.obtain_piano_rolls()

    def __len__(self):
        return len(self.midi_to_composer)

    def __getitem__(self, idx):
        info = self.midi_to_composer[idx]
        # Return the random data segment
        oh_data = torch.from_numpy(info[0]).float()
        data = info[1]
        mask = torch.from_numpy(get_concat_mask(oh_data))
        composer_name = info[2]
        label = self.composer_map[composer_name]

        return data, oh_data, mask, idx, label

    def find_min_max_pitch(self):  # Used to find min and max pitch in the dataset
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

    def obtain_piano_rolls(self):
        lst = []
        for idx in range(0, len(self.midi_lst)):
            midi_name = self.midi_lst[idx]
            composer_name = self.composer_lst[idx]
            piano_roll3d = self.separate_piano_roll_by_ts(midi_name)
            # from the list of 3d piano rolls, sample one 3d piano roll and convert it to 2d representation
            piano_roll3d_3 = piano_roll3d
            if len(piano_roll3d) > 1:
                piano_roll3d_3 = random.sample(piano_roll3d, 1)

            for i in piano_roll3d_3:
                piano_roll2d = convert_3d_to_2d(i, 0, self.timestep_len)  # Output is already a tensor
                lst.append((i, piano_roll2d, composer_name))

        return lst

    def separate_piano_roll_by_ts(self, midi_name):
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
    start_time = time.time()
    md = MaestroDataset('../data/maestro-v3.0.0', 128)
    a = md[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(a)
