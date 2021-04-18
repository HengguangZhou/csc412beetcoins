import torch
import numpy as np
import mido
from dataset import MidiDataset


def convert_3d_to_2d(encoding3d, min_pitch, timestep_size=32):
    """
    Encoding3d is a an array of shape 4xtimestep_sizexP for some P.
    Want to return an array of shape Tx4 where the (i,j)th entry of the array
    is the midi pitch of the jth voice at time i.
    
    :param encoding3d: 3d tensor encoding of the midi, a tensor of shape 4xtimestep_sizexP
    :return: array of shape (T, 4), where T=timestep_size is the time
    """
    encoding2d = np.zeros((timestep_size, 4))
    ins, time, pitch = encoding3d.shape
    for i in range(0, ins):
        for t in range(0, time):
            vec = encoding3d[i, t, :]
            pitch_rel_idx = np.where(vec == 1)
            if len(pitch_rel_idx) != 0:
                # print(pitch_rel_idx)
                pitch = min_pitch + pitch_rel_idx[0][0]
            else:
                pitch = 0
            encoding2d[t, i] = pitch
    return torch.from_numpy(encoding2d)


def convert_2d_to_3d(encoding2d, min_pitch, pitch_range, timestep_size=32):
    """
    Encoding2d is a an array of shape IxT (4xtimestep_size) for some P where the (i,j)th entry of the array
    is the midi pitch of the ith voice at time j.
    Want to return an array of shape IxTxP.

    :param encoding2d: 2d tensor encoding of the midi, tensor of shape TxI
    :param min_pitch: minimum midi pitch number this piano roll contains
    :param pitch_range: range of midi pitch number this piano roll contains
    :return: 3d tensor encoding of the piano roll, tensor of shape IxTxP
    """
    ins, time = encoding2d.shape

    encoding3d = np.zeros((4, timestep_size, pitch_range))  # IxTxP
    for i in range(0, ins):
        for t in range(0, time):
            pitch_idx = int(encoding2d[i, t] - min_pitch)
            encoding3d[i, t, pitch_idx] = 1

    return torch.from_numpy(encoding3d)

# Adapted from https://github.com/kevindonoghue/coconet-pytorch
def piano_roll2d_to_midi(piece):
    """
    piece is a an array of shape (T, 4) for some T.
    The (i,j)th entry of the array is the midi pitch of the jth voice at time i. It's an integer in range(128).
    outputs a mido object mid that you can convert to a midi file by called its .save() method
    """
    piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0)

    bpm = 50
    microseconds_per_beat = 60 * 1000000 / bpm

    mid = mido.MidiFile()
    tracks = {'soprano': mido.MidiTrack(), 'alto': mido.MidiTrack(),
              'tenor': mido.MidiTrack(), 'bass': mido.MidiTrack()}
    past_pitches = {'soprano': np.nan, 'alto': np.nan,
                    'tenor': np.nan, 'bass': np.nan}
    delta_time = {'soprano': 0, 'alto': 0, 'tenor': 0, 'bass': 0}

    # create a track containing tempo data
    metatrack = mido.MidiTrack()
    metatrack.append(mido.MetaMessage('set_tempo',
                                      tempo=int(microseconds_per_beat), time=0))
    mid.tracks.append(metatrack)

    # create the four voice tracks
    for voice in tracks:
        mid.tracks.append(tracks[voice])
        tracks[voice].append(mido.Message(
            'program_change', program=52, time=0))

    # add notes to the four voice tracks
    for i in range(len(piece)):
        pitches = {'soprano': piece[i, 0], 'alto': piece[i, 1],
                   'tenor': piece[i, 2], 'bass': piece[i, 3]}
        for voice in tracks:
            if np.isnan(past_pitches[voice]):
                past_pitches[voice] = None
            if np.isnan(pitches[voice]):
                pitches[voice] = None
            if pitches[voice] != past_pitches[voice]:
                if past_pitches[voice]:
                    tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
                if pitches[voice]:
                    tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
            past_pitches[voice] = pitches[voice]
            # 480 ticks per beat and each line of the array is a 16th note
            delta_time[voice] += 120

    return mid


if __name__ == '__main__':
    md = MidiDataset('./data/jsb/jsb-chorales-16th.pkl')

    test_midi = md[0]
    # print(test_midi[0].shape)

    test_midi2d = convert_3d_to_2d(test_midi[1], md.get_min_midi_pitch())
    # mido = piano_roll2d_to_midi(test_midi2d)
    #
    # mido.save('testmidi.mid')

    # Testing if converting 2d and 3d works
    print(test_midi2d.shape)
    # print(test_midi2d.transpose().shape)
    test_midi3d = convert_2d_to_3d(torch.transpose(test_midi2d, 0, 1), md.get_min_midi_pitch(), md.get_pitch_range())
    print(test_midi3d.shape)
    test_midi2d2 = convert_3d_to_2d(test_midi3d, md.get_min_midi_pitch())

    # Testing if the 2d and 3d conversion are equal
    print(torch.equal(test_midi2d, test_midi2d2))

    # Try generating again
    mido2 = piano_roll2d_to_midi(test_midi2d2)
    mido2.save('music_of_heaven2.mid')
