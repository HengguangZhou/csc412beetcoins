import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pretty_midi
from dataset import MidiDataset
import midi2audio
from models.coconet import coco_decoder
import math

# Adapted from https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
def dataset_with_indices(cls):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index
    return type(cls.__name__, (cls,), {'__getitem__': __getitem__, })

# Adapted from https://github.com/facebookresearch/DeepSDF/blob/master/reconstruct.py
def adjust_learning_rate(
    initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='coconet')
    parser.add_argument('--data', type=str, required=True)
    # parser.add_argument('--val_data', type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weights_dir", type=str, default=".\\weights\\")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument('--input_channels', type=int, default=9)
    parser.add_argument('--time_steps', type=int, default=128)
    parser.add_argument('--all_perm', type=bool, default=False)

    opts = parser.parse_args()
    print(opts)

    writer = SummaryWriter()

    weights_path = os.path.join(opts.weights_dir,
                                opts.model)

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'coconet':
        model = coco_decoder(in_channels=9, hidden_channels=opts.time_steps)

    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters())}")
    model = model.to(device)

    train_midi = MidiDataset(opts.data, fold='train', timestep_len=opts.time_steps, all_comb=opts.all_perm)
    # train_midi = dataset_with_indices(train_midi)
    # test_midi = MidiDataset(opts.data, fold='valid', timestep_len=opts.time_steps, all_comb=opts.all_perm)

    train_midi_loader = DataLoader(dataset=train_midi,
                                    batch_size=opts.batch_size,
                                    shuffle=True)

    # test_midi_loader = DataLoader(dataset=test_midi,
    #                               batch_size=opts.batch_size,
    #                               shuffle=True)

    # latent_size = (opts.time_steps, 46)

    num_chorales = len(train_midi)
    latent_size = opts.time_steps * 46
    latents = torch.nn.Embedding(num_chorales, opts.time_steps * 46)
    torch.nn.init.normal_(
        latents.weight.data,
        0.0,
        1.0 / math.sqrt(latent_size),
    )
    # latents.requires_grad_()
    optimizer = torch.optim.Adam([{
                "params": model.parameters(),
                "lr": opts.lr,
                "weight_decay": 1e-5,
                },
                {
                "params": latents.parameters(),
                "lr": opts.lr,
                "weight_decay": 1e-5,
                },])


    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(opts.num_epochs):
        model.train()
        with tqdm(total=(len(train_midi) - len(train_midi) % opts.batch_size)) as t:
            t.set_description(f'train epoch: {epoch}/{opts.num_epochs - 1}')

            los = 0
            for idx, data in enumerate(train_midi_loader):
                midi, oh_midi, mask, ind = data
                B, I, T, P = oh_midi.shape
                midi, oh_midi, mask = midi.to(device), oh_midi.to(device), mask.to(device)
                # print(f"midi: {midi.dtype}, mask:{mask.dtype}")
                latent = latents(ind).to(device)
                pred = model(oh_midi, mask, latent)

                loss = criterion(pred.reshape(-1, P), midi.reshape(-1).long())

                optimizer.zero_grad()
                los += loss.item()
                loss.backward()
                optimizer.step()

                t.update(midi.shape[0])

                t.set_postfix(loss='{:.6f}'.format(los / (idx + 1)))

            print("\nloss: {:.2f}".format(los / len(train_midi) * opts.batch_size))
            writer.add_scalar(f'CELoss/train',
                              los / len(train_midi) * opts.batch_size, epoch)

        torch.save(model.state_dict(), os.path.join(weights_path,
                                    f"{opts.model}_latest.pth"))

        # model.eval()
        #
        # with tqdm(total=(len(test_midi) - len(test_midi) % opts.batch_size)) as t:
        #     t.set_description(f'val epoch: {epoch}/{opts.num_epochs - 1}')
        #     for idx, data in enumerate(test_midi_loader):
        #         midi, label = data.to(device)
        #         pred = model(midi)
        #
        #         t.update(midi.shape[0])
        #todo: save intermediate results
