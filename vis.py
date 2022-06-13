import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from utils import grouping, revert_frame

gt = {}
with open('result/vis/gt.txt') as f:
    lines = f.readlines()
for line in lines:
    name, start, end, label = line.split()
    if name not in gt:
        gt[name] = []
    gt[name].append([start, end, label])

old_dict = np.load('result/vis/old.npy', allow_pickle=True).item()
new_dict = np.load('result/vis/new.npy', allow_pickle=True).item()

old_scores, new_scores = {}, {}
for key, value in old_dict.items():
    old_scores[key.decode()] = torch.softmax(value['cas'], dim=-1) * value['attn']
for key, value in new_dict.items():
    new_scores[key.decode()] = torch.softmax(value['cas'], dim=-1) * value['attn']

for key in tqdm(gt.keys()):
    old_score = old_scores[key].squeeze(dim=0).cpu()
    new_score = new_scores[key].squeeze(dim=0).cpu()
    gts = gt[key]
    num_frames = old_score.shape[0] * 16
    frame_indexes = np.arange(0, num_frames)
    old_score = np.clip(revert_frame(old_score.numpy(), num_frames), a_min=0.0, a_max=1.0)
    new_score = np.clip(revert_frame(new_score.numpy(), num_frames), a_min=0.0, a_max=1.0)

    fig, axs = plt.subplots(5, 1, figsize=(7, 3))
    for i in range(len(gts)):
        start, end, label = gts[i]
        start, end, label = int(start), int(end), int(label)
        start, end = start * 16, min(end * 16, num_frames - 1)
        count = np.zeros(num_frames)
        count[start:end] = 1
        if start < end:
            axs[0].fill_between(frame_indexes, count, color='green')
            # axs[0].plot(frame_indexes, count, color='green')

    axs[2].plot(frame_indexes, old_score[:, label], color='blue')
    axs[4].plot(frame_indexes, new_score[:, label], color='red')

    old_proposals = grouping(np.where(old_score[:, label] >= 0.2)[0])
    for proposal in old_proposals:
        if len(proposal) >= 2:
            start, end = proposal[0], proposal[-1]
            end = min(end, num_frames - 1)
            count = np.zeros(num_frames)
            count[start:end] = 1
            if start < end:
                axs[1].fill_between(frame_indexes, count, color='blue')
                # axs[1].plot(frame_indexes, count, color='blue')

    new_proposals = grouping(np.where(new_score[:, label] >= 0.2)[0])
    for proposal in new_proposals:
        if len(proposal) >= 2:
            start, end = proposal[0], proposal[-1]
            end = min(end, num_frames - 1)
            count = np.zeros(num_frames)
            count[start:end] = 1
            if start < end:
                axs[3].fill_between(frame_indexes, count, color='red')
                # axs[3].plot(frame_indexes, count, color='red')

    plt.setp(axs, xticks=[], yticks=[], xlim=(0, num_frames), ylim=(0, 1))

    save_name = 'result/vis/{}.pdf'.format(key)
    plt.savefig(save_name, bbox_inches='tight')
    plt.cla()
    plt.close('all')
