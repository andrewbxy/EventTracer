import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
from tqdm import *

import numpy as np
import torch
from torch.utils.data import Dataset

from tile_based_storage import TileBasedStorage, block_lin_log

class EventDataset(Dataset):
    def __init__(
        self,
        data_path,
        positions,
        max_len,
        batch_size,
        randomize,
        type='log',
        train=True,
    ):
        self.data_path = data_path
        self.positions = positions
        self.max_len = max_len
        self.batch_size = batch_size
        self.randomize = randomize
        self.type = type
        self.train = train

        self.data_path = data_path
        image_dir = os.path.join(self.data_path, "Output")
        event_dir = os.path.join(self.data_path, "Events")
            
        self.inputTile = TileBasedStorage(
            [64, 64, 64],
            [20, 12, 62],
            image_dir,
            data_format="rgba",
        )
        self.targetTile = TileBasedStorage(
            [64, 64, 64],
            [20, 12, 62],
            event_dir,
            data_format="int8",
        )
        
        self.cache = None
        if self.max_len > 2000:
            self.pseudo_idx_lst = list(range(len(self.positions) * 2))
        else:
            self.pseudo_idx_lst = list(range(len(self.positions)))
        random.shuffle(self.pseudo_idx_lst)

    def __len__(self):
        if self.train:
            bpf = 64 * 64 // self.batch_size
        else:
            bpf = 1
        if self.max_len > 2000:
            return len(self.positions) * 2 * bpf
        else:
            return len(self.positions) * bpf

    def __getitem__(self, idx):
        if self.train:
            idx = self.pseudo_idx_lst[0]
        if self.max_len > 2000:
            row, col = self.positions[idx // 2]
        else:
            row, col = self.positions[idx]

        if self.cache is None or (not self.train):
            input_data = block_lin_log(self.inputTile.get_tile(row, col), self.type).numpy()
            gt_data = self.targetTile.get_tile(row, col)[:, :, :, 0].numpy()
        
            time_stamps, height, width = input_data.shape
            if self.max_len > 2000:
                idx_s, idx_e = self.max_len // 2 * (idx % 2), self.max_len // 2 * (idx % 2 + 1)
                input_data = input_data.reshape(time_stamps, height * width).transpose()[:,idx_s:idx_e]
                gt_data = gt_data.reshape(time_stamps, height * width).transpose()[:,idx_s:idx_e]
            else:
                input_data = input_data.reshape(time_stamps, height * width).transpose()[:,:self.max_len]
                gt_data = gt_data.reshape(time_stamps, height * width).transpose()[:,:self.max_len]
        else:
            input_data = self.cache[0]
            gt_data = self.cache[1]

        total_rows = input_data.shape[0]
        if self.randomize:
            selected_indices = np.random.choice(total_rows, self.batch_size, replace=total_rows < self.batch_size)
        else:
            selected_indices = np.arange(self.batch_size)

        # Select batch rows all at once
        input_batch = input_data[selected_indices, :]
        gt_batch = gt_data[selected_indices, :]
        
        if input_data.shape[0] - selected_indices.shape[0] >= self.batch_size and self.train:
            remain_input = np.delete(input_data, selected_indices, axis=0)
            remain_gt = np.delete(gt_data, selected_indices, axis=0)
            self.cache = [remain_input, remain_gt]
        else:
            self.cache = None
            self.pseudo_idx_lst.pop(0)
            if len(self.pseudo_idx_lst) == 0:
                if self.max_len > 2000:
                    self.pseudo_idx_lst = list(range(len(self.positions) * 2))
                else:
                    self.pseudo_idx_lst = list(range(len(self.positions)))
                random.shuffle(self.pseudo_idx_lst)

        return (
            torch.from_numpy(input_batch).permute(1, 0).unsqueeze(-1),
            torch.from_numpy(gt_batch).permute(1, 0).unsqueeze(-1),
        )