import copy
import math

import numpy as np
import torch
import itertools
from torch.utils import data
import scipy.io as scio


class EMGDataset(data.Dataset):
    def __init__(self, clip_length, step, human, with_remix, get_first_part=False, with_extra_small=False,
                 first_is_train=True, first_part_rate=0, input_second_index=None, logger_setup=None):

        self.with_remix = with_remix
        self.get_first_part = get_first_part
        self.with_extra_small = with_extra_small
        self.second_index = []
        meta = scio.loadmat('Dataset/matlab.mat')
        # fittedModel = model.fit()
        # predicted = model.predict()
        name_index = meta.get('S')['name'][0]
        file_index = meta.get('S')['file'][0]

        X_train = []
        Y_train = []
        for i in range(len(human)):
            name = name_index[human[i]][0]
            file = file_index[human[i]][0]

            nam_index = file['nam']
            data_index = file['data']
            # for j in range(len(nam_index) - 1):
            for j in range(1):
                nam = nam_index[j][0]
                data = data_index[j][0]

                d_index = data['d']
                start_index = data['start_index']
                # split_num=int(first_part_rate * len(d_index))
                split_num = int(first_part_rate * len(d_index))
                if first_is_train:
                    split_num = math.ceil(first_part_rate * len(d_index))

                total_list = np.arange(len(d_index)).tolist()
                total_combines = list(itertools.combinations(total_list, split_num))[np.random.randint(len(d_index))]
                first_index = list(total_combines)
                second_index = list(set(total_list).difference(set(first_index)))
                self.second_index.append(copy.deepcopy(second_index))

                if input_second_index is not None:
                    second_index = input_second_index[j]

                if not self.get_first_part:
                    start = 0
                    end = len(d_index) - split_num
                else:
                    start = 0
                    end = split_num
                    # end = int(first_part_rate * len(d_index))
                # for k in range(len(d_index) - 1):
                #
                # for this_index in range(start, end):
                for this_index in range(1):
                    if self.get_first_part:
                        k = first_index[this_index]
                    else:
                        k = second_index[this_index]
                    k = 4
                    tmp_start_index = start_index[k].item()
                    if tmp_start_index < 1000:
                        tmp_start_index = 1000
                    data_matrix = d_index[k][tmp_start_index - 900:]
                    data_temper = []
                    if self.with_extra_small:
                        if nam == 'sit':
                            data_matrix = data_matrix[:2250 + 900]
                        if nam == 'stand':
                            data_matrix = data_matrix[:2250 + 900]
                        if nam == 'walk_left' or nam == 'walk_right':
                            data_matrix = data_matrix[:3000 + 900]
                        if nam == 'stair':
                            data_matrix = data_matrix[:1500 + 900]
                    step_threshold = (data_matrix.shape[0] - clip_length) / step + 1
                    X_train_start = len(X_train)
                    for l in range(int(step_threshold)):
                        if nam == 'sit':
                            data_clip = data_matrix[l * step:clip_length + l * step, 1:13]
                            # if (data_clip * data_clip).sum() / (data_clip.shape[0]) > 6000:
                            data_temper.append(data_clip)
                            if len(data_temper) == 10:
                                X_train.append(copy.deepcopy(data_temper))
                                Y_train.append([0])
                                data_temper.pop(0)
                        if nam == 'stand':
                            data_clip = data_matrix[l * step:clip_length + l * step, 1:13]
                            # if (data_clip * data_clip).sum() / (data_clip.shape[0]) > 6000:
                            data_temper.append(data_clip)
                            if len(data_temper) == 10:
                                X_train.append(copy.deepcopy(data_temper))
                                Y_train.append([1])
                                data_temper.pop(0)
                        if nam == 'walk_left' or nam == 'walk_right':
                            data_clip = data_matrix[l * step:clip_length + l * step, 1:13]
                            # if (data_clip * data_clip).sum() / (data_clip.shape[0]) > 6000:
                            data_temper.append(data_clip)
                            if len(data_temper) == 10:
                                X_train.append(copy.deepcopy(data_temper))
                                Y_train.append([2])
                                data_temper.pop(0)
                        if nam == 'stair':
                            data_clip = data_matrix[l * step:clip_length + l * step, 1:13]
                            # if (data_clip * data_clip).sum() / (data_clip.shape[0]) > 6000:
                            data_temper.append(data_clip)
                            if len(data_temper) == 10:
                                X_train.append(copy.deepcopy(data_temper))
                                Y_train.append([3])
                                data_temper.pop(0)

                    if logger_setup is not None:
                        logger_setup.info(
                            "Human {0} Motion {1} Seq {2} added from {3} to {4}".format(name, nam, k, X_train_start,
                                                                                        len(X_train) - 1))

        self.X = torch.tensor(
            np.abs(np.array(X_train).transpose([0, 1, 3, 2]).reshape(-1, 10, 12, clip_length, 1) * 0.001),
            dtype=float)
        self.Y = (torch.tensor(np.array([Y_train])))[0]

    def __getitem__(self, item):
        if self.with_remix:
            length = len(self.X)
            rand_idx = torch.randint(0, length, [1])
            rand_ratio = torch.rand([1]) * 0.2
            rand_idx_noise = torch.randint(0, length, [1])
            rand_ratio_noise = torch.rand([1]) * 0.05

            label = self.Y[item]
            while self.Y[rand_idx] != label:
                rand_idx = torch.randint(0, length, [1])

            merged = ((self.X[item] * (1 - rand_ratio) + self.X[rand_idx] * rand_ratio) * (1 - rand_ratio_noise) +
                      self.X[rand_idx_noise] * rand_ratio_noise)[0]
        else:
            merged = self.X[item]
            label = self.Y[item]
        return merged, label

    def __len__(self):
        return len(self.X)

    def get_train_test(self, index):
        result_train = copy.deepcopy(self)
        result_train.X = self.X[index.nonzero(), :][0]
        result_train.Y = self.Y[0, index.nonzero(), :][0]
        return result_train

    def get_second_index(self):
        return self.second_index
