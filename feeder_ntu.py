import numpy as np
import torch
import random
from torch.utils.data import Dataset

from . import tools

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        aug_method='a123489'
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.aug_method = aug_method
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.intra_p = 0.5
        self.inter_p = 0.0
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # # data: N C V T M
        # npz_data = np.load(self.data_path)
        # if self.split == 'train':
        #     self.data = npz_data['x_train']
        #     self.label = np.where(npz_data['y_train'] > 0)[1]
        #     self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        # elif self.split == 'test':
        #     self.data = npz_data['x_test']
        #     self.label = np.where(npz_data['y_test'] > 0)[1]
        #     self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        # else:
        #     raise NotImplementedError('data split only supports train/test')
        # N, T, _ = self.data.shape
        # # 原版：self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        # self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
        if self.split == 'train':
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        in_channels = 3
        if self.random_choose == True:
            in_channels = 9
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            return torch.zeros((in_channels, self.window_size, 17, 2)), label, index
        # reshape Tx(MVC) to CTVM
        data_numpy, index_t = tools.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                       self.window_size, self.window_size)
        # if self.random_rot:
        #     data_numpy = tools.random_rot(data_numpy)
        # chosen_augmentation = index % 4
        # if chosen_augmentation == 0:
        #     data_numpy = tools.random_rot(data_numpy)
        # elif chosen_augmentation == 1:
        #     data_numpy = tools.random_shift(data_numpy)
        # elif chosen_augmentation == 2:
        #     data_numpy = tools.random_move(data_numpy)

        p = np.random.rand(1)
        if p < self.intra_p:

            if 'a' in self.aug_method:
                if np.random.rand(1) < 0.5:
                    data_numpy = data_numpy[:, :, :, np.array([1, 0])]

            if '1' in self.aug_method:
                data_numpy = tools.shear(data_numpy, p=0.5)
            if '2' in self.aug_method:
                data_numpy = tools.rotate(data_numpy, p=0.5)
            if '3' in self.aug_method:
                data_numpy = tools.scale(data_numpy, p=0.5)
            if '4' in self.aug_method:
                data_numpy = tools.spatial_flip(data_numpy, p=0.5)
            if '5' in self.aug_method:
                data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
            if '6' in self.aug_method:
                data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
            if '7' in self.aug_method:
                data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
            if '8' in self.aug_method:
                data_numpy = tools.drop_axis(data_numpy, p=0.5)
            if '9' in self.aug_method:
                data_numpy = tools.drop_joint(data_numpy, p=0.5)

        # inter-instance augmentation
        elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
            adain_idx = random.choice(np.where(self.label == label)[0])
            data_adain = self.data[adain_idx]
            data_adain = np.array(data_adain)
            f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
            t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int64)
            data_adain = data_adain[:, t_idx]
            data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

        else:
            data_numpy = data_numpy.copy()

        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        if in_channels == 9:
            data_numpy = tools.get_JA(data_numpy)
        
        if isinstance(data_numpy, np.ndarray):
            data_numpy = torch.from_numpy(data_numpy)
        
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
