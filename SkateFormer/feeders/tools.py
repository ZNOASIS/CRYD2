import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window, thres):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
        c_b = begin + bias
        c_e = end - bias
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)  # constraint cropped_length lower bound as thres
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
        c_b = begin + bias
        c_e = begin + bias + cropped_length

    # resize
    data = torch.tensor(data, dtype=torch.float)  # C, crop_t, V, M
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, cropped_length)  # V*M, C, crop_t
    data = F.interpolate(data, size=window, mode='linear', align_corners=False)  # V*M, C, T
    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float)
    index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()


def valid_crop_uniform(data_numpy, valid_frame_num, p_interval, window, thres):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        bias = int((1 - p) * valid_size / 2)

        if cropped_length < window:
            inds = np.arange(cropped_length)
        else:
            bids = np.array(
                [i * cropped_length // window for i in range(window + 1)])
            bst = bids[:window]
            inds = bst

        inds = inds + bias
        data = data_numpy[:, inds, :, :]

    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)

        if cropped_length < window:
            inds = np.arange(cropped_length)
        elif window <= cropped_length < 2 * window:
            basic = np.arange(window)
            inds = np.random.choice(window + 1, cropped_length - window, replace=False)
            offset = np.zeros(window + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bsize = np.diff(bids)
            bst = bids[:window]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + bias
        data = data_numpy[:, inds, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    index_t = torch.tensor(inds, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(inds))  # V*M, C, crop_t

    if len(inds) != window:
        data = F.interpolate(data, size=window, mode='linear', align_corners=False)  # V*M, C, T
        index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()

    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()


def scale(data_numpy, scale=0.2, p=0.5):
    if random.random() < p:
        scale = 1 + np.random.uniform(-1, 1, size=(3, 1, 1, 1)) * np.array(scale)
        return data_numpy * scale
    else:
        return data_numpy.copy()


''' AimCLR '''
transform_order = {'ntu': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]}


def subtract(data_numpy, p=0.5):
    joint = random.randint(0, 24)

    C, T, V, M = data_numpy.shape
    if random.random() < p:
        data_numpy_new = np.zeros((C, T, V, M))
        for i in range(V):
            data_numpy_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, joint, :]
        return data_numpy_new
    else:
        return data_numpy.copy()


def temporal_flip(data_numpy, index_t, p=0.5):
    C, T, V, M = data_numpy.shape
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return data_numpy[:, time_range_reverse, :, :], -index_t
    else:
        return data_numpy.copy(), index_t.copy()


def spatial_flip(data_numpy, p=0.5):
    if random.random() < p:
        index = transform_order['ntu']
        return data_numpy[:, :, index, :]
    else:
        return data_numpy.copy()


def rotate(data_numpy, axis=None, angle=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)

    if angle != None:
        angle_next = random.uniform(-angle, angle)
    else:
        angle_next = random.uniform(-30, 30)

    if random.random() < p:
        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, math.cos(angle), math.sin(angle)],
                          [0, -math.sin(angle), math.cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[math.cos(angle), 0, -math.sin(angle)],
                          [0, 1, 0],
                          [math.sin(angle), 0, math.cos(angle)]])
        # z
        if axis_next == 2:
            R = np.array([[math.cos(angle), math.sin(angle), 0],
                          [-math.sin(angle), math.cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def shear(data_numpy, s1=None, s2=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        if s1 != None:
            s1_list = s1
        else:
            s1_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        if s2 != None:
            s2_list = s2
        else:
            s2_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

        R = np.array([[1, s1_list[0], s2_list[0]],
                      [s1_list[1], 1, s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def drop_axis(data_numpy, axis=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)

    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp
    else:
        return data_numpy.copy()


def drop_joint(data_numpy, joint_list=None, time_range=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        if joint_list != None:
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, joint_list)
            joint_list_ = sorted(joint_list_)
        else:
            random_int = random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)

        if time_range != None:
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, time_range)
            time_range_ = sorted(time_range_)
        else:
            random_int = random.randint(16, 32)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp
    else:
        return data_numpy.copy()


def gaussian_noise(data_numpy, mean=0, std=0.05, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy.copy()


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 2], p=0.5):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)
        self.p = p

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x).double()
        if prob < self.p:
            x = x.permute(3, 0, 2, 1)
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2)), groups=self.channels)
            x = x.permute(1, -1, -2, 0)

        return x.numpy()


def gaussian_filter(data_numpy, kernel=15, sig_list=[0.1, 2], p=0.5):
    g = GaussianBlurConv(3, kernel, sig_list, p)
    return g(data_numpy)


''' Skeleton AdaIN '''
def skeleton_adain_bone_length(input, ref): # C T V M
    eps = 1e-5
    center = 1
    ref_c = ref[:, :, center, :]

    # joint to bone (joint2bone)
    j2b = joint2bone()
    bone_i = j2b(input) # C T V M
    bone_r = j2b(ref)

    bone_length_i = np.linalg.norm(bone_i, axis=0) # T V M
    bone_length_r = np.linalg.norm(bone_r, axis=0)

    bone_length_scale = (bone_length_r + eps) / (bone_length_i + eps) # T V M
    bone_length_scale = np.expand_dims(bone_length_scale, axis=0) # 1 T V M

    bone_i = bone_i * bone_length_scale

    # bone to joint (bone2joint)
    b2j = bone2joint()
    joint = b2j(bone_i, ref_c)
    return joint


class joint2bone(nn.Module):
    def __init__(self):
        super(joint2bone, self).__init__()
        self.pairs = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)]

    def __call__(self, joint):
        bone = np.zeros_like(joint)
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
        return bone


class bone2joint(nn.Module):
    def __init__(self):
        super(bone2joint, self).__init__()
        self.center = 0
        self.pairs_1 = [(10, 8)]
        self.pairs_2 = [(8, 6), (9, 7), (7, 5)]
        self.pairs_3 = [(15, 13), (13, 11), (16, 14), (14, 12)]
        self.pairs_4 = [(11, 5), (12, 6), (11, 12)]
        self.pairs_5 = [(5, 6), (5, 0), (6, 0)]
        self.pairs_6 = [(1, 0), (2, 0), (3, 1), (4, 2)]

    def __call__(self, bone, center):
        joint = np.zeros_like(bone)
        joint[:, :, self.center, :] = center
        for v1, v2 in self.pairs_1:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_2:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_3:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_4:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_5:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_6:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        return joint


def to_motion(input): # C T V M
    C, T, V, M = input.shape
    motion = np.zeros_like(input)
    motion[:, :T - 1] = np.diff(input, axis=1)
    return motion

pairs_local = {'fsd':
                   (
                       (0, 1, 2), (1, 0, 3), (2, 0, 4), (3, 1, 1), (4, 2, 2), (5, 6, 7), (6, 5, 8), (7, 5, 9),
                       (8, 6, 10), (9, 7, 7),
                       (10, 8, 8), (11, 12, 13), (12, 11, 14), (13, 11, 15), (14, 12, 16), (15, 13, 11), (16, 14, 12)
                   )
}

pairs_center1 = {'fsd':
                     (
                         (0, 11, 12), (1, 6, 12), (2, 5, 11), (3, 1, 1), (4, 2, 2), (5, 6, 12), (6, 5, 11), (7, 5, 11),
                         (8, 6, 12), (9, 5, 11),
                         (10, 6, 12), (11, 6, 12), (12, 5, 11), (13, 5, 11), (14, 6, 12), (15, 5, 11), (16, 6, 12)
                     )
}

pairs_center2 = {'fsd':
                     (
                         (12, 11, 0), (12, 6, 1), (11, 5, 2), (1, 3, 3), (2, 4, 4), (12, 6, 5), (11, 5, 6), (11, 5, 7),
                         (12, 6, 8), (11, 5, 9),
                         (12, 6, 10), (12, 6, 11), (11, 5, 12), (11, 5, 13), (12, 6, 14), (11, 5, 15), (12, 6, 16)
                     )
}

pairs_hands = {'fsd':
                   (
                       (0, 9, 10), (1, 9, 10), (2, 9, 10), (3, 9, 10), (4, 9, 10), (5, 9, 10), (6, 9, 10), (7, 9, 10),
                       (8, 9, 10),
                       (9, 10, 10), (10, 9, 9), (11, 9, 10), (12, 9, 10), (13, 9, 10), (14, 9, 10), (15, 9, 10),
                       (16, 9, 10)
                   )
}

pairs_elbows = {'fsd':
                   (
                       (0, 7, 8), (1, 7, 8), (2, 7, 8), (3, 7, 8), (4, 7, 8), (5, 7, 8), (6, 7, 8), (7, 8, 8),
                       (8, 7, 7), (9, 7, 8), (10, 7, 8),
                       (11, 7, 8), (12, 7, 8), (13, 7, 8), (14, 7, 8), (15, 7, 8), (16, 7, 8)
                   )
}

pairs_knees = {'fsd':
                   (
                       (0, 13, 14), (1, 13, 14), (2, 13, 14), (3, 13, 14), (4, 13, 14), (5, 13, 14), (6, 13, 14),
                       (7, 13, 14), (8, 13, 14), (9, 13, 14),
                       (10, 13, 14), (11, 13, 14), (12, 13, 14), (13, 14, 14), (14, 13, 13), (15, 13, 14), (16, 13, 14)
                   )
}

def get_single_angle(vector, data):
    # data 现在是形状 (C, T, V, M)
    data = data[0:3, :, :, :]  # 保持原来的切片方式
    C, T, V, M = data.shape  # 更新维度定义
    info = vector['fsd']
    fp_sp = np.zeros((1, T, V, M))  # 去掉 N 维度
    for idx, (target, neigh1, neigh2) in enumerate(info):
        vec1 = data[:, :, neigh1, :] - data[:, :, target, :]
        vec2 = data[:, :, neigh2, :] - data[:, :, target, :]
        inner_product = (vec1 * vec2).sum(0)
        mod1 = np.clip(np.linalg.norm(vec1, ord=2, axis=0), 1e-6, np.inf)
        mod2 = np.clip(np.linalg.norm(vec2, ord=2, axis=0), 1e-6, np.inf)
        mod = mod1 * mod2
        theta = 1 - inner_product / mod
        theta = np.clip(theta, 0, 2)
        fp_sp[:, :, idx, :] = theta  # 根据新的维度更新
    return fp_sp


def get_JA(J):
    C, T, V, M = J.shape
    l = [pairs_local, pairs_center1, pairs_center2, pairs_hands, pairs_elbows, pairs_knees]
    res = np.zeros((len(l), T, V, M), dtype='float32')
    for i, pairs in enumerate(l):
        ans = get_single_angle(pairs, J)
        res[i, :, :, :] = ans
    JA = np.concatenate((J, res), axis=0).astype('float32')

    return JA