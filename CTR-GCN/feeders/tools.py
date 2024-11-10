import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    # print(window)
    # print(data_numpy.shape)
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),window), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

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

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    if valid_frame.sum() > 0:
        begin = np.where(valid_frame)[0][0]
        end = len(valid_frame) - valid_frame[::-1].argmax()
    else:
        begin = 0
        end = 299
    #begin = valid_frame.argmax()


    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

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