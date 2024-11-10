#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from torchlight.util import DictAction
from timm.scheduler.cosine_lr import CosineLRScheduler
# 由于在Windows下，注释掉
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)  # 设置所有 CUDA 设备的随机种子，以确保 CUDA 操作的可重复性
    torch.manual_seed(seed)  # 设置 CPU 的随机种子，以确保 PyTorch 操作的可重复性
    np.random.seed(seed)  # 设置 NumPy 的随机种子，以确保 NumPy 操作的可重复性
    random.seed(seed)     # 设置 Python 内置的随机模块的随机种子，以确保 Python 随机操作的可重复性
    # torch.backends.cudnn.enabled = False  # 取消此行的注释可以禁用 cuDNN 后端（可能影响性能，但有助于调试）
    torch.backends.cudnn.deterministic = True  # 确保 cuDNN 后端使用确定性算法，以提高可重复性
    torch.backends.cudnn.benchmark = False # 禁用 cuDNN 的自动优化功能，以避免不同硬件上性能优化差异导致的不可重复性

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')   # 从传入的字符串中分离出模块名和类名
    __import__(mod_str)  # 使用 __import__ 函数导入模块
    try:  #捕捉并处理异常（错误）
        return getattr(sys.modules[mod_str], class_str)  # 从导入的模块中获取指定的类
    except AttributeError:  # except 块来捕捉并处理这些异常
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))
        # 如果在模块中找不到指定的类，抛出 ImportError 异常，并输出详细的错误信息
def str2bool(v):   # 定义一个函数 str2bool，将字符串参数 v 转换为布尔值（True 或 False）
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True # 如果 v（转换为小写）是 'yes'、'true'、't'、'y' 或 '1'，则返回 True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False  # 如果 v（转换为小写）是 'no'、'false'、'f'、'n' 或 '0'，则返回 False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        # 如果 v 不符合上述任何条件，抛出一个 argparse.ArgumentTypeError 异常，表示遇到了不支持的值

def get_parser():  #创建一个命令行参数解析器
    # 参数优先级：命令行参数 > 配置文件参数 > 默认值 parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')  # 解析器的描述信息
    parser.add_argument(
        '--work-dir',
        default='work_dir_test',
        help='the work folder for storing results') # 工作目录，用于存储结果

    parser.add_argument('-model_saved_name', default='')  # 模型保存的名称
    parser.add_argument(
        '--config',
        default='work_dir/ntu/csub/ctrgcn/config.yaml',
        help='path to the configuration file')  # 配置文件的路径

    # processor 处理器
    parser.add_argument(
        '--phase', default='train', help='must be train or test') # 训练或测试阶段的选择
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')# 是否保存分类分数

    # visulize and debug
    parser.add_argument(  # 随机种子，用于PyTorch
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument( # 打印日志的间隔（每多少次迭代）
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(  # 存储模型的间隔（每多少次迭代）
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument( # 从哪个epoch开始保存模型  在每个 epoch 中，模型的权重会根据训练数据进行更新
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(  # 评估模型的间隔（每多少次迭代）
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument( # 是否打印日志
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(  # 显示的Top K准确度 Top-K 准确度是一个有用的指标，尤其是在多分类任务中，帮助评估模型在给定数量的预测中是否能够正确地包含真实类别。
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder 从存储中提取数据，并将其准备好以供模型训练或评估使用
    parser.add_argument(  # 数据加载器的选择
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(  # 数据加载器的工作线程数量
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument( # 训练数据加载器的参数
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument( # 测试数据加载器的参数
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')  # 使用的模型
    parser.add_argument( # 模型的参数
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument( # 网络初始化的权重
        '--weights',
        default= None ,
        help='the weights for network initialization')
    parser.add_argument( # 初始化时将忽略的权重名称
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optimoptim 是 PyTorch 中用于优化模型的模块。它提供了多种优化算法，如随机梯度下降（SGD）、Adam、RMSprop 等。optim 的主要功能是更新模型的参数，以最小化损失函数。
    parser.add_argument(   # 初始学习率
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument( # 优化器减少学习率的epoch
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument( # 用于训练或测试的GPU索引
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')  # 优化器类型
    parser.add_argument( # 是否使用Nesterov动量
        '--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument(  # 训练批次大小
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument( # 测试批次大小
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument( # 从哪个epoch开始训练
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(  # 训练到哪个epoch停止
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument( # 优化器的权重衰减
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument( # 学习率衰减率
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0) # 预热的epoch数

    return parser # 返回创建的解析器


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition  用于骨架基础动作识别的处理器
    """

    def __init__(self, arg):  # 初始化函数，接收一个参数 `arg`
        self.arg = arg  # 将参数 `arg` 存储为实例变量
        self.save_arg()  # 调用 `save_arg` 方法保存参数
        if arg.phase == 'train': # 如果运行模式是训练模式
            if not arg.train_feeder_args['debug']: # 如果 `debug` 模式未启用
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs') # 设置模型保存目录
                if os.path.isdir(arg.model_saved_name): # 检查保存目录是否已存在
                    print('log_dir: ', arg.model_saved_name, 'already exist')# 输出目录已存在的信息
                    answer = input('delete it? y/n:')# 询问是否删除目录
                    if answer == 'y':# 如果用户选择删除
                        shutil.rmtree(arg.model_saved_name) # 删除目录及其内容
                        print('Dir removed: ', arg.model_saved_name) # 输出目录已删除的信息
                        input('Refresh the website of tensorboard by pressing any keys')# 提示用户刷新 TensorBoard 网站
                    else:# 如果用户选择不删除
                        print('Dir not removed: ', arg.model_saved_name)# 输出目录未删除的信息
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')  # 初始化训练和验证日志记录器
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else: # 如果 `debug` 模式启用，初始化测试日志记录器
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0 # 初始化全局步骤计数器
        # pdb.set_trace()
        self.load_model()# 加载模型

        if self.arg.phase == 'model_size':# 如果运行模式是模型大小检测
            pass# 不做任何操作
        elif self.arg.phase == 'test':
            self.load_data()
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['test']))
        else:# 否则，加载优化器和数据
            self.load_data()
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr # 设置学习率
        self.best_acc = 0  # 初始化最佳准确率和最佳准确率所在的训练轮次
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device) # 将模型移动到指定的计算设备（例如 GPU）

        if type(self.arg.device) is list: # 如果 `arg.device` 是一个列表
            if len(self.arg.device) > 1: # 并且如果设备列表中有多个设备
                self.model = nn.DataParallel( # 使用数据并行处理模型
                    self.model,
                    device_ids=self.arg.device, # 指定设备列表
                    output_device=self.output_device) # 指定输出设备

    def load_data(self):  # 字典将包含用于训练和测试的数据加载器，配置好所有必要的参数。
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):  #确保模型的正确初始化，设置了损失函数，并处理了权重的加载与错误处理。
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device #如果 self.arg.device 是列表，则取第一个设备，否则直接使用 self.arg.device
        Model = import_class(self.arg.model) #使用 import_class(self.arg.model) 动态导入模型类
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir) # 复制模型的源文件到指定的工作目录
        print(Model)
        self.model = Model(**self.arg.model_args) # 使用指定的参数初始化模型
        print(self.model)
        self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

        if self.arg.weights:  # 如果提供了权重文件路径
            self.global_step = int(arg.weights[:-3].split('-')[-1]) # 从权重文件路径中提取全局步数
            self.print_log('Load weights from {}.'.format(self.arg.weights)) # 打印加载权重的日志
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            # 根据权重文件的扩展名加载权重
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            # 遍历要忽略的权重，并从加载的权重中移除
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            # 尝试加载模型状态字典
            try:
                self.model.load_state_dict(weights)
            except: # 如果加载失败，获取当前模型状态字典，并打印缺失的权重
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)  # 更新状态字典并重新加载
                self.model.load_state_dict(state)

    def load_optimizer(self):  # 根据self.arg.optimizer的值选择优化器类型
        if self.arg.optimizer == 'SGD':  # 如果选择SGD优化器，配置其参数
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam': # 如果选择Adam优化器，配置其参数
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()  # 如果提供的优化器类型不被支持，抛出ValueError异常
        # 打印日志，显示是否使用了warm-up以及warm-up的epoch数量
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = None
        if 1:
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps),
                lr_min=1e-5,
                warmup_lr_init=0.01,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=True,
            )
        else:
            raise ValueError()

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)  # 将self.arg转换为字典格式
        if not os.path.exists(self.arg.work_dir): # 如果工作目录不存在，则创建它
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f: # 打开config.yaml文件进行写操作
            f.write(f"# command line: {' '.join(sys.argv)}\n\n") # 写入命令行参数作为注释
            yaml.dump(arg_dict, f) # 将arg_dict内容以YAML格式写入文件

    # def adjust_learning_rate(self, epoch): #根据训练的epoch和优化器类型调整学习率
    #     if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam': # 检查优化器类型是否为SGD或Adam
    #         if epoch < self.arg.warm_up_epoch:
    #             lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
    #         else:
    #             lr = self.arg.base_lr * (
    #                     self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #         return lr
    #     else:
    #         raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):  # 检查是否需要在日志信息前加上时间戳
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False): #训练神经网络模型的 train
        self.model.train()  # 设置模型为训练模式
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    #实现了模型的评估过程。它遍历指定的测试数据加载器，计算损失和准确率，并根据需要记录错误预测和结果
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40) # 使用 tqdm 进度条显示数据加载过程
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None: # 如果需要记录预测结果或错误信息
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:  # 如果需要保存分数，将分数保存到 pkl 文件中
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:  # 计算每类的准确率
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f) # 将每类准确率和混淆矩阵保存到 csv 文件中
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):  # 处理了模型的训练和测试阶段
        if self.arg.phase == 'train': # 判断当前阶段是训练还是测试
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model  测试最佳模型
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0] # 查找保存的最佳模型的路径
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser() # 获取命令行参数解析器实例

    # load arg form config file
    p = parser.parse_args() # 从命令行解析参数
    if p.config is not None: # 如果指定了配置文件
        with open(p.config, 'r') as f:
            #default_arg = yaml.load(f)
            default_arg = yaml.safe_load(f.read())
        key = vars(p).keys()
        for k in default_arg.keys():  # 遍历配置文件中的每个键
            if k not in key: # 如果配置文件中的键不在命令行参数中，则输出错误信息
                print('WRONG ARG: {}'.format(k))
                assert (k in key) # 抛出断言异常，确保所有配置文件中的键都在命令行参数中
        parser.set_defaults(**default_arg)

    arg = parser.parse_args() # 重新解析参数，以包括从配置文件中加载的默认值
    init_seed(arg.seed)# 初始化随机种子，以确保结果的可重复性
    processor = Processor(arg)# 创建一个 Processor 实例并传入命令行参数
    processor.start() # 启动 Processor 实例的处理流程
