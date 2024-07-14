import logging
import numpy as np
import os
import pickle

import pandas as pd
import scipy.sparse as sp
import sys
import torch

from scipy.sparse import linalg
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from numpy.random import Generator, PCG64

from torch.autograd import Variable
from torch.nn import init
from torch import nn

# from sklearn.preprocessing import MinMaxScaler


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    '''
    wrapper function for endless data loader.
    '''
    for loader in repeat(data_loader):
        yield from loader


class DataLoader(object):
    def __init__(self, xs, ys,  batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class StandardScaler:
    """
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


def scaled_laplacian(W):
    """
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = linalg.eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def get_identity_mat(num_nodes):
    eye = np.identity(num_nodes)
    return sp.coo_matrix(eye)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_windows(df, sample_size, t_shuffle, time_input_number,
                 time_forward_pred, station_id_2_node_id_map, t_start, t_end):
    stride = None
    # Not mess up the order:

    # sample_size = None
    if sample_size is None and t_shuffle is False:
        t_val_iter = range(t_start, t_end, 24)
        order_id_2_time_id_map = dict((n, tid) for n, tid in enumerate(t_val_iter))
        time_id_2_order_id_map = {v: k for k, v in order_id_2_time_id_map.items()}
        np.save('Data/time_id_2_order_id_map.npy', time_id_2_order_id_map)

    # Mess up the order:
    elif (not sample_size is None) and t_shuffle:
        seed_rg = Generator(PCG64())
        t_val_iter = seed_rg.choice(range(t_start, t_end), size=sample_size, replace=False)
        # save time_id_2_order_id_map
        order_id_2_time_id_map = dict((n, tid) for n, tid in enumerate(t_val_iter))
        time_id_2_order_id_map = {v: k for k, v in order_id_2_time_id_map.items()}
        np.save('Data/time_id_2_order_id_map.npy', time_id_2_order_id_map)

    null_list = [0] * time_input_number
    # ext_null_list = [0] * (ext_df.shape[1] - 4)
    for t_val in t_val_iter:
        df_x = df.loc[(df['time_id'] < t_val + time_input_number) & (df['time_id'] >= t_val)]
        # df_y = df.loc[df['time_id'] == t_val + time_input_number + time_forward_pred - 1]
        df_y = df.loc[(df['time_id'] < t_val + time_input_number + time_forward_pred) & (df['time_id'] >= t_val + time_input_number)]

        # ext_x = ext_df.loc[
        #     (ext_df['interval_id'] < t_val + time_input_number) & (ext_df['interval_id'] >= t_val)]
        # ext_y = ext_df.loc[(ext_df['interval_id'] < t_val + time_input_number + time_forward_pred) & (
        #             ext_df['interval_id'] >= t_val + time_input_number)]

        if len(df_x) == 0 or len(df_y) == 0:
            raise RuntimeError('Encountered missing time period')
        # Move from pandas to torch tensor compatible data

        data_xt = [[null_list, null_list]] * len(station_id_2_node_id_map)
        # ext_xt = [[ext_null_list, ext_null_list, ext_null_list, ext_null_list, ext_null_list, ext_null_list,
        #            ext_null_list, ext_null_list, ext_null_list, ext_null_list, ext_null_list, ext_null_list]] * len(
        #     station_id_2_node_id_map)
        # ext_xt = [[ext_null_list, ext_null_list]] * len(station_id_2_node_id_map)

        for station_id, chunk in df_x.groupby('station_id'):
            ind = station_id
            # 时间序列数据
            # data_xt[ind] = chunk['bikes_demand'].tolist()
            data_xt[ind] = chunk[['bikes_demand', 'time_id']].values.tolist()  # TODO: 附上时间id信息

            # ext_xt[ind] = ext_x[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'weekend', 'holiday',
            #                      'NormalizedCases']].values.tolist()

        data_yt = [[0, 0]] * len(station_id_2_node_id_map)
        # ext_yt = [[0, 0]] * len(station_id_2_node_id_map)
        for station_id, chunk in df_y.groupby('station_id'):
            ind = station_id
            # data_yt[ind] = chunk['bikes_demand'].tolist()
            data_yt[ind] = chunk[['bikes_demand', 'time_id']].values.tolist()

            # ext_yt[ind] = ext_y[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'weekend', 'holiday',
            #                      'NormalizedCases']].values.tolist()

        # yield data_xt, data_yt, ext_xt, ext_yt
        #  TODO: 解决缺站点76
        data_xt[76] = data_xt[75]
        data_yt[76] =  data_yt[75]
        yield data_xt, data_yt



def load_dataset(config, dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}

    try:
        # data_x = np.load('Data/data_x_1.npy').tolist()  # generated by def time_windows with (t_shuffle=False) & (sample_size=None)
        data_x= np.load('Data/data_x.npy', allow_pickle=True).tolist()  # generated by def time_windows with (t_shuffle=True) & (sample_size=16055S)

    except FileNotFoundError:

        df = pd.read_csv(os.path.join(dataset_dir + 'TS_Demand_240.csv'))  # TODO：读取TS_Demand
        # df = pd.read_csv(os.path.join(dataset_dir + 'TS_Demand_240_busy.csv'))  #TODO: 【TRY改动】
        df = df.drop(columns='Unnamed: 0')
        seq_len = config['arch']['args']['seq_len']
        slice_generator = time_windows(df,
                                       sample_size=config['dataset']['sample_size'],
                                       # t_shuffle=False,
                                       t_shuffle=True,
                                       time_input_number=seq_len,
                                       time_forward_pred=seq_len,
                                       station_id_2_node_id_map=np.load('DataPreprocessing/station_id_2_node_id_map.npy',
                                                                        allow_pickle=True).item(),
                                       # station_id_2_node_id_map=np.load('DataPreprocessing/station_id_2_node_id_map_busy.npy',
                                       #                                  allow_pickle=True).item(),  # TODO: 【TRY.py改动】
                                       t_start=df['time_id'].min(),
                                       t_end=(df['time_id'].max() - seq_len - seq_len))
       # #TODO: 生成不打乱时间顺序的数据(1)
       #  slice_generator = time_windows(df,
       #                                 sample_size=None,
       #                                 t_shuffle=False,
       #                                 # t_shuffle=True,
       #                                 time_input_number=seq_len,
       #                                 time_forward_pred=seq_len,
       #                                 station_id_2_node_id_map=np.load('DataPreprocessing/station_id_2_node_id_map.npy',
       #                                                                  allow_pickle=True).item(),
       #                                 t_start=df['time_id'].min(),
       #                                 t_end=(df['time_id'].max() - seq_len - seq_len))
        data_x, data_y = [], []
        # for count, (data_g_xt, data_g_yt, ext_g_xt, ext_g_yt) in enumerate(slice_generator):
        for count, (data_g_xt, data_g_yt) in enumerate(slice_generator):
            data_x.append(data_g_xt)
            data_y.append(data_g_yt)
            # ext_x.append(ext_g_xt)
            # ext_y.append(ext_g_yt)
        np.save('Data/data_x.npy', data_x)
        np.save('Data/data_y.npy', data_y)
        # np.save('Data/ext_x.npy', ext_x)
        # np.save('Data/ext_y.npy', ext_y)

    else:
        # data_y = np.load('Data/data_y_1.npy').tolist()  # generated by def time_windows with (t_shuffle=False) & (sample_size=None)
        data_y = np.load('Data/data_x.npy', allow_pickle=True).tolist()
        # ext_x = np.load('Data/ext_x.npy').tolist()
        # ext_y = np.load('Data/ext_y.npy').tolist()

        # # TODO: 减少记录数据 16055-->6055
        # data_x = data_x[:6055]
        # data_y = data_y[:6055]

    # Divide to train, val and test

    # #TODO:生成不打乱顺序的全部test数据(2)
    # train_n = int((len(data_x) * 0))
    # val_n = int((len(data_x) * 0))
    # data['x_train'] = np.array(data_x[0:train_n])
    # data['y_train'] = np.array(data_y[0:train_n])
    # data['x_val'] = np.array(data_x[train_n:val_n])
    # data['y_val'] = np.array(data_y[train_n:val_n])
    # data['x_test'] = np.array(data_x[val_n:len(data_x)])
    # data['y_test'] = np.array(data_y[val_n:len(data_y)])
    # for category in ['x_test', 'y_test']:
    #     data[category] = data[category].transpose(0, 2, 1, 3)
    #     data[category] = data[category].astype(np.float64)
    # scaler = MinMaxScaler(min=data['x_test'][...,0].min(), max=data['x_test'][...,0].max())
    # data['x_test' ][..., 0] = scaler.transform(data['x_test'][..., 0])
    # data['test_loader'] = DataLoader(data['x_test'], data['y_test'],  test_batch_size, shuffle=False)
    # data['scaler'] = scaler
    # return data

    #TODO：正常流程
    train_n = int((len(data_x)*0.6))
    # train_n = int((len(data_x)*0))
    val_n = int((len(data_x)*0.8))
    # val_n = int((len(data_x)*0))
    data['x_train'] = np.array(data_x[0:train_n])
    data['y_train'] = np.array(data_y[0:train_n])
    data['x_val'] = np.array(data_x[train_n:val_n])
    data['y_val'] = np.array(data_y[train_n:val_n])
    data['x_test'] = np.array(data_x[val_n:len(data_x)])
    data['y_test'] = np.array(data_y[val_n:len(data_y)])

    for category in ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']:
        data[category] = data[category].transpose(0, 2, 1, 3)
        data[category] = data[category].astype(np.float64)

    #Todo: 加入ext数据
    time_ids = data['x_train'][..., 1]

    #TODO: 标准化需求数据 （稀疏）
    scaler = MinMaxScaler(min=data['x_train'][...,0].min(), max=data['x_train'][...,0].max())

    # 标准化需求数据
    # scaler = StandardScaler(mean=data['x_train'][...,0].mean(), std=data['x_train'][...,0].std())
    # scaler2 = StandardScaler(mean=data['x_train'][...,1].mean(), std=data['x_train'][...,1].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,0] = scaler.transform(data['x_' + category][...,0])
        # data['x_' + category][...,1] = scaler2.transform(data['x_' + category][...,1])
        if category == "test":
            continue
        data['y_' + category][...,0] = scaler.transform(data['y_' + category][...,0])
        # data['y_' + category][...,1] = scaler2.transform(data['y_' + category][...,1])
    # TODO: DataLoader里洗牌
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    # data['train_loader'] = DataLoader(data['x_train'], data['y_train'], data['ext_x_train'], data['ext_y_train'] , batch_size, shuffle=True)
    # data['ext_train_loader'] = DataLoader(data['ext_x_train'], data['ext_y_train'], batch_size, shuffle=False)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    # data['ext_val_loader'] = DataLoader(data['ext_x_val'], data['ext_y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'],  test_batch_size, shuffle=False)
    # data['ext_test_loader'] = DataLoader(data['ext_x_test'], data['ext_y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler
    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def build_sparse_matrix(sp_mat):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = sp_mat.shape
    i = torch.LongTensor(np.vstack((sp_mat.row, sp_mat.col)).astype(int))
    v = torch.FloatTensor(sp_mat.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def z_score(x, mean, std):
    """
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    """
    return (x - mean) / std


def Linear(args, output_size, bias, bias_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias(default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.data.size(2) for a in args]
    for shape in shapes:
        total_arg_size += shape

    # Now the computation.
    weights = nn.Parameter(torch.FloatTensor(total_arg_size, output_size))
    init.xavier_uniform_(weights)
    weights = weights.to('cuda:0')
    # weights = Variable(torch.zeros(total_arg_size, output_size))
    if len(args) == 1:
        res = torch.matmul(args[0], weights)
    else:
        res = torch.matmul(torch.cat(args, 2), weights)
    if not bias:
        return res

    if bias_initializer is None:
        biases = Variable(torch.zeros(output_size))
        biases = biases.to('cuda:0')

    return torch.add(res, biases)
