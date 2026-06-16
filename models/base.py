import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from collections import OrderedDict

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._init_cls = args['init_cls']
        self.task_count = 0
        
        if isinstance(args['increment'], list):
            self._increments = args['increment']
            self._incrment_cls = None
        else:  # 单个数据集
            self._increments = None
            self._incrment_cls = args['increment']

        self._incrment_cls = args['increment']


    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        # Memory parameters have been removed, returning default value
        return 0

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def save_checkpoint(self, filename, head_only=False, learnable_only=False):
        if hasattr(self._network, 'module'):
            to_save = self._network.module
        else:
            to_save = self._network

        if head_only:
            to_save_dict = to_save.fc.state_dict()
        else:
            to_save_dict = to_save.state_dict()
            
        if learnable_only:
            new_dict = OrderedDict()
            filtered_keys = [n for n, p in to_save.named_parameters() if p.requires_grad]
            for k in filtered_keys:
                new_dict[k] = to_save_dict[k]
            to_save_dict = new_dict

        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': to_save_dict}
        
        torch.save(save_dict, f'{filename}_{self._cur_task}.pth')

    def after_task(self):
        # increment task counter
        self.task_count += 1

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(correct.numpy()*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        return np.concatenate(y_pred), np.concatenate(y_true)