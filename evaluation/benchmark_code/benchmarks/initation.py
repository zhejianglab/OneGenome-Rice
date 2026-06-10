#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
"""

import os
import torch
import yaml
from copy import deepcopy
import random
import json
torch.use_deterministic_algorithms(True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For numpy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """配置管理类 - 只读版本"""

    def __init__(self, args):
        """
        初始化配置

        Args:
            config_path (str): 配置文件路径，支持.yaml和.yml格式
        """

        self.args = args
        self.config_path = self.args.config
        self._config = self._load_config()
        self._datasets_info = self._load_dataset_feature()
        self._model_config = self._load_model_config()
        self._process_config()
        set_seed(self._config['seed'])

    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # 根据文件扩展名决定加载方式
                if self.config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    return json.load(f)
                else:
                    # 默认尝试YAML格式
                    try:
                        return yaml.safe_load(f)
                    except yaml.YAMLError:
                        raise ValueError(
                            f"无法解析配置文件 {self.config_path}，请确保使用有效的YAML或JSON格式")
        else:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")

    def _load_dataset_feature(self):
        """加载数据集特征配置文件"""
        # 从主配置中获取数据集特征配置文件路径
        dataset_feature_path = self._config.get(
            'datasets_feature_path', 'benchmarks/datasets_info.yaml')

        if os.path.exists(dataset_feature_path):
            try:
                with open(dataset_feature_path, 'r', encoding='utf-8') as f:
                    dataset_feature = yaml.safe_load(f)
                    print(f"✓ 成功加载数据集特征配置: {dataset_feature_path}")
                    return dataset_feature
            except yaml.YAMLError as e:
                raise ValueError(
                    f"⚠️ 警告: 无法解析数据集特征配置文件 {dataset_feature_path}: {e}")
        else:
            raise FileNotFoundError(
                f"⚠️ 警告: 数据集特征配置文件 {dataset_feature_path} 不存在")

    def _load_model_config(self):
        """加载模型配置文件"""
        # 从主配置中获取模型配置文件路径
        model_path = self._config.get('model_path', None)
        model_config_path = f"{model_path}/config.json" if model_path is not None else None
        if model_config_path is not None and os.path.exists(model_config_path):
            with open(model_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            assert "num_layers" in self._config, "模型配置文件不存在，且num_layers参数未提供"
            return {
                'num_hidden_layers': self._config['num_layers']
            }

    def _process_config(self):
        """处理配置参数"""
        # 验证必需参数
        if 'model_path' not in self._config or not self._config['model_path']:
            assert 'model_name' in self._config, "config中不存在模型路径，且model_name参数未提供"
        else:
            # 处理模型路径，去除末尾的斜杠，获取模型名称
            model_path = self._config['model_path'].rstrip('/')
            self._config['model_name'] = model_path.split("/")[-1]

        # 处理GPU列表
        if 'gpu_list' in self._config and self._config['gpu_list']:
            gpu_list = self._config['gpu_list']
            # 如果gpu_list是字符串，则按逗号分割
            if isinstance(gpu_list, str):
                gpu_list = gpu_list.split(",")
            # 确保是列表格式
            if isinstance(gpu_list, list):
                self._config['gpu_list'] = [
                    int(str(gpu).strip()) for gpu in gpu_list]
            else:
                raise ValueError("gpu_list必须是字符串或列表格式")
        else:
            device_count = torch.cuda.device_count()
            self._config['gpu_list'] = [i for i in range(device_count)]

        if 'layer_to_eval' not in self._config:
            self._config['layer_to_eval'] = list(
                range(self._model_config['num_hidden_layers']+1))
        else:
            self._config['layer_to_eval'] = [
                int(layer) for layer in self._config['layer_to_eval']]

        if 'batch_size' not in self._config:
            self._config['batch_size'] = None

        if 'eval_datasets' not in self._config:
            self._config['eval_datasets'] = self._datasets_info['default_dataset']
        if 'wandb_online' not in self._config or self._config['wandb_online']:
            pass
        else:
            os.environ["WANDB_MODE"] = "offline"

        # 打印配置信息
        self._print_config()

    def _print_config(self):
        """打印配置信息"""
        print("=== 配置参数 ===")
        print(f"配置文件: {self.config_path}")
        for key, value in self._config.items():
            print(f"{key}: {value}")
        print("================")

    @property
    def all_datasets_feature(self):
        """只读的数据集特征配置"""
        return deepcopy(self._datasets_info)

    @property
    def dataset_info(self):
        """只读的评估数据集"""
        return deepcopy(self._datasets_info["dataset_feature"])

    @property
    def model_config(self):
        """只读的模型配置"""
        return deepcopy(self._model_config)

    def get(self, key, default=None):
        """
        获取配置值

        Args:
            key (str): 配置键
            default: 默认值

        Returns:
            配置值或默认值
        """
        value = self._config.get(key, default)
        # 返回深拷贝，防止外部修改
        if isinstance(value, (list, dict)):
            return deepcopy(value)
        return value

    def __getitem__(self, key):
        """支持字典式访问 - 返回只读副本"""
        value = self._config[key]
        # 返回深拷贝，防止外部修改
        if isinstance(value, (list, dict)):
            return deepcopy(value)
        return value

    def __contains__(self, key):
        """支持 in 操作符"""
        return key in self._config

    def __iter__(self):
        """支持迭代 - 返回键的副本"""
        return iter(list(self._config.keys()))

    def keys(self):
        """获取所有配置键 - 返回副本"""
        return list(self._config.keys())

    def values(self):
        """获取所有配置值 - 返回副本"""
        return [deepcopy(v) if isinstance(v, (list, dict)) else v for v in self._config.values()]

    def items(self):
        """获取所有配置项 - 返回副本"""
        return [(k, deepcopy(v) if isinstance(v, (list, dict)) else v) for k, v in self._config.items()]

    def copy(self):
        """复制配置 - 返回深拷贝"""
        return deepcopy(self._config)

    def to_dict(self):
        """转换为字典 - 返回深拷贝"""
        return deepcopy(self._config)

    def __len__(self):
        """获取配置项数量"""
        return len(self._config)

    def __repr__(self):
        """字符串表示"""
        return f"Config({self.config_path})"

    def __str__(self):
        """字符串表示"""
        return f"Config({self.config_path})"

    # 禁用修改操作
    def __setitem__(self, key, value):
        """禁止设置配置项"""
        raise TypeError("Config对象是只读的，不能修改配置项")

    def __delitem__(self, key):
        """禁止删除配置项"""
        raise TypeError("Config对象是只读的，不能删除配置项")

    def update(self, other):
        """禁止更新配置"""
        raise TypeError("Config对象是只读的，不能更新配置")

    def clear(self):
        """禁止清空配置"""
        raise TypeError("Config对象是只读的，不能清空配置")

    def pop(self, key, default=None):
        """禁止弹出配置项"""
        raise TypeError("Config对象是只读的，不能弹出配置项")

    def popitem(self):
        """禁止弹出配置项"""
        raise TypeError("Config对象是只读的，不能弹出配置项")

    def setdefault(self, key, default=None):
        """禁止设置默认值"""
        raise TypeError("Config对象是只读的，不能设置默认值")
