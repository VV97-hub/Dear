"""1.改回了一轮p一轮q的计算和通信，发现同时计算和通信pq没有意义，那和powersgd是不同的，powersgd一轮需要通信两次。
2.加入敏感组和非敏感组分割的策略（但由于一些代码改动导致通信加长了很多）"""
# Copyright 2020 HKBU. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import collections
import numpy as np

from comm_core import rank, size, Communicator, init as comm_init
from tensorfusion import CommReduceScatter, CollectiveOp
import utils
from compression import compressors


comm_init()
comm = None
all_gather_comm = None
reduce_scatter_comm = None

# Please set THRESHOLD=None and NUM_NEARBY_LAYERS=1 to disable tensor fusion for notf experiments. 
# 上面的参数设置是禁用 notf 实验中的张量融合功能。
NUM_NEARBY_LAYERS = 4 # default: 4
THRESHOLD = 25 # default: 25MB
NSTREAMS = 1
def init():
    global comm
    global all_gather_comm 
    global reduce_scatter_comm 
    comm = Communicator(NSTREAMS)
    reduce_scatter_comm = CommReduceScatter(op=CollectiveOp.REDUCE_SCATTER)
    all_gather_comm = CommReduceScatter(op=CollectiveOp.ALL_GATHER)


DEBUG = False

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, 
            num_nearby_layers=NUM_NEARBY_LAYERS, 
            threshold=THRESHOLD, 
            exclude_parts='',
            compression=None):  # 添加 compression 参数（压缩新增）
        r"""Distributed optimizer with overlapping reduceScatter and allGather and tensor fusion.

        Args:
            params: optimizer parameters.
            model: training model.
            num_nearby_layers: number of neaby layers merged for tensor fusion.
        """
        super(self.__class__, self).__init__(params)
        self._model = model
        self._threshold = threshold
        self._num_nearby_layers = num_nearby_layers
        self._num_steps = 0
        self._grad_accs = []
        self._compression = compression  # 保存为实例属性（压缩新增）
        self.param_to_group = {}
        for group in self.param_groups:
            for p in group['params']:
                self.param_to_group[p] = group
        
        # 控制是否跳过某些 collective
        self.exclude_reducescatter = True if exclude_parts.find('reducescatter') >=0 else False
        self.exclude_allgather = True if exclude_parts.find('allgather') >=0 else False

        # parameter names 建立 参数 → 名字 映射字典
        named_parameters = list(model.named_parameters())
        # 按内存地址去重，tied weights 只保留第一个名字
        seen_ptrs = set()
        deduped = []
        for k, v in named_parameters:
            if v.data_ptr() not in seen_ptrs:
                seen_ptrs.add(v.data_ptr())
                deduped.append((k, v))
        named_parameters = deduped

        if len(named_parameters) > 0:
            self._param_names = {v: k for k, v in sorted(named_parameters)}
        else:
            self._param_names = {v: 'param.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        if size() > 1:
            self._register_hooks() # 注册钩子backward hooks

            if self._threshold is not None: # 分组的方法
                self._generate_groups_with_threshold() # 按照通信量大小分组
            else:
                self._generate_groups_with_nearby_layers() # 按照组顺序分组

    def _generate_groups_with_nearby_layers(self):
        """
        Generate groups with nearby layers for tensor fusion.
        """
        module_groups = []
        current_group = []
        for i, module in enumerate(self._register_modules):
            current_group.append(module)
            if not self._num_nearby_layers < 0 and (i+1) % self._num_nearby_layers == 0: 
                module_groups.append(current_group)
                current_group = []
        if len(current_group) > 0:
            module_groups.append(current_group)
        self._prepare_tensor_fusion(module_groups)
    
    """ 这是未增加敏感层边界分组的原版
    def _generate_groups_with_threshold(self):
    """
        # Generate groups with buffer size threshold (in MB) for tensor fusion. 
    """
        module_sizes = {}
        model_total_size = 0
        for module in self._register_modules:
            module_name = self._module_names[module]
            tot_size = 0
            for p in self._module_direct_parameters[module_name]:
                tot_size += p.data.numel()
                model_total_size += p.data.numel()
            module_sizes[module_name] = tot_size*4/1024/1024
        if rank() == 0:
            print('# of parameters: ', model_total_size)

        module_groups = []
        current_group = []
        tot_size = 0
        for module in self._register_modules: # forward order
            mod_size = module_sizes.get(self._module_names[module])
            if tot_size == 0 or tot_size + mod_size < self._threshold:
                current_group.append(module)
                tot_size += mod_size
            else:
                module_groups.append(current_group)
                current_group = [module]
                tot_size = mod_size
        if len(current_group) > 0:
            module_groups.append(current_group)
        self._prepare_tensor_fusion(module_groups)
    """
    
    def _generate_groups_with_threshold(self):
        """
        Generate groups with buffer size threshold (in MB) for tensor fusion.
        在原有25MB阈值基础上，增加敏感性边界切割：
        同一组内所有module必须同属敏感或非敏感，保证每组只用一套buffer。
        """
        module_sizes = {}
        model_total_size = 0
        for module in self._register_modules:
            module_name = self._module_names[module]
            tot_size = 0
            for p in self._module_direct_parameters[module_name]:
                tot_size += p.data.numel()
                model_total_size += p.data.numel()
            module_sizes[module_name] = tot_size * 4 / 1024 / 1024
        if rank() == 0:
            print('# of parameters: ', model_total_size)

        def _module_is_sensitive(module):
            """
            判断一个module是否属于敏感组。
            只要module内有任意一个参数命中敏感关键词，整个module归为敏感组。
            无压缩时所有module视为非敏感（走原始路径即可）。
            """
            if not self._compression:
                return False
            module_name = self._module_names[module]
            for p in self._module_direct_parameters[module_name]:
                name = self._param_names[p]
                if self._compression.is_sensitive_layer(name):
                    return True
            return False

        module_groups = []
        current_group = []
        tot_size = 0
        current_is_sensitive = None  # 当前组的敏感性，None表示尚未开始

        for module in self._register_modules:  # 严格保持前向顺序
            module_name = self._module_names[module]
            mod_size = module_sizes.get(module_name, 0)
            mod_sensitive = _module_is_sensitive(module)

            if not current_group:
                # 新组的第一个module，直接加入
                current_group.append(module)
                tot_size = mod_size
                current_is_sensitive = mod_sensitive
            else:
                size_overflow = (tot_size + mod_size >= self._threshold)
                sensitivity_change = (mod_sensitive != current_is_sensitive)

                if size_overflow or sensitivity_change:
                    # 封存当前组，开启新组
                    module_groups.append(current_group)
                    current_group = [module]
                    tot_size = mod_size
                    current_is_sensitive = mod_sensitive
                else:
                    current_group.append(module)
                    tot_size += mod_size

        if current_group:
            module_groups.append(current_group)

        self._prepare_tensor_fusion(module_groups)
        
    """ 这是未增加敏感层边界分组的原版
    def _prepare_tensor_fusion(self, module_groups):
    """
        # Prepare tensor fusion based on module groups, e.g. [[m1, m2], [m3]] in forward order.
    """
        assert module_groups[0][0] == self._register_modules[0], "Module groups are not in forward order."
        self._pad_buffers = []       # group buffers with padding
        self._shard_buffers = []     # sharded group buffers
        self._module_group_idx = {}  # get group idx of module by name
        self._param_group_idx = {}   # get group idx of parameter by name
        
        start_p = 0
        param_groups = []
        for group_idx, module_group in enumerate(module_groups):
            current_param_group = []
            start_p = 0
            for sub_idx, module in enumerate(module_group):
                module_name = self._module_names[module]
                self._module_group_idx[module_name] = (group_idx, sub_idx)

                for p in self._module_direct_parameters[module_name]:
                    param_name = self._param_names[p]
                    numel = p.data.numel()
                    self._param_group_idx[param_name] = (group_idx, len(current_param_group),
                            start_p, start_p+numel)
                    current_param_group.append(param_name)
                    start_p += numel

            param_groups.append(current_param_group)        
            _, pad_tensor, shard_tensor = self._get_pad_tensor(p.data, start_p, size())
            self._pad_buffers.append(pad_tensor)
            self._shard_buffers.append(shard_tensor)

        assert len(module_groups) == len(param_groups)
        self._num_groups = len(module_groups)
        self._module_group_flags = [0]*len(module_groups) # check whether module group is gathered
        self._param_group_flags = [[0]*len(g) for g in param_groups] # check whether param group is ready

        if rank() == 0: 
            print('#Tensor fusion groups:', len(module_groups))
            print('Buffer sizes (MB):', 
                    ', '.join('{:.2f}'.format(buf.numel()*4/1024/1024) for buf in self._pad_buffers))
            print('module groups:', module_groups)
            print('parameter groups:', param_groups)

        # 压缩新增：为每个参数记录压缩后在 pad_buffer 中的位置
        if self._compression :
            self._compressed_param_offsets = {}  # name -> (group_idx, start, end)
            compressed_group_sizes = []
            for group_idx, module_group in enumerate(module_groups):
                offset = 0
                for module in module_group:
                    module_name = self._module_names[module]
                    for p in self._module_direct_parameters[module_name]:
                        name = self._param_names[p]
                        if p.ndimension() <= 1:
                        # 1D tensor 不压缩，大小就是本身
                            compressed_size = p.numel()
                        else:
                            matrix = p.data.view(p.shape[0], -1)
                            n, m = matrix.shape
                            rank_compression = min(n, m, self._compression.rank)
                            # p 和 q 交替传，保证buffer足够大取 max
                            compressed_size = max(n * rank_compression, m * rank_compression)
                            # p和q同时通信的版本
                            # compressed_size = (n + m) * rank_compression
                        self._compressed_param_offsets[name] = (group_idx, offset, offset + compressed_size)
                        offset += compressed_size
                compressed_group_sizes.append(offset)

            # 新建压缩专用的 pad_buffer 和 shard_buffer
            self._compressed_pad_buffers = []
            self._compressed_shard_buffers = []
            for group_idx, total_size in enumerate(compressed_group_sizes):
                pad_num = size() - total_size % size()
                if total_size % size() == 0:
                    pad_num = 0
                self._compressed_pad_buffers.append(
                    torch.empty(total_size + pad_num, device=self._pad_buffers[group_idx].device))
                self._compressed_shard_buffers.append(
                    torch.empty((total_size + pad_num) // size(), device=self._pad_buffers[group_idx].device))
                
            # 在 _prepare_tensor_fusion 末尾加，
            # 同一个 group 内的所有参数，要么全部压缩，要么全部不压缩。可以在 _prepare_tensor_fusion 阶段预判每个 group 的策略
            self._group_use_compression = {}   # {group_idx: bool}
            for group_idx, module_group in enumerate(module_groups):
                has_compressible = False
                for module in module_group:
                    module_name = self._module_names[module]
                    for p in self._module_direct_parameters[module_name]:
                        name = self._param_names[p]
                        if p.ndimension() > 1 and not self._compression.is_sensitive_layer(name):
                            has_compressible = True
                self._group_use_compression[group_idx] = has_compressible
    """
    
    def _prepare_tensor_fusion(self, module_groups):
        """
        Prepare tensor fusion based on module groups, e.g. [[m1, m2], [m3]] in forward order.
        分组已保证同组内同质（全敏感或全非敏感），每组只用一套buffer，只通信一次。
        """
        assert module_groups[0][0] == self._register_modules[0], \
            "Module groups are not in forward order."

        self._pad_buffers = []
        self._shard_buffers = []
        self._module_group_idx = {}
        self._param_group_idx = {}

        param_groups = []
        for group_idx, module_group in enumerate(module_groups):
            current_param_group = []
            start_p = 0
            for sub_idx, module in enumerate(module_group):
                module_name = self._module_names[module]
                self._module_group_idx[module_name] = (group_idx, sub_idx)

                for p in self._module_direct_parameters[module_name]:
                    param_name = self._param_names[p]
                    numel = p.data.numel()
                    self._param_group_idx[param_name] = (
                        group_idx, len(current_param_group), start_p, start_p + numel
                    )
                    current_param_group.append(param_name)
                    start_p += numel

            param_groups.append(current_param_group)
            _, pad_tensor, shard_tensor = self._get_pad_tensor(p.data, start_p, size())
            self._pad_buffers.append(pad_tensor)
            self._shard_buffers.append(shard_tensor)

        assert len(module_groups) == len(param_groups)
        self._num_groups = len(module_groups)
        self._module_group_flags = [0] * len(module_groups)
        self._param_group_flags = [[0] * len(g) for g in param_groups]

        if rank() == 0:
            print('#Tensor fusion groups:', len(module_groups))
            print('Buffer sizes (MB):',
                ', '.join('{:.2f}'.format(buf.numel() * 4 / 1024 / 1024)
                            for buf in self._pad_buffers))
            print('module groups:', module_groups)
            print('parameter groups:', param_groups)

        # -------------------------------------------------------
        # 压缩相关预分配
        # -------------------------------------------------------
        if self._compression:

            # step1：预判每个group的敏感性
            # 分组时已保证同组同质，取第一个module的第一个参数判断即可
            self._group_use_compression = {}
            for group_idx, module_group in enumerate(module_groups):
                first_module = module_group[0]
                first_module_name = self._module_names[first_module]
                group_sensitive = False
                for p in self._module_direct_parameters[first_module_name]:
                    name = self._param_names[p]
                    if self._compression.is_sensitive_layer(name):
                        group_sensitive = True
                        break
                self._group_use_compression[group_idx] = not group_sensitive

            if rank() == 0:
                print('Group use compression:', self._group_use_compression)

            # step2：为非敏感组的每个参数分配compressed buffer中的slot
            # 敏感组参数不分配，运行时走_pad_buffers
            self._compressed_param_offsets = {}
            compressed_group_sizes = []

            for group_idx, module_group in enumerate(module_groups):
                offset = 0
                use_compress = self._group_use_compression[group_idx]

                if use_compress:
                    for module in module_group:
                        module_name = self._module_names[module]
                        for p in self._module_direct_parameters[module_name]:
                            name = self._param_names[p]
                            if p.ndimension() <= 1:
                                # 1D参数：slot等于原始大小，原样通信
                                compressed_size = p.numel()
                            else:
                                matrix = p.data.view(p.shape[0], -1)
                                n, m = matrix.shape
                                rank_compression = min(n, m, self._compression.rank)
                                # p和q交替传，取max保证slot够大
                                compressed_size = max(n * rank_compression,
                                                    m * rank_compression)
                            self._compressed_param_offsets[name] = (
                                group_idx, offset, offset + compressed_size
                            )
                            offset += compressed_size
                # 敏感组：offset不增长，不分配slot

                compressed_group_sizes.append(offset)

            # step3：只为非敏感组分配compressed buffer，敏感组保持None
            self._compressed_pad_buffers = [None] * len(module_groups)
            self._compressed_shard_buffers = [None] * len(module_groups)

            for group_idx, total_size in enumerate(compressed_group_sizes):
                if not self._group_use_compression[group_idx]:
                    # 敏感组：保持None，运行时走_pad_buffers
                    continue
                if total_size == 0:
                    # 非敏感组但全是1D参数（极少见），退化为原始buffer
                    self._group_use_compression[group_idx] = False
                    continue
                pad_num = size() - total_size % size()
                if total_size % size() == 0:
                    pad_num = 0
                self._compressed_pad_buffers[group_idx] = torch.empty(
                    total_size + pad_num,
                    device=self._pad_buffers[group_idx].device
                )
                self._compressed_shard_buffers[group_idx] = torch.empty(
                    (total_size + pad_num) // size(),
                    device=self._pad_buffers[group_idx].device
                )

        else:
            # 无压缩：所有组走原始buffer，统一初始化为False和None
            self._group_use_compression = {i: False for i in range(len(module_groups))}
            self._compressed_pad_buffers = [None] * len(module_groups)
            self._compressed_shard_buffers = [None] * len(module_groups)
            self._compressed_param_offsets = {}

    @torch.no_grad()
    def _get_pad_tensor(self, tensor, numel, size): 
        """
        Get padding tensors
        """
        pad_num = size - numel % size
        pad_tensor = tensor.new_empty(numel+pad_num)
        shard_tensor = tensor.new_empty((numel+pad_num) // size)
        return pad_num, pad_tensor, shard_tensor

    def _register_hooks(self):
        """
        Register hooks for both feed-forward and back-propagation. 
        """
        # find all trainable modules and parameters
        self._register_modules = []
        self._register_parameters = []
        self._module_names = {}             # get module name
        self._module_direct_parameters = {} # get module direct params by name
        
        """ 修改去重方式。
        register_param_names = []
        for module in self._model.modules():
            params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            direct_params = []
            for p in params:
                # avoid repeat registration, e.g. shared parameters
                p_name = self._param_names.get(p)
                if p_name not in register_param_names:
                    register_param_names.append(p_name)
                    direct_params.append(p)
        """
        register_param_ptrs = []  # 改为按内存地址去重
        for module in self._model.modules():
            params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            direct_params = []
            for p in params:
                if p.data_ptr() not in register_param_ptrs:
                    register_param_ptrs.append(p.data_ptr())
                    direct_params.append(p)

            if len(direct_params) > 0:
                module_name = 'module_name_%s_%d' % (module.__class__.__name__, 
                        len(self._register_modules))
                self._module_names[module] = module_name
                self._register_modules.append(module)
                self._register_parameters.extend(direct_params)
                self._module_direct_parameters[module_name] = direct_params
        
        # register forward hooks 
        for i, module in enumerate(self._register_modules):
            if self.exclude_allgather: # for time breakdown record
                break
            module.register_forward_pre_hook(self._forward_pre_hook) # 为模型注册 前向传播钩子
        
        # register backward hooks
        for i, p in enumerate(self._register_parameters):
            if self.exclude_reducescatter: # for time breakdown record
                break
            # ❌ 删掉这行
            # p.grad = p.data.new(p.size()).zero_()
            # ❌ 删掉这三行：
            # p_tmp = p.expand_as(p)
            # grad_acc = p_tmp.grad_fn.next_functions[0][0]
            # grad_acc.register_hook(self._make_hook(p))
            # self._grad_accs.append(grad_acc)
            # ✅ 改成一行：
            p.register_hook(self._make_hook(p))
            #if rank() == 0:
            #    print("register hook for %s" % self._param_names.get(p))
    
    def _make_hook(self, p):
        def hook(grad):
            name = self._param_names.get(p)

            """ 观察 loss 曲线，只有在出现明显的 spike 时再加。
            # 梯度裁剪：限制单个参数的梯度范数
            grad_norm = grad.norm()
            max_norm = 5.0
            if grad_norm > max_norm:
                grad = grad * (max_norm / (grad_norm + 1e-8))
            """
            tensor = grad

            if torch.isnan(grad).any():
                print(f"[NaN in hook] step={self._num_steps}, name={name}, "
                    f"max={grad.abs().max().item():.3e}")

            # 取出该参数所在group的信息
            group_idx, sub_idx, start_p, end_p = self._param_group_idx[name]

            if self._compression and self._num_steps > self._compression.warmup_steps:
                use_compress = self._group_use_compression.get(group_idx, False)

                if use_compress:
                    # ---- 非敏感组：走压缩buffer ----
                    compressed_vector, _, _ = self._compression.compress(
                        tensor, name, step=self._num_steps
                    )
                    _, start, end = self._compressed_param_offsets[name]
                    with torch.no_grad():
                        actual_end = start + compressed_vector.numel()
                        self._compressed_pad_buffers[group_idx][start:actual_end].copy_(
                            compressed_vector.view(-1)
                        )
                        self._param_group_flags[group_idx][sub_idx] = 1
                        for flag in self._param_group_flags[group_idx]:
                            if flag == 0:
                                return grad
                        reduce_scatter_comm.collective_async_(
                            'reduceScatter-group-%d' % group_idx,
                            self._compressed_pad_buffers[group_idx],
                            self._compressed_shard_buffers[group_idx]
                        )
                else:
                    # ---- 敏感组：走原始buffer ----
                    # 影子预热：即使走原始路径也顺便更新p/q
                    if self._compression:
                        shadow_start = (self._compression.warmup_steps
                                        - self._compression.preshadow_steps)
                        if self._num_steps >= shadow_start:
                            self._compression.compress(
                                tensor, name, step=self._num_steps
                            )
                    new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
                    if pad_grad is not None:
                        reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)

            else:
                # ---- warmup阶段或无压缩：全走原始buffer ----
                if self._compression:
                    shadow_start = (self._compression.warmup_steps
                                    - self._compression.preshadow_steps)
                    if self._num_steps >= shadow_start:
                        self._compression.compress(tensor, name, step=self._num_steps)

                new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
                if pad_grad is not None:
                    reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)

            return grad
        return hook

    """ 这是未增加敏感层边界分组的原版
    def _make_hook(self, p):
        def hook(grad):
            name = self._param_names.get(p)

            # -------------------------------------------------------下面的代码加上之后 baseline不报NaN-------------------------------------------------------
            # 强制流同步（确保计算指令已发出）
            # torch.cuda.current_stream().synchronize()
            # 3. 【核心】原位空算，强制触发 L2 Cache 刷新到 VRAM
            # 这是一个读写操作，效果类似于 cpu() 的内存屏障，但在 GPU 内部完成
            # grad.add_(0)

            # 加入这个就完全不报错了
            if torch.isnan(grad).any():
                print(f"[NaN in hook] step={self._num_steps}, name={name}, "
                    f"max={grad.abs().max().item():.3e}")
            # -------------------------------------------------------上面的代码加上之后 baseline不报NaN-------------------------------------------------------
            tensor = grad   # ✅ 用这个！！！

            # ============================================================
            # 修改1：compress调用侧
            # 原逻辑：compress返回单个向量(p或q交替)，写入一个slot
            # 新逻辑：compress返回(p, q)两个向量，拼接后写入同一个slot
            #         buffer slot大小需在预分配时改为 (n + m) * rank
            # ============================================================
            
            if self._compression and self._num_steps > self._compression.warmup_steps:
                
                # P 和 Q每一轮都计算和通信
                # 调用压缩函数，通信p和q时返回 (p, q, None)
                # p_vec, q_vec, _ = self._compression.compress(tensor, name, step=self._num_steps)
                

                compressed_vector, _, _ = self._compression.compress(tensor, name, step=self._num_steps)

                # ← 查is_compressed，而不是靠numel推断
                if self._compression.is_compressed(name):
                    # 走压缩buffer
                    # ... 原有逻辑 ...
                    group_idx, start, end = self._compressed_param_offsets[name]
            
                    with torch.no_grad():
                        actual_end = start + compressed_vector.numel()
                        self._compressed_pad_buffers[group_idx][start:actual_end].copy_(
                            compressed_vector.view(-1)
                        )

                        # P 和 Q每一轮都计算和通信
                        # if p_vec is not None and q_vec is not None:
                            # 矩阵参数：将 p 和 q 拼接后写入 buffer
                            # buffer layout: [p_flat(n*rank) | q_flat(m*rank)]
                            # p_flat = p_vec.view(-1)
                            # q_flat = q_vec.view(-1)
                            # p_numel = p_flat.numel()
                            # q_numel = q_flat.numel()
                            # actual_end = start + p_numel + q_numel
                            # pad_buf[start         : start + p_numel].copy_(p_flat)
                            # pad_buf[start + p_numel : actual_end   ].copy_(q_flat)
                        # else:
                            # 一维tensor（bias等）：p_vec就是原始tensor，q_vec为None，直接写入
                            # flat = p_vec.view(-1)
                            # actual_end = start + flat.numel()
                            # pad_buf[start:actual_end].copy_(flat)
                        

                        # 标记 flag（与原有逻辑一致）
                        _, sub_idx, _, _ = self._param_group_idx[name]
                        self._param_group_flags[group_idx][sub_idx] = 1
                        for flag in self._param_group_flags[group_idx]:
                            if flag == 0:
                                return
                        # 全部 ready，触发通信
                        comm_name = 'reduceScatter-group-%d' % group_idx
                        reduce_scatter_comm.collective_async_(
                            comm_name,
                            self._compressed_pad_buffers[group_idx],
                            self._compressed_shard_buffers[group_idx]
                        )
                else:
                    # gate拒绝：走原始buffer（和warmup阶段完全一样）
                    new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
                    if pad_grad is not None:
                        reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)

                
            else:

                # powersgd的warmup阶段 或 不使用压缩
                # ---- 新增：影子预热，在进入warmup末尾时顺便更新p/q ----
                if self._compression:
                    shadow_start = self._compression.warmup_steps - self._compression.preshadow_steps
                    if self._num_steps >= shadow_start:
                        # 只调用compress触发p/q更新，返回值直接丢弃
                        self._compression.compress(tensor, name, step=self._num_steps)

                # 无论是否做了影子预热，梯度都走原始路径
                # 原有逻辑，push_to_buffer函数满了，才会返回pad_grad != None
                new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
                if pad_grad is not None:
                    reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)
            return grad # 注意：hook 应该返回处理后的 grad
        return hook
        """
    
    """  上面改为压缩版，创建了分支
    def _make_hook(self, p):

        # Add hooks for backward propagation. 

        
        def hook(*ignore): #  在hook中，一旦被触发，就会引用self._push_to_buffer和reduce_scatter_comm.collective_async_两个函数
    
            name = self._param_names.get(p)
            tensor = p.grad.data
            # 下面一行为新加的压缩代码（压缩新增）
            assert not p.grad.requires_grad

            # 添加梯度压缩（压缩新增）
            if self._compression:
                tensor_compressed, ctx, _ = self._compression.compress(tensor, name)
                tensor = tensor_compressed

            # Merging gradient tensors with padding for reduce_scatter推送梯度到缓冲区
            new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
            if pad_grad is not None:  # padding 完毕之后才会通信，# not ready就不会通信。
                # RS 的真正触发点：backward hook,只有当 一个 group 的所有参数都 ready 时，hook 被调用梯度被 copy 到 fusion buffer
                handle = reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)
                #if rank() == 0:
                    #print("BP ReduceScatter:", handle)
        return hook
    """

    def _push_to_buffer(self, name, tensor):
        """
        Push tensor to buffer for fusion.
        """
        group_idx, sub_idx, start_p, end_p = self._param_group_idx[name]
        with torch.no_grad():
            pad_buffer = self._pad_buffers[group_idx]
            pad_buffer[start_p:end_p].copy_(tensor.view(-1))
            self._param_group_flags[group_idx][sub_idx] = 1
            for flag in self._param_group_flags[group_idx]:
                if flag == 0: # not ready
                    return name, None, None
            comm_name = 'reduceScatter-group-%d' % group_idx
            shard_buffer = self._shard_buffers[group_idx]
            return comm_name, pad_buffer, shard_buffer

    def _forward_pre_hook(self, module, input):
        """
        Add hooks for pre-feedfoward.
        """
        if torch.is_grad_enabled() and self._num_steps > 0:
            name = self._module_names.get(module)
            group_idx, sub_idx = self._module_group_idx[name]

            # sync allGather for this group and send for next group
            if sub_idx == 0 and self._module_group_flags[group_idx] == 0:
                
                # 等待上一组AG同步完成（Group0的AG是由step函数发起的）
                all_gather_comm.synchronize()
                # torch.cuda.synchronize() 
                # torch.cuda.current_stream().synchronize()
                # print("Rank %d: Step %d, AllGather group %d time: %.10f sec" % (rank(), self._num_steps, group_idx, ag_time))
                
                # 发起下一轮的AG通信
                self._module_group_flags[group_idx] = 1  # done
                if group_idx < self._num_groups - 1 and self._module_group_flags[group_idx+1] == 0:
                    self._allgather_one_group(group_idx+1)

            # update params for this module
            self._update_one_module(module, name, group_idx)
    
    def _update_one_module(self, module, module_name, group_idx):
        torch.cuda.synchronize()

        # 判断本group走哪套buffer（预分配时已固定）
        use_compress = (
            self._compression is not None
            and self._num_steps > self._compression.warmup_steps + 1
            and self._group_use_compression.get(group_idx, False)
        )

        for p in self._module_direct_parameters[module_name]:
            name = self._param_names.get(p)

            if use_compress:
                # ---- 非敏感组：从compressed buffer读取并decompress ----
                _, start, end = self._compressed_param_offsets[name]

                if p.ndimension() > 1:
                    # 矩阵参数：根据last_sent决定读取p还是q的大小
                    key = (name, tuple(p.shape))
                    n = p.shape[0]
                    m = p.view(p.shape[0], -1).shape[1]
                    rank_c = self._compression.get_rank_for(name, p.shape)
                    sent = self._compression.last_sent.get(key, 'p')
                    curr_size = n * rank_c if sent == 'p' else m * rank_c

                    compressed_vector = self._compressed_pad_buffers[group_idx][
                        start: start + curr_size
                    ]
                    grad = self._compression.decompress(
                        compressed_vector, p.size(), p.numel(), name,
                        step=self._num_steps
                    )
                else:
                    # 1D参数：从compressed buffer原样读取
                    curr_size = p.numel()
                    grad = self._compressed_pad_buffers[group_idx][
                        start: start + curr_size
                    ].view(p.shape).clone()

            else:
                # ---- 敏感组或warmup：从原始buffer读取 ----
                group_idx_p, _, start_p, end_p = self._param_group_idx[name]
                grad = self._pad_buffers[group_idx][start_p:end_p].view(p.shape).clone()

            grad.div_(size())
            self._sgd(p, grad)

    # ============================================================
    # 修改2：decompress调用侧
    # 原逻辑：根据奇偶step判断buffer里存的是p还是q，取对应长度
    # 新逻辑：buffer里始终是[p_flat | q_flat]，按(n*rank, m*rank)分别切出来
    #         拼成(p, q)元组传给decompress
    # ============================================================
    """ 这是未增加敏感层边界分组的原版
    def _update_one_module(self, module, module_name, group_idx):
        torch.cuda.synchronize()
    
        for p in self._module_direct_parameters[module_name]:
            name = self._param_names.get(p)
    
            if self._compression and self._num_steps > self._compression.warmup_steps + 1:
                _, start, end = self._compressed_param_offsets[name]
    
                # ← 核心改动：先查本step该参数是否真正被压缩
                if self._compression.is_compressed(name) and p.ndimension() > 1:
                    # 走压缩buffer路径
                    _, start, end = self._compressed_param_offsets[name]
                    n = p.shape[0]
                    m = p.view(p.shape[0], -1).shape[1]
                    rank_c = self._compression.get_rank_for(name, p.shape)

                    sent = self._compression.last_sent.get((name, tuple(p.shape)), 'p')
                    curr_size = n * rank_c if sent == 'p' else m * rank_c

                    compressed_vector = self._compressed_pad_buffers[group_idx][start:start + curr_size]
                    grad = self._compression.decompress(
                        compressed_vector, p.size(), p.numel(), name, step=self._num_steps
                    )
                    
                    # p和q每一轮都计算和通信的版本
                    # p_numel = n * rank_c
                    # q_numel = m * rank_c
    
                    # buf = self._compressed_pad_buffers[group_idx]
                    # p_received = buf[start           : start + p_numel        ].clone()
                    # q_received = buf[start + p_numel : start + p_numel + q_numel].clone()
    
                    # 传入(p, q)元组，decompress内部解包使用
                    # compressed_input = (p_received, q_received)
                    
                # else:
                    # 一维tensor：直接取原始数据，decompress做透传
                    # curr_size = p.numel()
                    # compressed_input = self._compressed_pad_buffers[group_idx][start:start + curr_size]

    
                # grad = self._compression.decompress(
                    # compressed_input, p.size(), p.numel(), name, step=self._num_steps)
                    
                else:
                    # gate拒绝 或 1D参数：走原始buffer路径
                    group_idx_p, _, start_p, end_p = self._param_group_idx[name]
                    grad = self._pad_buffers[group_idx][start_p:end_p].view(p.shape).clone()

            else:
                # warmup阶段：走原始buffer路径
                group_idx_p, _, start_p, end_p = self._param_group_idx[name]
                grad = self._pad_buffers[group_idx][start_p:end_p].view(p.shape).clone()
    
            grad.div_(size())
            self._sgd(p, grad)
    """ 

    """ 上面改为了压缩版
    def _update_one_module(self, module, module_name, group_idx):
        #Update model parameters in the module
        pad_grad = self._pad_buffers[group_idx]

        for p in self._module_direct_parameters[module_name]:
            # copy grad values from buffers复制梯度值

            name = self._param_names.get(p)
            #if name not in self._param_group_idx:
            #    continue
            group_idx_p, _, start_p, end_p = self._param_group_idx[name]
            assert group_idx_p == group_idx
            # 原来的一句话，加入压缩后改为下面几句：p.grad.data.view(-1).copy_(pad_grad[start_p:end_p])
            # 解压梯度（如果使用了压缩）（压缩新增）
            if self._compression:
                compressed_grad = pad_grad[start_p:end_p]
                decompressed_grad = self._compression.decompress(compressed_grad, p.size(),name)
                p.grad.data.view(-1).copy_(decompressed_grad.view(-1))
            else:
                p.grad.data.view(-1).copy_(pad_grad[start_p:end_p])

            # to be checked: average the grad here平均梯度
            p.grad.data.div_(size())
            # apply one optimizer step应用优化器步骤
            self._sgd(p)
    """
    
    # 手写 SGD（weight decay / momentum / nesterov）
    def _sgd(self, p, grad):
        # ✅ 优先用传入的 grad，兜底用 p.grad SHAN
        if grad is None:
            grad = p.grad
        if grad is None:
            return
        """
        if p.grad is None:
            return
        """

        # 🔹 打印参数 id，用于检查每个参数只被更新一次
        # print(f"_sgd called for param id={id(p)}, name={self._param_names.get(p)}")

        group = self.param_to_group[p]   # ✅ 修正后的 group 获取方式

        if 'momentum' in group:
            # ===== SGD =====
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            d_p = grad

            with torch.no_grad():
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(p)
                        param_state['momentum_buffer'] = buf
                        buf.copy_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        else:
            # ===== AdamW =====
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            state['step'] += 1
            step_n = state['step']

            grad = grad

            with torch.no_grad():
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_sq.clamp_(min=0.0)          # 新增：防止数值误差产生负数
                    
                bias_correction1 = 1 - beta1 ** step_n
                bias_correction2 = 1 - beta2 ** step_n

                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                """ 原本写法
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
                """

                denom = exp_avg_sq.sqrt().add(eps)   # 非原地，去掉 clamp
                update = exp_avg / denom
                p.add_(update, alpha=-step_size)
                                
        # 清梯度
        p.grad = None

    def zero_grad(self):
        pass
    
    def _allgather_one_group(self, group_idx):
        use_compress = (
            self._compression is not None
            and self._num_steps > self._compression.warmup_steps + 1
            and self._group_use_compression.get(group_idx, False)
        )

        if use_compress:
            all_gather_comm.collective_async_(
                "allGather-group-%d" % group_idx,
                self._compressed_pad_buffers[group_idx],
                self._compressed_shard_buffers[group_idx]
            )
        else:
            all_gather_comm.collective_async_(
                "allGather-group-%d" % group_idx,
                self._pad_buffers[group_idx],
                self._shard_buffers[group_idx]
            )
    """  上面增加了敏感层和非敏感层切割分组
    def _allgather_one_group(self, group_idx):
        if self._compression and self._num_steps > self._compression.warmup_steps + 1:  # +1 因为此时step+1了
            all_gather_comm.collective_async_(
                "allGather-group-%d" % group_idx,
                self._compressed_pad_buffers[group_idx],
                self._compressed_shard_buffers[group_idx]
            )
        else:
            pad_grad = self._pad_buffers[group_idx]
            shard_grad = self._shard_buffers[group_idx]
            all_gather_comm.collective_async_("allGather-group-%d" % group_idx, pad_grad, shard_grad)
    """  

    """"    上面改为了压缩版
    def _allgather_one_group(self, group_idx):
        # Apply allgather on one group
        pad_grad = self._pad_buffers[group_idx]
        shard_grad = self._shard_buffers[group_idx]
        comm_name = "allGather-group-%d" % group_idx
        # 真正发起 AG 的地方
        all_gather_comm.collective_async_(comm_name, pad_grad, shard_grad)        
    """
    
    def _bp_barrier(self):
        """
        Synchronize the reduce-scatter operations and start the all-gather on the first group.
        """
        # 计时（所有 reduce-scatter 通信 + 被 overlap 剩下的尾巴）  = RS_total_time − backward_overlap_time
        # 所有RS的同步操作
        reduce_scatter_comm.synchronize()
        # print("Rank %d: Step %d, ReduceScatter time: %.10f sec" % (rank(), self._num_steps, rs_time))
        
        # 第一次调用AG 立即调用group0的AG通信
        self._allgather_one_group(group_idx=0)
        #if rank() == 0:
            #print("param group flags:", self._param_group_flags)
            #print("module group flags:", self._module_group_flags)
        
        # Clear flags
        for group_idx in range(len(self._param_group_flags)):
            self._param_group_flags[group_idx] = [0] * len(self._param_group_flags[group_idx])
        self._module_group_flags = [0] * self._num_groups

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        if size() > 1:
            self._bp_barrier()
        #else:
        #    todo: step with non-distributed optimzier
        # Note: the last step is skipped
        self._num_steps += 1

        #test_tensor = torch.tensor([rank()]).cuda()
        #print("test tensor before:", test_tensor)
        #comm.bcast(test_tensor, 0)
        #comm.synchronize()
        #print("test tensor after:", test_tensor)

# 将任意 PyTorch Optimizer 动态包装成 DeAR 分布式优化器
def DistributedOptimizer(optimizer, model, compression=None, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, fp16=False, mgwfbp=False, rdma=False, multi_job_scheduling=False, exclude_parts=''):
    """
    Wrap optimizer to gurantee the consistency. 
    Warning: some functions are not supported now, so we will simply skip these parameters.
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, model, exclude_parts=exclude_parts,compression=compression) # 原本compression参数是没有要的（压缩新增）

def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    for name, p in params:
        if p is not None:
            comm.bcast(p.view(-1), root_rank)
    comm.synchronize()


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                # p.grad = p.data.new(p.size()).zero_()
                pass
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if p is not None and not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, _ in params:
        if key in callbacks:
            callbacks[key]()

# 备用，但主流程 没有用它（被 RS+AG 替代）
def allreduce(tensor, name=None):
    comm.allReduce(tensor.view(-1))
    comm.synchronize()
    return tensor/size()
