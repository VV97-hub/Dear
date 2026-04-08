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

            # 测试是否有重复
            if rank() == 0:
                print(f"Total registered parameters: {len(self._register_parameters)}")
                # 检查是否还有重复的 data_ptr
                ptrs = [p.data_ptr() for p in self._register_parameters]
                duplicates = len(ptrs) - len(set(ptrs))
                print(f"Duplicate tensors in registered parameters: {duplicates}")

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
        
    def _generate_groups_with_threshold(self):
        """
        Generate groups with buffer size threshold (in MB) for tensor fusion. 
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


    def _prepare_tensor_fusion(self, module_groups):
        """
        Prepare tensor fusion based on module groups, e.g. [[m1, m2], [m3]] in forward order.
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
        if self._compression:
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
                            # 保守起见取 max，因为 p 和 q 交替传
                            compressed_size = max(n * rank_compression, m * rank_compression)
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
            p.grad = p.data.new(p.size()).zero_()
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_hook(p)) # 为梯度注册 反向传播钩子
            self._grad_accs.append(grad_acc)
            #if rank() == 0:
            #    print("register hook for %s" % self._param_names.get(p))
    
    def _make_hook(self, p):
        def hook(*ignore):
            name = self._param_names.get(p)
                
            tensor = p.grad.data

            if self._compression:

                # 调用压缩函数
                compressed_vector, _, _ = self._compression.compress(tensor, name, step=self._num_steps)
                # 写入压缩专用 buffer
                group_idx, start, end = self._compressed_param_offsets[name]
                pad_buf = self._compressed_pad_buffers[group_idx]

                # 重要：写入时使用 compressed_vector 的实际大小，而不是 end-start
                # 因为 P 和 Q 的长度可能不一致，我们 buffer 是按 max 分配的
                actual_end = start + compressed_vector.numel()
                # print(f"name={name}, compressed_vector size={compressed_vector.numel()}, buf slot size={end-start}")
                pad_buf[start:actual_end].copy_(compressed_vector.view(-1))
                # 标记 flag（复用原有 flag 机制）
                # 这里没有和（原有逻辑）一样： 调用pad_grad，而是自己复现了（如果一个组的flag都ok了，说明整个通信组才可以触发通信）
                _, sub_idx, _, _ = self._param_group_idx[name]
                self._param_group_flags[group_idx][sub_idx] = 1
                for flag in self._param_group_flags[group_idx]:
                    if flag == 0:
                        return
                # 全部 ready，触发通信（用压缩 buffer）
                comm_name = 'reduceScatter-group-%d' % group_idx
                reduce_scatter_comm.collective_async_(
                    comm_name,
                    self._compressed_pad_buffers[group_idx],
                    self._compressed_shard_buffers[group_idx]
                )
            else:
                # 原有逻辑，push_to_buffer函数满了，才会返回pad_grad != None
                new_name, pad_grad, shard_grad = self._push_to_buffer(name, tensor)
                if pad_grad is not None:
                    reduce_scatter_comm.collective_async_(new_name, pad_grad, shard_grad)
        return hook
    
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
                
                # timing for allGather记录allgather的时间   AG_exposed =AG 没被 forward compute 覆盖的剩余通信时间
                torch.cuda.synchronize()
                t0 = time.time()
                
                # 等待上一组AG同步完成（Group0的AG是由step函数发起的）
                all_gather_comm.synchronize()

                torch.cuda.synchronize()
                t1 = time.time()

                ag_time = t1 - t0
                # print("Rank %d: Step %d, AllGather group %d time: %.10f sec" % (rank(), self._num_steps, group_idx, ag_time))
                
                # 发起下一轮的AG通信
                self._module_group_flags[group_idx] = 1  # done
                if group_idx < self._num_groups - 1 and self._module_group_flags[group_idx+1] == 0:
                    self._allgather_one_group(group_idx+1)

            # update params for this module
            self._update_one_module(module, name, group_idx)
    
    def _update_one_module(self, module, module_name, group_idx):
        
        for p in self._module_direct_parameters[module_name]:
            name = self._param_names.get(p)

            if self._compression:
                _, start, end = self._compressed_param_offsets[name]
                # 判断逻辑要与 compress 严格一致
                if p.ndimension() > 1:
                    # 矩阵：根据 step 偏移决定取 P 还是 Q.在 PowerSGD 中，P 和 Q 维度不同。我们需要知道当前这一步对应的向量到底占了多少长度。
                    n, m = p.shape[0], p.view(p.shape[0], -1).shape[1]
                    rank_c = min(n, m, self._compression.rank)
                    # 根据步数计算当前收到的数据长度
                    # 同样逻辑：如果当前 step 是奇数，说明 Buffer 里存的是上一步传的 P
                    if self._num_steps % 2 != 0:
                        curr_size = n * rank_c
                    else:
                        curr_size = m * rank_c
                else:
                # 一维向量：直接取全部
                    curr_size = p.numel()
                
                compressed_vector = self._compressed_pad_buffers[group_idx][start:start + curr_size]
                decompressed_grad = self._compression.decompress(compressed_vector, p.size(), p.numel(), name, step=self._num_steps)
                p.grad.data.view(-1).copy_(decompressed_grad.view(-1))
            else:
                group_idx_p, _, start_p, end_p = self._param_group_idx[name]
                p.grad.data.view(-1).copy_(self._pad_buffers[group_idx][start_p:end_p])

            # 恢复原来的shape，用_sgd()函数更新
            p.grad.data.div_(size())
            

            self._sgd(p)

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
    def _sgd(self, p):
        if p.grad is None:
            return

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

            d_p = p.grad

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

            grad = p.grad

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
        if self._compression:
            all_gather_comm.collective_async_(
                "allGather-group-%d" % group_idx,
                self._compressed_pad_buffers[group_idx],
                self._compressed_shard_buffers[group_idx]
            )
        else:
            pad_grad = self._pad_buffers[group_idx]
            shard_grad = self._shard_buffers[group_idx]
            all_gather_comm.collective_async_("allGather-group-%d" % group_idx, pad_grad, shard_grad)

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
        torch.cuda.synchronize()
        t0 = time.time()
        
        # RS的同步操作
        reduce_scatter_comm.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()

        rs_time = t1 - t0

        # print("Rank %d: Step %d, ReduceScatter time: %.10f sec" % (rank(), self._num_steps, rs_time))
        
        # 立即调用group0的AG通信
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
                p.grad = p.data.new(p.size()).zero_()
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
