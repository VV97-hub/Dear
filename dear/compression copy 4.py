"""1.改回了一轮p一轮q的计算和通信，发现同时计算和通信pq没有意义，那和powersgd是不同的，powersgd一轮需要通信两次。
2.加入敏感组和非敏感组分割的策略（但由于一些代码改动导致通信加长了很多）"""
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time
import math
import utils
from scipy import stats

import torch
import numpy as np
from collections import deque


"""
本实验使用的模型配置：
bert.encoder.layer.{N}.attention.self.query.weight   ← 敏感
bert.encoder.layer.{N}.attention.self.key.weight     ← 敏感  
bert.encoder.layer.{N}.attention.self.value.weight   ← 可压缩
bert.encoder.layer.{N}.attention.output.dense.weight ← 可压缩
bert.encoder.layer.{N}.intermediate.dense.weight     ← 最适合压缩 (768→3072)
bert.encoder.layer.{N}.output.dense.weight           ← 可压缩 (3072→768)
bert.embeddings.word_embeddings.weight               ← 不压缩 (30522×768，奇异值均匀)
bert.embeddings.position_embeddings.weight           ← 不压缩 (512×768，太小)
bert.embeddings.token_type_embeddings.weight         ← 不压缩 (2×768，1D退化)
bert.pooler.dense.weight                             ← 可压缩
cls.predictions.transform.dense.weight              ← 可压缩
cls.predictions.decoder.weight                      ← 不压缩 (与word_embeddings权重共享!)
"""
        
class Reducer:
    def __init__(self, random_seed=0, device=None, timer=None):
        # 生成相同的随机种子，初始化相同的随机向量
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        """
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        """
        
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()
    
# （新增压缩方法HalfRank——优势：通信次数更少 1 step 1 powersgd iteration）
class HalfRankKReducer(Reducer):
    """
    动态rank版本的HalfRankK压缩器。
    支持按训练阶段动态切换rank，rank变化时自动重建p/q内存，
    避免形状不匹配导致的crash。
    """

    def __init__(self, random_seed=0, device=None, timer=None, rank=2,
             rank_schedule=None, warmup_steps=200, preshadow_steps=300):
        """
        参数说明：
          rank          : 默认/初始rank值，当rank_schedule未覆盖当前step时使用。
          rank_schedule : 动态rank调度器，支持两种形式：
                          1. dict  {step_threshold: rank}
                             例：{0: 4, 1000: 2, 5000: 1}
                             表示 step<1000 用rank=4，1000<=step<5000 用rank=2，
                             step>=5000 用rank=1。
                          2. callable  fn(step) -> int
                             完全自定义，可实现任意调度逻辑。
                          如果为 None，则始终使用 rank 参数（固定rank，兼容旧行为）。
        """
        super().__init__(random_seed, device, timer)

        # 默认rank（兜底值）
        self.default_rank = rank

        # 动态rank调度器，None表示退化为固定rank
        self.rank_schedule = rank_schedule

        # ---- 每个tensor key的内存 ----
        self.p_memory = {}          # {key: Tensor(n, rank)}
        self.q_memory = {}          # {key: Tensor(m, rank)}
        self.residuals = {}         # {key: Tensor(n, m)}  Error Feedback残差
        self.last_input = {}        # {key: Tensor(n, m)}  上一轮原始梯度，用于更新残差
        self.last_sent = {}   # {key: 'p' or 'q'}

        # 记录每个key上次使用的rank，用于检测rank是否发生变化
        # 结构：{key: int}
        # compress写入，decompress读取，保证二者使用完全相同的rank
        self.rank_memory = {}

        self.name = 'halfrankk'

        self.warmup_steps = warmup_steps  # 供 dopt_rsag 读取，替换硬编码的 6000
        self.preshadow_steps = preshadow_steps  # 提前多少步开始预热p/q

        self.device = device

        self.sensitive_name_keywords = [
            'attention.self.query',
            'attention.self.key',
            'embeddings',
            'LayerNorm',
            'cls.predictions.bias',
            'cls.seq_relationship',
        ]

    def is_sensitive_layer(self, name):
        """供dopt_rsag分组阶段查询，判断参数是否属于敏感层"""
        name_lower = name.lower()
        for kw in self.sensitive_name_keywords:
            if kw.lower() in name_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # 工具方法：根据当前step计算本轮实际使用的rank
    # ------------------------------------------------------------------
    def _get_rank(self, step, n, m):
        """
        计算当前step应使用的rank值。

        规则：
          1. 如果提供了rank_schedule，按调度器确定基础rank；
          2. 再用 min(n, m, base_rank) 保证rank不超过矩阵维度上限；
          3. 保证最小值为1，防止rank退化为0导致空矩阵。

        参数：
          step : 当前训练步数（外部传入，从dopt_rsag的step获取）
          n, m : 当前tensor reshape后的矩阵维度
        """
        if self.rank_schedule is None:
            # 未配置调度器，使用固定rank（兼容旧接口）
            base_rank = self.default_rank
        elif callable(self.rank_schedule):
            # 函数式调度：完全由外部逻辑控制
            base_rank = self.rank_schedule(step)
        else:
            # dict调度：找到所有不超过当前step的阈值中最大的那个
            # 例如 schedule={0:4, 1000:2, 5000:1}，step=1500 → 取阈值1000 → rank=2
            applicable = {s: r for s, r in self.rank_schedule.items() if s <= step}
            if applicable:
                base_rank = applicable[max(applicable)]
            else:
                # step比所有阈值都小时，退回default_rank
                base_rank = self.default_rank

        # TODO 未来可在这里加入按 tensor 大小动态调整 rank 的逻辑
        """ 
        numel = n * m
        if numel > 1_000_000:
            base_rank = min(base_rank + 2, 8)   # 大矩阵适当提高
        elif numel < 10_000:
            base_rank = max(base_rank - 1, 1)   # 小矩阵适当降低

        # rank不能超过矩阵的任意一维，也不能为0
        """

        return max(1, min(n, m, base_rank))
        

    # ------------------------------------------------------------------
    # 工具方法：检测rank是否变化，如果变化则清除该key的全部内存
    # ------------------------------------------------------------------
    def _maybe_reset_memory(self, key, new_rank):
        """
        如果当前key的rank与上次不同，说明调度器切换了阶段。
        此时必须清除p_memory、q_memory、residuals、last_input，
        否则旧shape的tensor会在新rank下引发矩阵乘法形状不匹配错误。

        rank变化时residual也必须清零，因为旧rank下累积的残差
        对新rank的p/q基底没有意义，强行保留会污染梯度。
        """
        old_rank = self.rank_memory.get(key, None)
        if old_rank is not None and old_rank != new_rank:
            # rank发生切换，清除该key的全部历史内存
            self.p_memory.pop(key, None)
            self.q_memory.pop(key, None)
            self.residuals.pop(key, None)    # 残差随rank清零，防止梯度污染
            self.last_input.pop(key, None)
        # 记录本轮rank，供decompress同步读取
        self.rank_memory[key] = new_rank

    @property
    def rank(self):
        """
        暴露给 dopt_rsag._prepare_tensor_fusion 使用。
        返回整个训练过程中可能出现的最大 rank，
        用于静态预分配 compressed_pad_buffer 的最坏情况大小。
        固定rank时直接返回 default_rank；
        dict 调度时取所有 rank 值的最大值；
        callable 调度时无法静态确定，退回 default_rank（调用方负责保证够大）。
        """
        if self.rank_schedule is None:
            return self.default_rank
        
        # TODO 这里逻辑有问题，为什么不按照轮次来选择：
        elif callable(self.rank_schedule):
            # 函数式调度无法静态分析，由外部保证 default_rank >= 实际最大rank
            return self.default_rank
        
        # TODO 这里逻辑有问题，为什么不按照轮次来选择：
        else:
            # dict 调度：取所有阶段 rank 的最大值，保证 buffer 足够大
            return max(self.rank_schedule.values())

    # compression.py 里加一个方法
    def get_rank_for(self, name, shape):
        """
        供外部（dopt_rsag）查询某个tensor当前实际使用的rank。
        封装 key 的构造方式，调用方不需要知道内部 key 格式，
        保证与 compress 端的 key 定义严格一致。
        """
        key = (name, tuple(shape))
        return self.rank_memory.get(key, self.default_rank)

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        orthogonalize(vector)

    def compress(self, tensor, name=None, step=0, **kwargs):
        if tensor.ndimension() <= 1:
            return tensor, None, None

        if name is None:
            name = 'default'


        # ---- 后续原有逻辑不变 ----
        matrix = tensor.reshape(tensor.shape[0], -1)
        n, m = matrix.shape

        rank = self._get_rank(step, n, m)
        key = (name, tuple(tensor.shape))
        self._maybe_reset_memory(key, rank)

        shadow_start = self.warmup_steps - self.preshadow_steps

        # ---- 影子预热阶段 or 正式压缩阶段，都需要p/q ----
        is_shadow = (step <= self.warmup_steps)

        if key not in self.residuals:
            self.residuals[key] = torch.zeros_like(matrix)

        if key not in self.p_memory:
            self.p_memory[key] = torch.zeros(n, rank, device=tensor.device)
            self.q_memory[key] = torch.zeros(m, rank, device=tensor.device)
            self.set_random(self.q_memory[key])  # 随机初始化q
            orthogonalize(self.q_memory[key])  # 仅首次正交化q

        p = self.p_memory[key]
        q = self.q_memory[key]
        
        # p和q的影子预热机制：
        if is_shadow:
        # 影子阶段：只更新 p/q，不动 last_input，不动 residuals
            with torch.no_grad():
                if step % 2 == 0:
                    orthogonalize(q)
                    torch.matmul(matrix, q, out=p)
                else:
                    orthogonalize(p)
                    torch.matmul(matrix.t(), p, out=q)
            # ✅ 关键：不写 last_input，让它保持未初始化或上一正式步的值
            return tensor, None, None

         # ---- 正式压缩阶段 ----
        pure_gradient = matrix.clone()          # 存纯梯度（加残差之前）
        matrix = pure_gradient + self.residuals[key]
        
        # ✅ 只有正式阶段才写 last_input
        self.last_input[key] = pure_gradient

        with torch.no_grad():
            if step % 2 == 0:
                # 偶数步：用当前q投影，计算p = M @ q
                orthogonalize(q)
                torch.matmul(matrix, q, out=p)
                self.p_memory[key] = p
                self.last_sent[key] = 'p'   # ← 新增
                return p.clone(), None, None
            else:
                # 奇数步：用当前p投影，计算q = M^T @ p
                orthogonalize(p)
                torch.matmul(matrix.t(), p, out=q)
                self.q_memory[key] = q
                self.last_sent[key] = 'q'   # ← 新增
                return q.clone(), None, None
        """ 每一步都计算和通信p和q的做法，这样对精度没用很大提升(与上文互相替代)
        with torch.no_grad():
            torch.matmul(matrix, q, out=p)        # P = M @ Q
            orthogonalize(p)                       # P̂ = orth(P)
            torch.matmul(matrix.t(), p, out=q)    # Q = M^T @ P̂
            self.p_memory[key] = p
            self.q_memory[key] = q
        # self.last_input[key] = original_matrix
        # return p.clone(), q.clone(), None
        """
        

 
    def decompress(self, compressed_data, original_tensor_size, numel, name, step=0):
        """
        用通信后的p和q重构梯度，并更新Error Feedback残差。
 
        修改说明：
          原版根据奇偶step决定覆盖p或q，现在每步都同时收到新的p和q，
          直接用通信后的p和q重构，不再需要奇偶判断。
 
        参数：
          compressed_data: (p, q) 元组，allreduce之后的结果
        """
        if compressed_data is None:
            return torch.zeros(original_tensor_size)
 
        if len(original_tensor_size) <= 1:
            return compressed_data.view(original_tensor_size)
 
        key = (name, tuple(original_tensor_size))

        with torch.no_grad():
            sent = self.last_sent.get(key, 'p')   # ← 读记录，不再用 step 推断
            if sent == 'p':
                # 收到新 p，q 是旧的——先把 p 正交化再重构
                p_updated = compressed_data.view(self.p_memory[key].shape)
                # orthogonalize(p_updated)
                self.p_memory[key].copy_(p_updated)
            else:
                # 收到新 q，p 是旧的——先把 q 正交化再重构
                q_updated = compressed_data.view(self.q_memory[key].shape)
                # orthogonalize(q_updated)
                self.q_memory[key].copy_(q_updated)
        """每一步都计算和通信p和q的做法，这样对精度没用很大提升（与上文相互代替）
        # 解包通信后的p和q
        p_received, q_received = compressed_data
 
        with torch.no_grad():
            self.p_memory[key].copy_(p_received.view(self.p_memory[key].shape))
            self.q_memory[key].copy_(q_received.view(self.q_memory[key].shape))

        """

        p = self.p_memory[key]
        q = self.q_memory[key]
        # 重构低秩近似梯度
        reconstructed = p @ q.t()
 
        # [EF] 更新残差：残差 = (原始梯度 + 旧残差) - 低秩近似
        if key in self.last_input:
            with torch.no_grad():
                self.residuals[key] = self.last_input[key] - reconstructed
 
        return reconstructed.view(original_tensor_size)
        
    # reduce方法废弃不用，输入和输出与要求的不一样
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # Communicate rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        if self.p_memory is None:
            with self.timer("reduce.allocate_memory", verbosity=2):
                p_total_size = 0
                q_total_size = 0
                for tensor, _, _ in high_rank_tensors:
                    matrix = tensor.view(tensor.shape[0], -1)
                    n, m = matrix.shape
                    rank = min(n, m, self.rank)
                    p_total_size += n * rank
                    q_total_size += m * rank
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)

        with self.timer("reduce.build_index", verbosity=2):
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx : p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx : q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        if self.next_operation == "p":
            self.next_operation = "q"
            with self.timer("reduce.normalize.q", verbosity=2):
                for q in qs:
                    if memory_is_uninitialized:
                        self.set_random(q)
                    else:
                        orthogonalize(q)

            with self.timer("reduce.compute.p", verbosity=2):
                for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                    matrix = tensor.view(tensor.shape[0], -1)
                    torch.matmul(matrix, q, out=p)

            with self.timer("reduce.fill_memory"):
                for p, q, (tensor, _, mem) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    # Keep what we couldn't send in memory
                    mem.data[:] = (matrix - torch.einsum("nr, mr -> nm", (p, q))).view(
                        *tensor.shape
                    )

            with self.timer("reduce.p", verbosity=2):
                all_reduce(self.p_memory)
                bits_communicated += n_bits(self.p_memory)
                self.p_memory.data[:] /= self.n_workers

        elif self.next_operation == "q":
            self.next_operation = "p"
            with self.timer("reduce.normalize.p", verbosity=2):
                for p in ps:
                    orthogonalize(p)

            with self.timer("reduce.compute.q", verbosity=2):
                for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    torch.matmul(matrix.t(), p, out=q)

            with self.timer("reduce.fill_memory", verbosity=2):
                for p, q, (tensor, _, mem) in zip(ps, qs, high_rank_tensors):
                    matrix = tensor.view(tensor.shape[0], -1)
                    # Keep what we couldn't send in memory
                    mem.data[:] = (matrix - torch.einsum("nr, mr -> nm", (p, q))).view(
                        *tensor.shape
                    )

            with self.timer("reduce.q", verbosity=2):
                all_reduce(self.q_memory)
                bits_communicated += n_bits(self.q_memory)
                self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, _) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                out.data[:] = torch.einsum("nr, mr -> nm", (p, q)).view(*tensor.shape)

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated



class NoneCompressor():
    def __init__(self,**kwargs):
        self.name = 'none'
    
    # 让 NoneCompressor 返回 False（很优雅），防止不压缩时，进入压缩的代码。
    def __bool__(self):
        return False

    def compress(self, tensor):
        return tensor, tensor.dtype

    def decompress(self, tensor, ctc):
        z = tensor 
        return z 


# def orthogonalize(matrix):
#     # This is super slow
#     r = torch.empty(1, device=matrix.device)  # dummy memory, we don't care about r
#     torch.qr(matrix, out=(matrix, r))
#     del r

# 在列范数过小时跳过投影：
@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        col = matrix[:, i : i + 1]
        norm = torch.sqrt(torch.sum(col ** 2))
        if norm < eps:          # 新增：列向量几乎为零，跳过
            continue
        col /= norm + eps
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col
            
""" powersgd原版
@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col
"""

# （新增压缩方法BasicPowerSGD——优势：GPU计算实现方式的内核效率高）  @ TODO还没修改
class PowerSGDCompressor():
    """
    PowerSGD: Practical Low-Rank Gradient Compression for Distributed Deep Learning
    """

    def __init__(self, rank=1, reuse_query=False, n_power_iterations=0, random_seed=42):
        self.name = 'powersgd'
        self.rank = rank
        self.reuse_query = reuse_query
        self.n_power_iterations = n_power_iterations
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.residuals = {}
        self.p_memory = {}
        self.q_memory = {}

    def _process_data_before_selecting(self, name, data):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, data, reconstructed_tensor):
        self.residuals[name].data = data - reconstructed_tensor

    def clear(self):
        self.residuals = {}
        self.p_memory = {}
        self.q_memory = {}

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name is None:
                name = 'default'
            
            self._process_data_before_selecting(name, tensor.data)
            
            # Handle rank 1 tensors (no compression needed)
            if tensor.ndimension() <= 1:
                return tensor, None, None
            
            # Reshape to matrix
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m, self.rank)
            
            # Initialize or reuse p and q memory
            if name not in self.p_memory or self.p_memory[name].shape != (n, rank):
                self.p_memory[name] = torch.empty(n, rank, device=tensor.device)
                self.q_memory[name] = torch.empty(m, rank, device=tensor.device)
            
            p = self.p_memory[name]
            q = self.q_memory[name]
            
            # Sample query vector q
            if not self.reuse_query or name not in self.q_memory:
                torch.manual_seed(self.rng.randint(1_000_000_000))
                q.data[:] = torch.randn(*q.shape, device=tensor.device)
            
            # Optional power iterations
            for _ in range(self.n_power_iterations):
                torch.matmul(matrix, q, out=p)
                orthogonalize(p)
                torch.matmul(matrix.t(), p, out=q)
                orthogonalize(q)
            
            # Compute p and q
            torch.matmul(matrix, q, out=p)
            orthogonalize(p)
            torch.matmul(matrix.t(), p, out=q)
            
            # Reconstruct tensor
            reconstructed = torch.matmul(p, q.t())
            reconstructed = reconstructed.view(tensor.shape)
            
            self._process_data_after_residual(name, tensor.data, reconstructed)
            
            return (p, q), None, None

    def decompress(self, compressed_data, original_tensor_size):
        if compressed_data is None:
            return torch.zeros(original_tensor_size)
        
        p, q = compressed_data
        reconstructed = torch.matmul(p, q.t())
        return reconstructed.view(original_tensor_size)
    
class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1

    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device) 
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition

    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            self._process_data_before_selecting(name, tensor.data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0. 
            self.values[name] = values
            self.indexes[name] = indexes

            self._process_data_after_residual(name, tensor.data)

            return tensor, indexes, values

    def get_residuals(self, name, like_tensor):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(like_tensor.data)
        return self.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = self.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = self.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[self.indexes[name]] += values.data

    def decompress(self, tensor, original_tensor_size):
        return tensor


class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'

    def _process_data_before_selecting(self, name, data):
        data.add_(self.residuals[name].data)


import bit2byte
class SignCompressor:
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""
    def __init__(self):
        self.zc = None
        self.name = 'signum'

    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data, original_tensor):
        pass

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        packed_data = src_tensor
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, packed_data

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        self._process_data_before_selecting(name, tensor)
        packed_tensor, packed_data = self.packing(tensor)
        self._process_data_after_residual(name, packed_data, tensor)
        return packed_tensor, None, None

    def decompress(self, tensor, original_tensor_size):
        dst_tensor = self.unpacking(tensor, original_tensor_size)
        return dst_tensor


class EFSignCompressor(SignCompressor):
    def __init__(self):
        super().__init__()
        self.zc = None
        self.name = 'efsignum'
        self.residuals = {}

    def _process_data_before_selecting(self, name, data):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, packed_data, original_tensor):
        self.residuals[name] = original_tensor - packed_data


class GaussianCompressor(TopKCompressor):
    """
    """

    def __init__(self):
        super().__init__()
        self.name = 'gaussian'
        self.iterations = {}
        self.sparsities = []

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            indexes = indexes[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


compressors = {
        'none': NoneCompressor,
        None: NoneCompressor,
        'topk': TopKCompressor,
        'eftopk': EFTopKCompressor, #TopK with error-feedback
        'gaussian': GaussianCompressor, #GaussianK with error-feedback

        'signum': SignCompressor,
        'efsignum': EFSignCompressor,
        'halfrankk':HalfRankKReducer
        }
