"""本备份是一轮传p一轮传q的powersgd的备份2026/4/2，下一步打算试试一轮内同时传p和q的powersgd"""
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time
import math
import utils
from scipy import stats

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
                 rank_schedule=None, warmup_steps=200, min_compression_numel=16384):
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
        self.last_input = {}        # {key: Tensor(n, m)}  保留调试信息

        # 记录每个key上次使用的rank，用于检测rank是否发生变化
        # 结构：{key: int}
        # compress写入，decompress读取，保证二者使用完全相同的rank
        self.rank_memory = {}

        self.name = 'halfrankk'

        self.warmup_steps = warmup_steps  # 供 dopt_rsag 读取，替换硬编码的 6000
        # 仅压缩“大张量”，小张量走原始通信以降低精度损失
        self.min_compression_numel = int(min_compression_numel)

        self.device = device

    def is_p_step(self, step):
        return step % 2 == 0

    def factor_kind(self, step):
        return 'p' if self.is_p_step(step) else 'q'

    def factor_kind_for_update(self, step):
        # dopt_rsag.step() 在 backward 之后自增 step，因此更新阶段需要回看上一轮发送的因子。
        return 'p' if step % 2 != 0 else 'q'

    def _should_skip_by_name(self, name):
        if not name:
            return False
        lower_name = name.lower()
        if lower_name.endswith("bias") or ".bias" in lower_name:
            return True
        if "embedding" in lower_name or "embeddings" in lower_name:
            return True
        if "layernorm" in lower_name or "layer_norm" in lower_name:
            return True
        return False

    def should_compress_tensor(self, tensor, name=None):
        if self._should_skip_by_name(name):
            return False
        return tensor.ndimension() > 1 and tensor.numel() >= self.min_compression_numel

    def should_compress_shape(self, shape, name=None):
        if self._should_skip_by_name(name):
            return False
        if len(shape) <= 1:
            return False
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel >= self.min_compression_numel

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

    def get_factor_numel(self, shape, name=None, factor_kind='p', rank=None):
        if not self.should_compress_shape(shape, name=name):
            numel = 1
            for dim in shape:
                numel *= int(dim)
            return numel

        n = int(shape[0])
        m = 1
        for dim in shape[1:]:
            m *= int(dim)
        if rank is None:
            rank = self.rank
        rank = min(n, m, rank)
        if factor_kind == 'p':
            return n * rank
        return m * rank

    def _orthogonalize_factor(self, factor):
        if factor.numel() == 0:
            return
        if factor.shape[1] == 1:
            col = factor[:, :1]
            col /= torch.linalg.vector_norm(col) + 1e-8
            return
        try:
            q = torch.linalg.qr(factor, mode='reduced').Q
            factor.copy_(q)
        except RuntimeError:
            orthogonalize(factor)

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        self._orthogonalize_factor(vector)

    def compress(self, tensor, name=None, step=0, **kwargs):
        """
        对单个tensor执行半步PowerSGD压缩（奇偶步交替算p/q）。

        动态rank逻辑：
          - 每次调用先通过_get_rank计算本轮rank；
          - 若rank与上次不同，_maybe_reset_memory会清空该key的所有历史；
          - 之后按新rank重新初始化p/q，流程与固定rank完全一致。
        """
        if not self.should_compress_tensor(tensor, name=name):
            # 一维tensor不压缩（如bias），直接透传
            return tensor, None, None

        if name is None:
            name = 'default'

        # reshape为二维矩阵：第0维保持，其余flatten
        grad_matrix = tensor.reshape(tensor.shape[0], -1)
        n, m = grad_matrix.shape

        # ---- 动态rank核心：计算本轮rank并按需重建内存 ----
        rank = self._get_rank(step, n, m)
        # key同时包含shape，防止同name不同shape的tensor复用同一内存
        key = (name, tuple(tensor.shape))
        # 检测rank变化，变化时清除旧内存（含residual）
        self._maybe_reset_memory(key, rank)
        # ---- 动态rank核心结束 ----

        # [EF] 初始化residual（首次或rank切换后重建）
        if key not in self.residuals:
            self.residuals[key] = torch.zeros_like(grad_matrix)


        # [EF] 使用“梯度 + 历史残差”作为本轮被压缩目标。
        # 这里保持 ACP-SGD 语义：残差在本地压缩阶段更新，而不是等通信后再更新。
        residual = self.residuals[key]
        matrix = grad_matrix + residual
        self.last_input[key] = matrix.clone()

        # 初始化p/q内存（首次或rank切换后重建，形状由当前rank决定）
        if key not in self.p_memory:
            self.p_memory[key] = torch.zeros(n, rank, device=tensor.device)
            self.q_memory[key] = torch.zeros(m, rank, device=tensor.device)
            # ACP-SGD 会为两个因子都保留持久状态；rank 切换后也重新随机化两者，
            # 避免在 q-step 上用全零的 P 退化成空更新。
            self.set_random(self.p_memory[key])
            self.set_random(self.q_memory[key])

        p = self.p_memory[key]
        q = self.q_memory[key]

        # 压缩计算在no_grad下进行，避免污染反向传播计算图
        with torch.no_grad():
            if self.is_p_step(step):
                # 偶数步：用当前q投影，计算p = M @ q
                self._orthogonalize_factor(q)
                torch.matmul(matrix, q, out=p)
                self.p_memory[key] = p
                residual.add_(grad_matrix - p @ q.t())
                return p.clone(), None, None
            else:
                # 奇数步：用当前p投影，计算q = M^T @ p
                self._orthogonalize_factor(p)
                torch.matmul(matrix.t(), p, out=q)
                self.q_memory[key] = q
                residual.add_(grad_matrix - p @ q.t())
                return q.clone(), None, None

    def decompress(self, compressed_data, original_tensor_size, numel, name, step=0, factor_kind=None):
        """
        用通信后的p或q重构梯度，并更新Error Feedback残差。

        动态rank注意事项：
          decompress必须与compress使用同一step（dopt_rsag中应保持一致），
          因为rank_memory[key]在compress时已写入当前rank，
          此处读取的p/q形状就是本轮压缩所用的形状，不会出现不匹配。
          调用方不需要额外传rank，从rank_memory读取即可。
        """
        
        if compressed_data is None:
            return torch.zeros(original_tensor_size)

        if not self.should_compress_shape(original_tensor_size, name=name):
            # 一维tensor compress时直接透传，decompress同样直接返回
            return compressed_data.view(original_tensor_size)

        key = (name, tuple(original_tensor_size))
        p = self.p_memory[key]
        q = self.q_memory[key]

        if factor_kind is None:
            factor_kind = self.factor_kind_for_update(step)

        # 注意：dopt_rsag 的 step() 在 backward 之后自增，因此更新阶段使用的是上一轮发送的因子。
        if factor_kind == 'p':
            # 奇数step：本轮通信的是p，用allreduce结果覆盖p
            with torch.no_grad():
                p.copy_(compressed_data.view(p.shape))
        else:
            # 偶数step：本轮通信的是q，用allreduce结果覆盖q
            with torch.no_grad():
                q.copy_(compressed_data.view(q.shape))

        # 用通信后的平均因子和本地保留因子重构低秩近似梯度。
        # 残差已经在 compress() 中按本地压缩误差更新，这里不再覆盖 residual。
        reconstructed = p @ q.t()

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
        'halfrankk':HalfRankKReducer,
        'acpsgd': HalfRankKReducer,
        }