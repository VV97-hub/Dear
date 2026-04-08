"""本备份是未增加阶段性rank动态变化的备份2026/3/31"""
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
    This is an adapted version of RankKReducer that
    only does one matrix multiplication per iteration
    """
    
    def __init__(self, random_seed=0, device=None, timer=None, rank=2):
        super().__init__(random_seed, device, timer)
        self.rank = rank
        self.p_memory = {}
        self.q_memory = {}
        self.residuals = {}   # [EF] 用于存 residual
        self.last_input = {}   # [LI] 用于存 上一轮输入梯度张量
        self.name = 'halfrankk'
        

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        orthogonalize(vector)
    
    def compress(self, tensor, name=None, step=0,**kwargs):

    # 对单个 tensor 做半步 PowerSGD 压缩，返回 p/q。
    # 交替执行：奇数次调用算 p，偶数次调用算 q。
        if tensor.ndimension() <= 1:
            return tensor, None, None  # 大小就是 numel()，和外部对齐

        if name is None:
            name = 'default'

        matrix = tensor.reshape(tensor.shape[0], -1) # 重塑为二维矩阵：保持第0维（通常是 batch size）不变，把后面所有维度flatten成一列
        n, m = matrix.shape
        rank = min(n, m, self.rank)
        # 同一个 name 的 tensor 会变形，所以不仅要根据name 判断
        key = (name, tuple(tensor.shape))

        # [EF] 初始化 residual
        if key not in self.residuals:
            self.residuals[key] = torch.zeros_like(matrix)

        # compress() 中，在加 residual 之前先保存原始梯度
        original_matrix = matrix.clone()          # ← 加这行
        # [EF] 加 residual（核心）
        matrix = matrix + self.residuals[key]

        # 初始化 p/q memory（按 name 管理，对齐 TopK 的 residuals 方式）
        # 同一个 name 的 tensor 会变形，所以不仅要根据name 判断
        if key not in self.p_memory:
            self.p_memory[key] = torch.zeros(n, rank, device=tensor.device)
            self.q_memory[key] = torch.zeros(m, rank, device=tensor.device)
            self.set_random(self.q_memory[key])  # 首次随机初始化

        p = self.p_memory[key]
        q = self.q_memory[key]
        
        # compress() 处理的是 梯度 tensor 的通信压缩，它 不属于模型前向/反向计算图的一部分。如果不关闭 autograd，PyTorch 可能会：
        # 记录不必要的计算图
        # 增加显存
        # 甚至导致梯度 graph 被污染
        with torch.no_grad():
            if step % 2 == 0:
                orthogonalize(q)                  
                torch.matmul(matrix, q, out=p)
                self.p_memory[key] = p
                # 残差存入 residuals（对齐 EFTopK 的做法）
                # self.residuals[key] = (matrix - p @ q.t()).view(tensor.shape)

                # [EF] 暂存当前输入（用于后面更新 residual）
                self.last_input[key] = original_matrix    # ← 存原始的
                
                return p.clone(), None, None

            else:  # next_operation == "q"
                orthogonalize(p)
                torch.matmul(matrix.t(), p, out=q)
                self.q_memory[key] = q
                # 残差存入 residuals
                # self.residuals[key] = (matrix - p @ q.t()).view(tensor.shape)

                self.last_input[key] = original_matrix    # ← 存原始的
                return q.clone(), None, None
    
    def decompress(self, compressed_data, original_tensor_size, numel,name, step=0):
        """与 PowerSGDCompressor.decompress 接口完全一致"""
        if compressed_data is None:
            return torch.zeros(original_tensor_size)
        # 用通信结果覆盖本轮发送的那个向量，另一个本地保留的向量不变
        # --- 新增判断逻辑 ---
        # 如果 original_tensor_size 只有一维，说明它没被 P/Q 压缩
        # 如果元素太小也不值得压缩。 numel() < 32
        if len(original_tensor_size) <= 1:
            return compressed_data.view(original_tensor_size)
        # ------------------
        key = (name, tuple(original_tensor_size))
        p = self.p_memory[key]
        q = self.q_memory[key]
    
        # 重点：因为在 dopt_rsag 中，step() 已经在 Backward 之后执行了
        # 此时的 step 已经是 N+1。
        # 如果 step=0 (偶数) 压缩了 P，那么在 step=1 (奇数) 时解压的正是 P。
        if step % 2 != 0:
            with torch.no_grad():
                p.copy_(compressed_data.view(p.shape))
        else:
            with torch.no_grad():
                q.copy_(compressed_data.view(q.shape))

        # 重构
        reconstructed = (p @ q.t())

        # [EF] 更新 residual（核心）
        if key in self.residuals:
            with torch.no_grad():
                self.residuals[key] = self.last_input[key] - reconstructed


        return (p @ q.t()).view(original_tensor_size)
        
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
    def __init__(self, **kwargs):
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
        'halfrankk':HalfRankKReducer
        }
