"""本备份是一轮传p一轮传q的powersgd的备份2026/4/2，下一步打算试试一轮内同时传p和q的powersgd"""
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
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
    采用 nested subspace: 每个张量持久维护最大rank的P/Q母空间，
    当前rank只使用前r列，rank变化时不重建P/Q和residual。
    """

    def __init__(self, random_seed=0, device=None, timer=None, rank=2,
                 rank_schedule=None, warmup_steps=200,
                 min_compression_numel=16384, rank_overrides=None,
                 update_norm_stable_rank=False,
                 stable_rank_levels=None,
                 update_norm_stable_tol=0.01, update_norm_critical_tol=0.3,
                 update_norm_patience=100, update_norm_smoothing=0.9,
                 update_norm_debug_every=0,
                 rank_reset_on_change=False,
                 embedding_policy='word'):
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
          rank_overrides: 按参数名覆盖rank，格式为"name=rank,name=rank"。
          update_norm_stable_rank:
                         True时使用同步后update norm的时间稳定性做阶梯式降rank。
          rank_reset_on_change:
                         True时rank变化后重建该tensor的P/Q/residual；False保留nested subspace。
          embedding_policy:
                         off跳过所有embedding相关参数；word只压缩word embedding/decoder；
                         broad允许embedding参数按通用规则参与压缩。
        """
        super().__init__(random_seed, device, timer)

        # 默认rank（兜底值）
        self.default_rank = rank

        # 动态rank调度器，None表示退化为固定rank
        self.rank_schedule = rank_schedule
        self._rank_schedule_items = None
        self._rank_schedule_cache = {}
        self._rank_schedule_max_rank = self.default_rank
        if isinstance(rank_schedule, dict):
            self._rank_schedule_items = sorted(
                (int(step), int(value)) for step, value in rank_schedule.items()
            )
            if self._rank_schedule_items:
                self._rank_schedule_max_rank = max(value for _, value in self._rank_schedule_items)

        self.rank_overrides = {}
        if rank_overrides:
            for item in str(rank_overrides).split(','):
                name, value = item.split('=', 1)
                self.rank_overrides[name.strip().lower()] = int(value)

        # ---- 每个tensor key的内存 ----
        # Nested subspace: P/Q 按该张量可能出现的最大rank分配，当前rank只使用前r列。
        self.p_memory = {}          # {key: Tensor(n, max_rank)}
        self.q_memory = {}          # {key: Tensor(m, max_rank)}
        self.residuals = {}         # {key: Tensor(n, m)}  Error Feedback残差

        # 记录每个key上次使用的rank，用于检测rank是否发生变化
        # 结构：{key: int}
        # compress写入，decompress读取，保证二者使用完全相同的active rank
        self.rank_memory = {}
        self.max_rank_memory = {}
        self.update_norm_stable_rank = bool(update_norm_stable_rank)
        self.stable_rank_levels = self._build_stable_rank_levels(stable_rank_levels, rank)
        self.update_norm_stable_tol = float(update_norm_stable_tol)
        self.update_norm_critical_tol = float(update_norm_critical_tol)
        self.update_norm_patience = max(1, int(update_norm_patience))
        self.update_norm_smoothing = float(update_norm_smoothing)
        self.update_norm_debug_every = max(0, int(update_norm_debug_every))
        self.global_rank_level_idx = 0
        self.pending_global_rank_level = None
        self.global_update_norm_ema = None
        self.global_update_norm_last = None
        self.global_update_norm_stable_count = 0
        self._global_update_norm_observation = None
        self._debug_print_pq_shapes = os.environ.get('DEAR_PRINT_PQ_SHAPES', '0') == '1' # 输出P和Q的维度大小
        self._debug_printed_pq_shapes = set()
        self._rank_change_debug = os.environ.get('DEAR_RANK_CHANGE_DEBUG', '0') == '1'
        self.rank_reset_on_change = bool(rank_reset_on_change)
        self.embedding_policy = str(
            os.environ.get('DEAR_EMBEDDING_POLICY', embedding_policy)
        ).strip().lower()
        if self.embedding_policy not in ('off', 'word', 'broad'):
            raise ValueError(
                "embedding_policy must be one of: off, word, broad; got %r"
                % self.embedding_policy
            )

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

    def _build_stable_rank_levels(self, levels, max_rank):
        if levels is None or str(levels).strip() == '':
            raw_levels = [max_rank, round(max_rank * 0.75), max_rank // 2]
        elif isinstance(levels, str):
            raw_levels = [int(x.strip()) for x in levels.split(',') if x.strip()]
        else:
            raw_levels = [int(x) for x in levels]

        cleaned = []
        for level in raw_levels:
            level = max(1, min(int(max_rank), int(level)))
            if level not in cleaned:
                cleaned.append(level)
        if not cleaned:
            cleaned = [max(1, int(max_rank))]
        return cleaned

    def _should_skip_by_name(self, name):
        if not name:
            return False
        lower_name = name.lower()
        if lower_name.endswith("bias") or ".bias" in lower_name:
            return True
        if "embedding" in lower_name or "embeddings" in lower_name:
            if self.embedding_policy == 'off':
                return True
            if self.embedding_policy == 'word':
                allowed_embedding_names = {
                    "bert.embeddings.word_embeddings.weight",
                    "cls.predictions.decoder.weight",
                }
                return lower_name not in allowed_embedding_names
        if (
            self.embedding_policy == 'off'
            and lower_name == "cls.predictions.decoder.weight"
        ):
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
    def _scheduled_base_rank(self, step, name=None):
        if name is not None and name.lower() in self.rank_overrides:
            return self.rank_overrides[name.lower()]
        if self.rank_schedule is None:
            # 未配置调度器，使用固定rank（兼容旧接口）
            return self.default_rank
        if callable(self.rank_schedule):
            # 函数式调度：完全由外部逻辑控制
            return self.rank_schedule(step)

        # dict调度：预排序后线性扫描少量阈值，并缓存同一step的结果。
        # aggressive/gentle这类schedule在一个step内会被多个tensor反复查询。
        step = int(step)
        cached = self._rank_schedule_cache.get(step)
        if cached is not None:
            return cached
        base_rank = self.default_rank
        schedule_items = self._rank_schedule_items
        if schedule_items is None:
            schedule_items = sorted(
                (int(s), int(r)) for s, r in self.rank_schedule.items()
            )
            self._rank_schedule_items = schedule_items
            if schedule_items:
                self._rank_schedule_max_rank = max(r for _, r in schedule_items)
        for threshold, rank_value in schedule_items:
            if threshold <= step:
                base_rank = rank_value
            else:
                break
        self._rank_schedule_cache[step] = base_rank
        return base_rank

    def _stable_base_rank(self, step, n, m, name=None, shape=None):
        if name is not None and name.lower() in self.rank_overrides:
            return self.rank_overrides[name.lower()]
        pending = self.pending_global_rank_level
        if pending is not None:
            pending_level_idx, apply_step = pending
            if step >= apply_step:
                self.global_rank_level_idx = pending_level_idx
                self.pending_global_rank_level = None
        level_idx = self.global_rank_level_idx
        level_idx = max(0, min(level_idx, len(self.stable_rank_levels) - 1))
        return self.stable_rank_levels[level_idx]

    def _get_rank(self, step, n, m, name=None, shape=None, device=None):
        """
        计算当前step应使用的rank值。

        规则：
          1. 如果参数名命中rank_overrides，优先使用覆盖rank；
          2. 否则如果提供了rank_schedule，按调度器确定基础rank；
          3. 再用 min(n, m, base_rank) 保证rank不超过矩阵维度上限；
          4. 保证最小值为1，防止rank退化为0导致空矩阵。

        参数：
          step : 当前训练步数（外部传入，从dopt_rsag的step获取）
          n, m : 当前tensor reshape后的矩阵维度
        """
        if self.update_norm_stable_rank:
            base_rank = self._stable_base_rank(step, n, m, name=name, shape=shape)
        else:
            base_rank = self._scheduled_base_rank(step, name=name)

        # TODO 未来可在这里加入按 tensor 大小动态调整 rank 的逻辑
        """ 
        numel = n * m
        if numel > 1_000_000:
            base_rank = min(base_rank + 2, 8)   # 大矩阵适当提高
        elif numel < 10_000:
            base_rank = max(base_rank - 1, 1)   # 小矩阵适当降低

        # rank不能超过矩阵的任意一维，也不能为0
        """

        max_allowed_rank = self.max_rank_for(name)
        upper_rank = max(1, min(n, m, max_allowed_rank))
        return min(upper_rank, max(1, min(upper_rank, base_rank)))
        

    # ------------------------------------------------------------------
    # 工具方法：记录rank变化。Nested subspace下不清除P/Q/residual。
    # ------------------------------------------------------------------
    def _maybe_reset_memory(self, key, new_rank, step=None, name=None):
        """
        Nested subspace使用固定最大rank母空间，当前rank只是active列数。
        因此rank变化时不能清除P/Q和residual，否则会破坏子空间连续性和EF累积。
        """
        old_rank = self.rank_memory.get(key, None)
        try:
            worker_rank = torch.distributed.get_rank()
        except Exception:
            worker_rank = 0
        if self._rank_change_debug and old_rank != new_rank and worker_rank == 0:
            label = name if name is not None else key[0]
            print(
                "[DynamicRank] step=%s name=%s rank=%s -> %s"
                % (step, label, old_rank, new_rank),
                flush=True,
            )
        if (
            self.rank_reset_on_change
            and old_rank is not None
            and old_rank != new_rank
        ):
            self.p_memory.pop(key, None)
            self.q_memory.pop(key, None)
            self.residuals.pop(key, None)
            self.max_rank_memory.pop(key, None)

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
            base_rank = self.default_rank
        
        # TODO 这里逻辑有问题，为什么不按照轮次来选择：
        elif callable(self.rank_schedule):
            # 函数式调度无法静态分析，由外部保证 default_rank >= 实际最大rank
            base_rank = self.default_rank
        
        # TODO 这里逻辑有问题，为什么不按照轮次来选择：
        else:
            # dict 调度：取所有阶段 rank 的最大值，保证 buffer 足够大
            base_rank = self._rank_schedule_max_rank

        if self.rank_overrides:
            return max(base_rank, max(self.rank_overrides.values()))
        return base_rank

    def max_rank_for(self, name=None):
        if name is not None and name.lower() in self.rank_overrides:
            return self.rank_overrides[name.lower()]
        if self.rank_schedule is None or callable(self.rank_schedule):
            return self.default_rank
        return self._rank_schedule_max_rank

    def layout_cache_key(self, step):
        """
        Return a compact key for active-prefix layout reuse.
        The communication layout depends on the effective global rank level, not
        on the absolute step number.  For staged schedules this avoids rebuilding
        identical offsets on every compressed step.
        """
        if self.update_norm_stable_rank:
            # Apply pending level changes consistently with _get_rank().
            self._stable_base_rank(step, 1, 1)
            return ('stable', int(self.global_rank_level_idx))
        if self.rank_schedule is None:
            return ('fixed', int(self.default_rank))
        if callable(self.rank_schedule):
            # Keep callable schedules conservative; arbitrary callables may not
            # be pure functions of rank level.
            return ('callable', int(step))
        return ('scheduled', int(self._scheduled_base_rank(step)))

    def wants_update_norm_observation(self):
        if not self.update_norm_stable_rank:
            return False
        if self.pending_global_rank_level is not None:
            return True
        return self.global_rank_level_idx < len(self.stable_rank_levels) - 1

    def begin_global_update_norm_observation(self, step, num_groups):
        """
        为当前step的同步后update norm开启一次全局稳定性观测。
        dopt_rsag在每个all-gather group解压并除以worker数后追加该group的norm。
        """
        if not self.update_norm_stable_rank:
            return
        self._global_update_norm_observation = {
            'step': int(step),
            'num_groups': int(num_groups),
            'seen_groups': set(),
            'norm_sq_terms': [],
        }

    def observe_global_update_norm_group(self, norm_sq, step, group_idx):
        """
        记录一个已经解压完成的同步update group的L2 norm平方。
        所有group收齐后，用update norm的时间稳定性决定下一轮统一rank。
        """
        if not self.update_norm_stable_rank or norm_sq is None:
            return
        obs = self._global_update_norm_observation
        if (
            obs is None
            or obs['step'] != int(step)
            or group_idx in obs['seen_groups']
        ):
            return

        obs['seen_groups'].add(group_idx)
        obs['norm_sq_terms'].append(norm_sq.detach().float())
        if len(obs['seen_groups']) >= obs['num_groups']:
            self._finish_global_update_norm_observation(obs)

    def _finish_global_update_norm_observation(self, obs):
        if not obs['norm_sq_terms']:
            return
        norm_sq = torch.stack(obs['norm_sq_terms']).sum()
        norm_value = float(torch.sqrt(norm_sq).item())

        old_ema = self.global_update_norm_ema
        old_norm = self.global_update_norm_last
        if old_ema is None:
            self.global_update_norm_ema = norm_value
            self.global_update_norm_last = norm_value
            self._global_update_norm_observation = None
            return

        alpha = self.update_norm_smoothing
        new_ema = alpha * old_ema + (1.0 - alpha) * norm_value
        raw_change = abs(norm_value - old_norm) / (abs(old_norm) + 1e-12)
        ema_change = abs(new_ema - old_ema) / (abs(old_ema) + 1e-12)
        self.global_update_norm_ema = new_ema
        self.global_update_norm_last = norm_value

        if raw_change >= self.update_norm_critical_tol:
            stable_count = 0
        elif ema_change <= self.update_norm_stable_tol:
            stable_count = self.global_update_norm_stable_count + 1
        else:
            stable_count = 0
        self.global_update_norm_stable_count = stable_count

        self._debug_global_update_norm_observation(
            obs, norm_value, new_ema, raw_change, ema_change, stable_count
        )

        level_idx = self.global_rank_level_idx
        if stable_count < self.update_norm_patience or level_idx >= len(self.stable_rank_levels) - 1:
            self._global_update_norm_observation = None
            return

        old_rank = self.stable_rank_levels[level_idx]
        new_level_idx = level_idx + 1
        new_rank = self.stable_rank_levels[new_level_idx]
        apply_step = int(obs['step']) + 1
        self.pending_global_rank_level = (new_level_idx, apply_step)
        self.global_update_norm_ema = None
        self.global_update_norm_last = None
        self.global_update_norm_stable_count = 0
        self._global_update_norm_observation = None
        try:
            worker_rank = torch.distributed.get_rank()
        except Exception:
            worker_rank = 0
        if self._rank_change_debug and worker_rank == 0:
            print(
                "[StableUpdateNormRank] step=%s global_rank=%s -> %s pending_from_step=%s stable_count=%s raw_change=%.6f ema_change=%.6f"
                % (obs['step'], old_rank, new_rank, apply_step, stable_count, raw_change, ema_change),
                flush=True,
            )

    def _debug_global_update_norm_observation(self, obs, norm_value, ema, raw_change, ema_change, stable_count):
        if self.update_norm_debug_every <= 0:
            return
        step = int(obs['step'])
        if step % self.update_norm_debug_every != 0:
            return
        try:
            worker_rank = torch.distributed.get_rank()
        except Exception:
            worker_rank = 0
        if worker_rank != 0:
            return
        current_rank = self.stable_rank_levels[self.global_rank_level_idx]
        print(
            "[StableUpdateNormDebug] step=%s norm_value=%.6e ema=%.6e raw_change=%.6f ema_change=%.6f stable_count=%s current_rank=%s"
            % (step, norm_value, ema, raw_change, ema_change, stable_count, current_rank),
            flush=True,
        )

    def get_rank_for_step(self, name, shape, step):
        if not self.should_compress_shape(shape, name=name):
            return None
        n = int(shape[0])
        m = 1
        for dim in shape[1:]:
            m *= int(dim)
        return self._get_rank(step, n, m, name=name, shape=shape, device=self.device)

    def get_rank_for(self, name, shape):
        """
        供外部（dopt_rsag）查询某个tensor当前实际使用的rank。
        封装 key 的构造方式，调用方不需要知道内部 key 格式，
        保证与 compress 端的 key 定义严格一致。
        """
        key = (name, tuple(shape))
        return self.rank_memory.get(key, self.max_rank_for(name))

    def get_factor_numel_for_step(self, shape, name=None, factor_kind='p', step=0):
        rank = self.get_rank_for_step(name, shape, step)
        return self.get_factor_numel(shape, name=name, factor_kind=factor_kind, rank=rank)

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

    def _orthogonalize_new_columns(self, factor, old_rank, new_rank):
        if new_rank <= old_rank:
            return
        self._orthogonalize_factor(factor[:, :new_rank])

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        self._orthogonalize_factor(vector)

    def _prepare_tensor_memory(self, tensor, name=None, step=0):
        if name is None:
            name = 'default'
        grad_matrix = tensor.reshape(tensor.shape[0], -1)
        n, m = grad_matrix.shape
        rank = self._get_rank(
            step, n, m, name=name, shape=tensor.shape, device=tensor.device
        )
        max_rank = max(1, min(n, m, self.max_rank_for(name)))
        key = (name, tuple(tensor.shape))
        old_rank = self.rank_memory.get(key, None)
        self._maybe_reset_memory(key, rank, step=step, name=name)
        if key not in self.residuals:
            self.residuals[key] = torch.zeros_like(grad_matrix)
        if key not in self.p_memory:
            self.p_memory[key] = torch.zeros(n, max_rank, device=tensor.device)
            self.q_memory[key] = torch.zeros(m, max_rank, device=tensor.device)
            self.max_rank_memory[key] = max_rank
            self.set_random(self.p_memory[key])
            self.set_random(self.q_memory[key])
        elif self.max_rank_memory.get(key) != max_rank:
            raise RuntimeError(
                'Nested rank max changed for %s: old=%s new=%s. '
                'Increase default max rank before training instead of changing it at runtime.'
                % (name, self.max_rank_memory.get(key), max_rank)
            )
        if old_rank is not None and rank > old_rank:
            self._orthogonalize_new_columns(self.p_memory[key], old_rank, rank)
            self._orthogonalize_new_columns(self.q_memory[key], old_rank, rank)
        p = self.p_memory[key][:, :rank]
        q = self.q_memory[key][:, :rank]
        return grad_matrix, key, rank, p, q

    def _debug_log_pq_shapes(self, name, tensor_shape, n, m, rank, step, factor_kind):
        if not self._debug_print_pq_shapes:
            return
        worker_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('RANK', '0')))
        if worker_rank != 0:
            return
        key = (name, tuple(tensor_shape), int(rank), factor_kind)
        if key in self._debug_printed_pq_shapes:
            return
        self._debug_printed_pq_shapes.add(key)
        print(
            "[HalfRankK][P/Q shape] step=%s factor=%s name=%s "
            "G=%s matrix=(%d,%d) rank=%d P=(%d,%d) Q=(%d,%d)"
            % (
                step,
                factor_kind,
                name,
                tuple(tensor_shape),
                n,
                m,
                rank,
                n,
                rank,
                m,
                rank,
            ),
            flush=True,
        )

    def compute_factor(self, tensor, name=None, step=0, factor_kind=None, update_residual=True):
        if not self.should_compress_tensor(tensor, name=name):
            return tensor.contiguous().clone()
        if factor_kind is None:
            factor_kind = self.factor_kind(step)
        grad_matrix, key, rank, p, q = self._prepare_tensor_memory(tensor, name, step)
        matrix = grad_matrix + self.residuals[key]
        self._debug_log_pq_shapes(
            name,
            tensor.shape,
            grad_matrix.shape[0],
            grad_matrix.shape[1],
            rank,
            step,
            factor_kind,
        )
        with torch.no_grad():
            if factor_kind == 'p':
                self._orthogonalize_factor(q)
                p.copy_(torch.matmul(matrix, q))
                compressed = p.contiguous().clone()
            else:
                self._orthogonalize_factor(p)
                q.copy_(torch.matmul(matrix.t(), p))
                compressed = q.contiguous().clone()
            if update_residual:
                self.residuals[key].copy_(matrix - p @ q.t())
        return compressed

    def load_factor(self, compressed_data, original_tensor_size, name, step=0, factor_kind=None):
        if not self.should_compress_shape(original_tensor_size, name=name):
            return
        key = (name, tuple(original_tensor_size))
        rank = self.rank_memory.get(key, self.max_rank_for(name))
        p = self.p_memory[key][:, :rank]
        q = self.q_memory[key][:, :rank]
        if factor_kind is None:
            factor_kind = self.factor_kind_for_update(step)
        with torch.no_grad():
            if factor_kind == 'p':
                p.copy_(compressed_data.view(p.shape))
            else:
                q.copy_(compressed_data.view(q.shape))

    def reconstruct_from_memory(self, original_tensor_size, name):
        if not self.should_compress_shape(original_tensor_size, name=name):
            return None
        key = (name, tuple(original_tensor_size))
        rank = self.rank_memory.get(key, self.max_rank_for(name))
        p = self.p_memory[key][:, :rank]
        q = self.q_memory[key][:, :rank]
        return (p @ q.t()).view(original_tensor_size)

    def update_residual_from_memory(self, tensor, name):
        if not self.should_compress_tensor(tensor, name=name):
            return
        key = (name, tuple(tensor.shape))
        if key not in self.residuals:
            return
        grad_matrix = tensor.reshape(tensor.shape[0], -1)
        rank = self.rank_memory.get(key, self.max_rank_for(name))
        p = self.p_memory[key][:, :rank]
        q = self.q_memory[key][:, :rank]
        with torch.no_grad():
            self.residuals[key].copy_(grad_matrix + self.residuals[key] - p @ q.t())

    def compress(self, tensor, name=None, step=0, **kwargs):
        """
        对单个tensor执行半步PowerSGD压缩（奇偶步交替算p/q）。

        动态rank逻辑：
          - 每次调用先通过_get_rank计算本轮rank；
          - P/Q按max_rank持久保存，当前rank只使用前r列；
          - rank变化不清空residual，从而保留Error Feedback语义。
        """
        if not self.should_compress_tensor(tensor, name=name):
            # 一维tensor不压缩（如bias），直接透传
            return tensor, None, None

        return self.compute_factor(
            tensor,
            name=name,
            step=step,
            factor_kind=self.factor_kind(step),
            update_residual=True,
        ), None, None

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

        if factor_kind is None:
            factor_kind = self.factor_kind_for_update(step)

        self.load_factor(compressed_data, original_tensor_size, name, step, factor_kind)
        return self.reconstruct_from_memory(original_tensor_size, name)
        
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
