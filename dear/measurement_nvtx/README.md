# 用一份 conf 提交 Helix 测量

日常只修改同目录的 `experiment.conf`。这是 INI 配置：一行一个参数，可写中文注释；等号右边不要加引号，不写 `export`。参数名写错或漏填会在提交前报错。

`[job]` 的 `cpus_per_task` 为可选项：可以删除、注释、留空或填 `auto`，提交时均不传 `--cpus-per-task`。当前集群 `gpu` 分区默认 `DefCpuPerGPU=8`，已通过 `sbatch --test-only` 和历史 `AllocCPUS` 验证，每节点一张 GPU 时默认分配 8 个 CPU。显式填 `1` 就是覆盖默认值，申请每任务 1 个 CPU。已提交作业的 CPU 申请不会随 conf 修改而改变，需要在下次提交时生效。

## 节点选择

在 `[job]` 中填写，两个新选项均可留空、注释或省略：

```ini
[job]
nodes = 2
nodelist = g0014,g0022
exclude = g0017,g0018
```

`nodes` 是节点数量；`nodelist` 对应 `sbatch --nodelist`（`-w`），请求指定节点；`exclude` 对应 `sbatch --exclude`（`-x`），禁止分配这些节点。上例会将 `--nodes=2 --nodelist=g0014,g0022 --exclude=g0017,g0018` 加入提交命令。节点名单也可写成 `g[0014,0022]` 或 `g[0017-0018]`，INI 中不加引号。

如希望固定在列出的节点上运行，请让 `nodes` 与指定节点数量一致。不要把 `nodelist` 当作可任选部分节点的候选池；节点可用性、表达式及与排除名单的冲突由 Slurm 检查。指定节点忙时仍可能排队。只想排除某些节点时，将 `nodelist` 留空；两项均留空时由 Slurm 自动选择。

修改后先执行 `bash submit.sh --dry-run`，在输出末尾核对完整 `sbatch` 命令。节点选择仅影响新提交的作业，不能修改已排队作业的资源申请。MPI hostfile 继续使用 Slurm 最终分配的节点。

## 提交

在服务器的本目录执行：

```bash
# 1. 编辑配置
nano experiment.conf

# 2. 检查最终参数和采集窗口；不会提交任务或申请 GPU
bash submit.sh --dry-run

# 3. 提交。此命令会申请配置中的 GPU 资源
bash submit.sh
```

也可以复制配置，为不同实验分别保存一份：

```bash
cp experiment.conf experiment_p1.local.conf
nano experiment_p1.local.conf
bash submit.sh experiment_p1.local.conf --dry-run
bash submit.sh experiment_p1.local.conf
```

提交成功后会显示 Job ID、日志文件和报告目录。不复制 conf，不生成配置 JSON 或提交命令记录。资源按提交时的配置申请；训练与采集参数在作业启动时从原 conf 读取，排队期间修改 conf 会影响尚未启动的任务。Python 训练开始后，参数不会随 conf 的后续修改动态变化。

仓库中的 `experiment.conf` 是可复用的默认模板。个人实验参数使用 `*.local.conf`，Git 会忽略这类文件；需要长期保存的实验方案可另取明确文件名后有意提交。原始日志、Nsight 报告和本机数据不随启动代码提交。修改本地模板不会自动覆盖服务器上正在使用的配置。

## 参数分别在哪里

| 文件／分组 | 修改内容 |
|---|---|
| `experiment.conf` 的 `[job]` | 节点数、指定/排除节点、GPU、CPU、分区、QoS、时限、输出根目录 |
| `[training]` | 模型、每卡 batch、句长、训练预热、每轮 batch 数、轮数、通信流数和融合阈值 |
| `[compression]` | 压缩器、rank、压缩启用时间、refresh、embedding、动态 rank 策略及其阈值 |
| `[runtime]` | event sync、通信 fp16 开关、NCCL 日志等 |
| `[capture]` | Nsight／NVTX 开关、自动算子标记、采集窗口、trace 类型 |
| `[paths]` | 项目、环境脚本和 Nsight 安装包路径 |
| `env_sczd744.sh` | Conda、CUDA/MPI 库路径、NCCL 网卡、HCA、GID、IB 设置；换环境才修改 |
| `submit.sh`、`submit_experiment.py`、`job.sbatch` | 固定执行逻辑，日常不修改参数 |
| `bootstrap_nsys.sh` | 每个计算节点解压当前 2026.5.1 Nsight CLI；换安装包版本时检查此文件 |
| `profile_rank.sh` | 固定的 Nsight 捕获方式和报告命名；trace/sample 来自 conf |
| `bert_benchmark.py`、`helix_capture.py` | 训练与捕获实现；更改实现时才修改 |

新链路：`submit.sh → submit_experiment.py → sbatch job.sbatch → 环境/每节点 CLI → MPI → 每个 rank 的 Nsight/Python`。

**此入口不经过旧的 `submission_command.txt`、`measurement_runtime/submit_smoke.sbatch`、`submit_nvtx.sbatch`、`run_mpi_nvtx.sh` 或 `horovod_mpi_cj.sh`。**这些旧入口保留用于历史复现；修改它们不会覆盖新 conf。

## 默认值来源与窗口

训练默认值于 2026-11-26 通过 SSH 核对服务器 `/data/home/sczd744/run/dear_pytorch-master/dear/horovod_mpi_cj.sh` 及 `bert_benchmark.py`，为 4 rank、每卡 batch 64、句长 64、10 步训练预热、每轮 10 batch、1000 轮、`compress_warmup=5000`、`rank_schedule=update_norm_stable`。rank 初始为 16，可按稳定性选择 16/12/8，不能把整个窗口假定成固定 rank 16。

CPU 数默认不指定，沿用集群的 GPU 配套 CPU 分配。时限 2 小时和采集窗口是本测量入口的申请/测量设置，不是 Python 原默认值。2 小时为申请上限，不保证训练一定在此时限内完成。

全局 step 从 0 开始：step 0 为初始调用，step 1--10 为训练预热，`Iter #0` 为 step 11--20。

```text
Iter #i 的首步 = 1 + num_warmup_batches + i * num_batches_per_iter
总训练调用 = 1 + num_warmup_batches + num_iters * num_batches_per_iter
```

默认核心采集 step 5211--5250，对应 `Iter #520--523`，再记录尾步 5251 来消费源 step 5250 的延迟更新。分析脚本依据核心 `source_step` 筛选；报告还包含边界/尾步活动。该窗口晚于压缩开始的 step 5001；仍需用时间线检查实际稳定性。

窗口选择不改变训练长度。采集结束后训练继续执行剩余轮次；若只想运行到该窗口附近，应显式缩短 `num_iters`，这属于训练参数变更。

## 开关

算子分析使用：

```ini
[capture]
nsys_enabled = 1
nvtx_enabled = 1
autograd = 1
```

只想保留手工阶段标记时，将 `autograd` 改为 0。无采集吞吐对照请将以上三项都设为 0，此时不解压 CLI、不启动 Nsight，直接启动 Python。单独设置 `nsys_enabled=0` 不会自动关闭手工 NVTX 标记；自动算子 profiler 则只在 Nsight 窗口中启用。

Nsight 开始/结束和自动算子标记会引入边界开销；带采集的控制台吞吐统计不能直接当作无采集吞吐。

## CPU 热点

在 `[capture]` 设置：

```ini
trace = cuda,nvtx,osrt
sample = process-tree
```

CPU 指令位置/调用栈采样范围为每个 rank 的进程及其子进程，采集窗口仍使用 `start_step/steps/tail_steps`。`sample=none` 关闭此类采样，但保留 CUDA、NVTX、OSRT。新入口不开放与本实验无关的整机采样 `system-wide`。Nsight 关闭时不会启动 CPU 采样。

每个 rank 会先在实际计算节点执行 `nsys status --environment`。如诊断明确显示 `CPU Profiling Environment (process-tree): Fail`，立即停止启动并给出原因。普通用户的原生 CPU 采样通常需要 `perf_event_paranoid<=2` 且 `perf_event_open` 可用；由管理员评估设置或授予适当权限，不在提交脚本中修改系统参数。

2026-11-26 检查登录节点 ln01：`perf_event_paranoid=4`、`perf_event_open: Fail`、process-tree 采样检查失败。计算节点诊断未获资源，尚不能确认计算节点的权限。不能把仅配置为 process-tree 当作已经采到了 CPU 数据。

采集成功后，在 Nsight GUI 打开 rank 报告，选中目标时间段/线程并筛选采样，查看 CPU Sampling 的 Bottom-Up、Top-Down 或 Flat 函数表。按自身样本数/占比查找热点，再用 Top-Down 查看调用路径；线程等待结合 OSRT 时间线分析。原生采样不等于 Python 函数级统计：当前 py38 环境不能使用 Nsight 要求 Python 3.9+ 的 `--python-sampling` 功能。

CPU 热点建议使用比两个 step 更长的窗口（例如 20--40 步），以获得足够样本；CPU 采样也会增加测量开销。纯吞吐对照仍应关闭 Nsight/NVTX。

## 结果

每次提交自动创建独立目录，默认在项目的 `dear/logs/nvtx/<时间_实验名_随机后缀>/`：

```text
job_<JobID>.out           作业日志
reports/                 每个 rank 的 nsys 报告及 hostfile
```

每个节点的 CLI 解压发生在作业内，节点数读取实际 `SLURM_NNODES`，不再写死为两个节点。新入口只支持每节点一个 rank、一张 GPU，MPI rank 数从节点数推导，无需再单独维护 `NWORKERS`。

本地修改需同步到服务器后再提交；修改本地归档 TXT 不会改变服务器任务。学习率 `5e-5`、AdamW `eps=1e-6` 仍来自现有训练源码，本 conf 没有伪造不可生效的学习率参数。
