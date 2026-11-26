# Helix NVTX 测量代码与使用说明

适用范围：当前 `Dear/dear/bert_benchmark.py` + 分布式 `halfrankk` 路径。
本版本修复了原草稿的导入顺序错误，补齐自动 BP 算子标记、分阶段标记、
连续采集窗口、MPI/rank/Slurm 启动接线及基于 CUDA launch 关联的统计脚本。
本代码维护在 GitHub `扩刊-2026-11-24` 测量分支，算法开发与正式训练维护在 main。
开发分支、测量分支及同步方式见 [分支说明](../HELIX_BRANCH_GUIDE.md)。

2026-11-25：已合并 main 的服务器适配提交 `93bdad7`，包括通信初始化前的本地 MPI
rank/GPU 绑定、保留已配置的网络接口、NCCL/UCX 启动参数及编译脚本注释。
NVTX 测量代码以该运行基线为对照。

2026-11-26：旧启动链路的两 rank 作业 `1405471` 已完成 GPU 冒烟采集，BP、
PowerSGD 编码/解码和 NCCL kernel 可归类；其窗口包含训练预热，不能作为稳态性能数据。
新增集中配置入口的提交为 `ae04ef8`，推荐从
[measurement_nvtx/README.md](measurement_nvtx/README.md) 使用 `experiment.conf` 和 `submit.sh`。
该入口已通过 CPU 回归、Linux 启动集成测试、参数预览及 Slurm `--test-only`；
这些检查不等于新入口已完成真实训练采集。CPU 热点采样仍需实际计算节点具备 perf 权限。

## 1. 入口文件

| 文件 | 用途 |
|---|---|
| `measurement_nvtx/experiment.conf`、`submit.sh` | 当前推荐入口：集中配置、预览和 Slurm 提交 |
| `measurement_nvtx/submit_experiment.py`、`job.sbatch` | 参数检查、资源申请、每节点 CLI 准备和 MPI 启动 |
| `helix_nvtx.py` | 可关闭的手工 NVTX 标记、线程内元信息、autograd hook 标记 |
| `helix_capture.py` | 采集窗口及一次性的 autograd/ATen 自动标记控制 |
| `measurement_nvtx/run_mpi_nvtx.sh` | 历史 MPI 入口；保留原调用方式与默认参数 |
| `measurement_nvtx/profile_rank.sh` | 每个 MPI worker 内启动 nsys，报告名包含 job 和 rank |
| `measurement_nvtx/submit_nvtx.sbatch` | 历史两节点 Slurm 入口；新 conf 链路不经过它 |
| `measurement_nvtx/env_sczd744.sh` | 原 sczd744 软件/互连环境配置；新机器需适配 |
| `measurement_nvtx/analyze_nvtx.py` | SQLite 中的 kernel 归属、算子汇总和区间重叠统计 |

没有修改 C++/CUDA 扩展；单纯应用本次 Python 标记不需要重编译。
前提是原环境中已有与成熟版兼容、支持 event handoff 的 `comm_core`。
请同时传输这四个被修改的文件：`bert_benchmark.py`、`compression.py`、
`dopt_rsag.py`、`tensorfusion.py`，以及上述两个 helper 和 measurement_nvtx 目录。
不要只复制启动脚本，也不要仅 clone 尚未包含本次修改的云端 main。

## 2. 三类算子及额外开销的边界

| 类别 | 实际工作 | 标签 |
|---|---|---|
| 纯 BP | BERT 线性/注意力矩阵乘的反向、Softmax/GELU/LayerNorm/Dropout 反向、Embedding 梯度累加、loss 反向 | 自动 autograd/ATen 标签 + `backward`，排除 hook 中的编解码/打包/通信 |
| 编码 | 梯度加残差；缓存互补因子正交化；P: M Q 或 Q: M^T P；因子 copy；本地 PQ^T 重构、减法和残差写回 | `codec_encode`、`compress_param`、`factor_state`、`add_residual`、`orthogonalize`、`project_p/q`、`factor_copy`、`ef_update` |
| 解码 | 从通信结果写回 P/Q；PQ^T 重构梯度 | `codec_decode`、`load_factor`、`factor_load_copy`、`reconstruct`、`reconstruct_matmul` |
| NCCL | 普通路径 Reduce-Scatter / All-Gather；可选 refresh 的两轮 AllReduce | `rs_submit`、`ag_submit`、`refresh_allreduce_first/second` |
| 打包/布局 | 因子融合、补零、active-prefix 布局；dense 旁路的 copy | `pack`、`padding`、`layout_prepare`、`dense_pack/unpack`、`refresh_pack` |
| 优化器/控制 | 除 worker 数、AdamW/SGD 更新、范数归约、rank 决策及原有标量回读 | `gradient_average`、`optimizer_update`、`norm_reduce`、`norm_decision`、`norm_host_read` |
| 依赖/CPU 等待 | 生产流→RS、RS→AG、AG→计算；CPU 同步回退 | `producer_wait_enqueue`、`producer_host_sync`、`rs_ag_wait_enqueue`、`ag_compute_wait_enqueue`、`comm_host_sync` |

`ef_update` 包含本地重构 GEMM、减法及 copy。正常压缩路径一般有投影、本地 EF 重构、
通信后解码三次矩阵乘，不能把编码成本简化成一次 GEMM。
只做 view/切片的表达式通常不产生 CUDA kernel；是否生成 copy 以 trace 为准。
`ag_schedule` 是外层调度范围，`ag_submit` 是实际提交范围，避免原草稿的同名嵌套。
refresh 保留专门标签，但第一轮归因建议 `--compress-refresh-k 0`。

标签可包含 worker、来源 source_step、消费 consume_step、group_idx、参数名、P/Q、
实际低秩 rank、payload 元素数及 bytes。RS bytes 是本地 padded 输入大小；AG bytes 是
本地 shard 输入大小，不是链路上真实传输字节数。
`step`/`runtime_step` 不能替代 source_step：普通解码和更新消费上一轮梯度。
当前 BERT 的 benchmark step 与优化器计数都从 0 开始，每次调用各递增一次。

## 3. 采集窗口

必须在 Python 导入前设置 `HELIX_NVTX=1`。新增参数如下：

| CLI 参数 | 环境默认值 | 含义 |
|---|---|---|
| `--nvtx-capture-start` | `HELIX_CAPTURE_START=40` | 从零开始的 benchmark_step 调用序号，包括独立首轮和 warmup |
| `--nvtx-capture-steps` | `HELIX_CAPTURE_STEPS=0` | 核心来源 step 数；0 关闭窗口控制 |
| `--nvtx-capture-tail` | `HELIX_CAPTURE_TAIL=1` | 多采的普通训练调用数，至少 1，用于消费末轮延迟更新 |
| `--nvtx-autograd` | `HELIX_NVTX_AUTOGRAD=0` | 是否在整个窗口内一次性启用自动 autograd/ATen 标记 |

MPI 测量启动脚本默认设置：start=40、steps=40、tail=1、autograd=1。
核心来源窗口是 [40,80)，额外采 step 80；其 forward 消费 source step 79。
首轮 forward 可能消费 source 39，尾轮 BP 又产生 source 80，统计时按 source_step 排除。
这没有插入额外优化器调用或改变训练数学，只是把原本就会执行的后续调用纳入采集。

Python 在窗口开始调用 `torch.cuda.profiler.start()`，最后调用 stop；
与 nsys 的 `--capture-range=cudaProfilerApi` 配套使用，缺任一侧会得到错误/空窗口。
自动 `emit_nvtx` 在整个窗口只进入/退出一次；其版本实现可能在边界同步。
关闭自动标记时，窗口末尾做一次 GPU drain 后停止。没有逐步/逐组新增 synchronize。
应舍弃采集边界附近周期，不把带 profiler 的 timeit 输出作为论文吞吐量。
如果窗口超出调用总数、未到压缩阶段、启用了旧 overlap 测量或 loss.item() 日志，代码会拒绝。
调用总数为 `1 + num_warmup_batches + num_iters * num_batches_per_iter`，不是 num_iters。

## 4. 在原 sczd744 集群提交

下面是历史服务器路径示例；本地已经生成代码，服务器当前可用性及路径未在本轮验证。
先把本次修改同步到同一个项目根目录，确认数据、tokenizer 和环境仍在。

```bash
export HELIX_PROJECT_DIR=/data/home/sczd744/run/dear_pytorch-master
export HELIX_ENV_SCRIPT=$HELIX_PROJECT_DIR/dear/measurement_nvtx/env_sczd744.sh
export NSYS_BIN=/data/home/sczd744/tools/nsight-systems-2026.5.1/target-linux-x64/nsys
sbatch --export=ALL "$HELIX_PROJECT_DIR/dear/measurement_nvtx/submit_nvtx.sbatch"
```

默认两节点各一张 GPU，固定 rank=16、refresh=0、压缩 warmup=20、训练 warmup=30，
每个计时 iter 一次 benchmark_step，60 个 iter；共有 91 次调用。
以上配置用于阶段诊断，不声称完全复现论文全部性能设置。
Slurm 资源/分区/QoS、新机器 Conda/CUDA/MPI/NCCL/接口都应按实际环境修改。
`env_sczd744.sh` 仅包含原环境的公开路径与互连配置，没有复制历史脚本中的代理凭据。

已在分配好的计算资源上加载原环境时，也可直接使用 MPI 入口：

```bash
export NSYS_BIN=/path/to/nsys
export HELIX_REPORT_DIR=/shared/path/helix_nvtx_run1
export HELIX_HOSTFILE=/path/to/allocated_hostfile
export NWORKERS=2
bash measurement_nvtx/run_mpi_nvtx.sh -- --compress-rank 16 --rank-schedule fixed
```

入口会切换到 dear 目录。`--` 前是额外 MPI 参数，之后是 BERT 参数，使用数组保留空字符串。
至少两个 worker；当前 BERT 单 worker 会绕过分布式编解码和 NCCL，不能验证这三类工作。
默认每个 rank 都采集；报告文件名为 `helix_<job>_rank<rank>.nsys-rep`。
原有 MPI 方式也可以保留，但必须同时做到：

1. 导出并通过 MPI `-x` 转发 HELIX_NVTX、四个窗口/自动标记开关以及 NSYS_BIN/输出目录。
2. 把每个 worker 的 `python ...` 包装成 `bash measurement_nvtx/profile_rank.sh python ...`。
3. Python 使用本次带 `helix_capture` 的 benchmark。仅把 nsys 放在 mpirun 外面不够。

先短测可用 start=24、steps=4、tail=1，仍保持总调用数足够；然后扩大到 20–50 个完整 P/Q 周期。
下一轮可改固定 rank=8，之后单独测动态 rank。比较 Event/CPU 时固定节点、模型、rank、输入和 seed。
关闭所有测量的正式性能运行使用原成熟快照；标记关闭也不能用 profiler 作业证明无测量性能。

## 5. Nsight 查看及统计

GUI 打开 nsys-rep，展开各 CPU 线程 NVTX / CUDA API / GPU stream。
先按 rs_submit/ag_submit 的 CUDA 关联识别 RS、AG stream，再找 BP/编解码 kernel。
stream 编号以本次报告为准。主机范围的宽度不是 GPU 完成时间；
GPU 的投影跨度也不是 kernel 时间之和，它可能包含排队与空隙。
自动 ATen 和手工阶段可以同时嵌套，同一个 kernel 只能按一套互斥类别统计一次。

```bash
nsys stats --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_api_sum helix_JOB_rank0.nsys-rep
nsys export --type=sqlite --output=helix_rank0.sqlite helix_JOB_rank0.nsys-rep
python measurement_nvtx/analyze_nvtx.py helix_rank0.sqlite --output analysis_rank0
```

统计脚本只用 Python 标准库，不依赖 torch 或 pandas。输出：

- `kernel_attribution.csv`：每个 kernel 的 CUDA 关联、CPU API、GPU 时间、唯一类别、
  手工阶段、ATen operator、autograd node、来源/消费 step、group、rank 和 P/Q，以及是否入选核心窗口。
- `operator_summary.csv`：按进程、设备、阶段、rank、P/Q、group、ATen/autograd 算子和 kernel 汇总，
  同时给出 kernel duration 总和及时间区间并集，自动移除逐次 seq 编号以便跨 step 汇总。
- `overlap_summary.json`：核心来源窗口、未知归属数量、各类别区间并集，NCCL 与 BP/编解码的交集。

归属通过同一进程的 correlationId 找到 CUDA launch API，并按 API 所在线程及时间寻找 NVTX。
不按 GPU kernel 是否落在 CPU NVTX 时间范围内归类，支持 kernel 晚于 NVTX 结束执行。
自动 BP 可能由另一个 autograd 线程提交，脚本用其自动算子标记识别，并从同一进程的
backward 调用区间恢复 step。手工编解码/打包等归属优先于外层 backward 标签。
跨 rank 不直接拼接不同机器的原始时间戳。旧的无 Helix 标记报告会明确报错，不能追补归属。
当前后处理不实现 CUDA Graph 分析，只统计 kernel，不声称包含 memcpy/memset 或硬件 SM 利用率。
未知归属不能默默当 BP；应查看 no_source_kernel_count、selected_unclassified_count、warnings
和 per_source_kernel_counts，确认每个来源 step 的编解码及通信都被覆盖。
`nccl_without_bp_or_codec_ns` 仍可能与前向/更新重叠，且 NCCL kernel 内可包含等待，不能称为纯网络传输或关键路径尾巴。

## 6. 验证状态

已通过 14 项 CPU 检查，以及所有新增 Bash 脚本语法检查和模拟 MPI/Nsight 参数传递检查。
本地检查涵盖真实压缩模块定义的依赖桩导入、线程内元信息、窗口起止/尾部调用/异常、
禁止旧同步探针、跨线程与跨进程 CUDA launch 归属、嵌套唯一归属及区间交集。
去掉标记和窗口接线后，四个原文件的 AST 与服务器运行基线 93bdad7 一致，
用于确认原数学表达式、调用和控制顺序未改。该静态检查不能替代 GPU 数值 A/B。

真实 GPU 两 rank smoke、NVTX 在 Nsight 中的实际可见性、loss/rank/collective 一致性、
自动标记开销及多 rank 正式采集仍需在训练环境验证；本机没有 torch/CUDA 训练环境。
首次采集先完成上述短测，再使用长窗口结果做研究结论。

参考：[NVIDIA NVTX 与采集说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#marking-and-labeling-regions)、
[NVIDIA NVTX/kernel 关联口径](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-kern-sum-base-mangled-nvtx-range-kernel-summary)、
[PyTorch emit_nvtx](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx)。
