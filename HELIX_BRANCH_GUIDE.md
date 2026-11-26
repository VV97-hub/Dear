# Helix 成熟版与分支说明

更新日期：2026-11-26。当前路线基于 main `93bdad7`、NVTX 合并基线 `3b2693a` 和启动入口提交 `ae04ef8`；历史归档信息保留在后文。

## 当前的算法开发与测量路线

| 路线 | 分支 | 放置的修改 |
|---|---|---|
| 算法开发、正式训练 | `main`；功能开发从它创建 `codex/helix-*` 分支 | 算法、压缩策略、运行正确性和通用修复。功能验证后合回 main。 |
| NVTX / Nsight 测量 | `扩刊-2026-11-24`；测量功能可从它创建 `codex/nvtx-*` 分支 | NVTX 标记、采集窗口、kernel 归属分析、conf / Slurm 启动和 CPU 采样支持。 |
| 历史复现 | `archive/*` | 固定历史材料，不作为日常开发分支。 |

现有测量分支已经包含 `9e949bf` 的标记/分析实现，并在 `3b2693a` 合并 main 的服务器运行修复。`ae04ef8` 是后续的集中配置与 Slurm 启动器提交；它继续沿用同一测量分支，不另建第二条长期测试线。

同步方向以 **main → NVTX 测量分支** 为主。算法功能先合回 main，再同步到测量线，使正式训练与测量使用同一算法基线。测量分支包含 instrumentation，不整体合回 main。确需回流的通用修复应单独提交，先落到 main，再同步到测量线。

### 使用独立工作目录

当前 Dear 目录可继续用于 NVTX。在这个仓库内可以另外建立算法开发工作区，例如：

```bash
git fetch origin
git worktree add -b codex/helix-next ../Dear_dev origin/main
```

这样 `Dear_dev` 用于算法功能开发，原 Dear 目录保留 NVTX 测量分支，两个目录的源码编辑互不覆盖。模型/数据路径和环境仍按各自运行环境准备。服务器也应保留独立的正式训练目录和 NVTX 部署目录，避免混拷两个版本的源码。

### 后续提交与同步

1. 算法改动从 main 创建功能分支，检查后合回 main；新的 CUDA/C++ 修改需按环境重新编译。
2. 在测量工作区同步 main：

   ```bash
   git switch 扩刊-2026-11-24
   git fetch origin
   git merge origin/main
   python -m unittest discover -s tests -p 'test_*nvtx*.py' -v
   ```

3. `tests/test_nvtx_measurement.py` 的数学/执行顺序检查当前以 `93bdad7` 为基线。同步新的算法实现时，应将其中的比较 SHA 更新为此次同步的 main 提交，使它比较同一个算法版本的带标记/无标记代码。
4. 测量工具修改留在测量线；如使用功能分支，合并目标是 `扩刊-2026-11-24`。
5. 正式数据先用无采集配置测吞吐，再用相同算法版本做 NVTX 采集。CPU 检查、短时 GPU 冒烟和正式性能验证分别记录结论，不能相互替代。

### 配置与实验数据

Git 保存 `experiment.conf` 默认模板、启动/分析源码和测试。个人实验参数可写成 `*.local.conf`，该模式已加入 `.gitignore`；需要有意保存的标准实验方案另取文件名后提交。原始日志、`.nsys-rep`、SQLite、模型和数据留在实验数据目录。启动器按要求不生成自动配置快照。

本次提交不将服务器上临时修改的节点数、rank 或短测步数覆盖到仓库默认模板。实际复现实验时，应在实验记录中标明算法提交、测量提交和使用的配置。

## 历史复现版本与基线

**成熟算法与正式训练：`main`，算法基线为 `d15bcf7`，当前运行适配提交为 `93bdad7`。**

这是 `Finalize Helix experiment code` 对应的成熟版，包含半步因子流水线、
active-prefix 动态 rank 和 CUDA event 非阻塞同步。它是 2026-09-11 在服务器
恢复使用的主版本。之后 main 的 `93bdad7` 同步了经服务器核对的 GPU 初始化、网络环境和运行脚本修复；NVTX 标记与新测量入口保留在测量分支。

**若要复现归档论文实验或查阅当时的完整环境资料，可使用
`archive/helix-paper-repro-20261024`。日常算法开发和最新 NVTX 测量按前文两条路线进行。**

该分支以 `d15bcf7` 为基础，额外保存了服务器上实际使用的多 GPU 初始化修复、
集群启动参数、BERT/GPT/CIFAR 提交脚本、重编译脚本、环境清单和论文绘图代码。
归档代码初始提交为 `f97635d`。

对照服务器的 78 个相关源码/脚本文件已逐一校验哈希。归档分支中的主要
`compression.py`、`dopt_rsag.py`、`tensorfusion.py` 与 C++ 三个源文件保留的是
已恢复的成熟实现。历史 timeline 代码只放在明确命名的历史目录/独立分支中。

这两者的区别是：`main` 是当前算法与正式训练入口，归档分支另外保存历史复现所需的完整脚本、环境和资料。
不应仅根据文件夹名或分支日期判断哪一版适合做性能实验。

## 全部分支的含义

| 分支 | 本次核对的基线/提交 | 含义与建议用途 |
| --- | --- | --- |
| `main` | 算法基线 `d15bcf7`；运行适配 `93bdad7` | 算法开发与正式训练入口。 |
| `扩刊-2026-11-24` | 标记/分析 `9e949bf`；合并基线 `3b2693a`；conf 启动器 `ae04ef8` | NVTX / Nsight 测量线；从 main 同步算法更新，测量工具在本分支维护。 |
| `archive/helix-paper-repro-20261024` | 归档初始提交 `f97635d` | 最新完整复现归档，包含服务器实际运行支持修改、实验提交与测量脚本、绘图代码、环境和数据来源清单。换机器继续研究优先读这个分支的 `reproducibility/README.md`。 |
| `archive/helix-timeline-20260706` | `3d00e8c` | 在成熟版上加入最终 T 系列 timeline instrumentation。用于追踪历史测量与复现可见窗口分析；性能比较时必须检查测量开关，不能把它当成默认无额外测量的主版本。 |
| `work/update-norm-dynamic-rank` | `d15bcf7` | 动态 rank 开发分支，远端仍停在与成熟版相同的算法提交。不是比 `main` 更新的自适应 rank 方案。 |
| `backup/cuda-event-sync-20260526` | `7dd261e` | 2026-05-26 CUDA event 同步开发状态的备份。适合追查当时同步设计，不是最终成熟版本。 |
| `backup/pre-event-sync-20260522` | `0f7eb5c` | 提交说明为 `Backup before event sync optimization`，保存 event 同步优化前的开发状态。不能直接视为最终论文的 CPU-sync 对照配置；应以最终运行开关和具体作业日志为准。 |
| `backup/pre-sync-fix-20260420` | `a27ef06` | 同步问题排查时保存的历史状态，提交说明为 `Backup current sync investigation state`。仅用于代码演化追踪。 |
| `fix/sync-validation-20260420` | `8a4080f` | 历史同步修复/验证开发线；分支尖端实际包含 nested rank 和 active-prefix 的提交。分支名保留了早期日期，不代表其所有改动都发生在该日。不是当前推荐主版本。 |
| `no-embedding-compression-cycle` | `fcea93a` | 早期 embedding 压缩策略与 cycle-based overlap timeline 探索状态；提交说明为 `make overlap timeline cycle-based`。不要仅凭分支名把它当成最终论文的 embedding-off 配置。 |

另有旧标签 `同步策略`，指向 `976a59f`（`Merge nested rank and active prefix buffers`）。
它是历史节点，不是最新成熟版标签。

## 下载和继续修改

完整归档：

```bash
git clone --branch archive/helix-paper-repro-20261024 \
  https://github.com/VV97-hub/Dear.git
cd Dear
git submodule update --init --recursive
git switch -c work/my-next-experiment
```

只查看确定的成熟算法快照：

```bash
git checkout d15bcf78ae5ef4417727cfee2c3a4e3993b5016c
```

如需继续改代码，先建自己的工作分支，保留成熟快照和历史归档便于回溯。

## C++/CUDA 扩展需要重新编译

仅替换 Python 文件不能更新已安装的 `comm_core` 扩展。原服务器上的命令是：

```bash
cd common/comm_core
bash compile.sh
```

脚本会加载原环境并执行 `python3 setup.py clean` 与
`python3 setup.py install`。新集群需先适配 Conda、CUDA、GCC、MPI 和 NCCL
路径；不要直接沿用旧机器的 `.so`。新机器上的编译与 GPU 测试尚未进行。

## 数据和复现范围

原始数据不上传到本仓库，保留在本地去重索引中。2026-10-24 的补充备份只新增：

- 40 份 BERT 扩展性原始日志；
- 20 份 GPT 吞吐量原始日志；
- 4 份 GPT 长训练原始日志；
- 2 份 BERT 扩展性来源/清理记录。

合计 66 份、约 9.97 MiB。原本已有的论文数据仅建立路径和哈希引用，未复制另一套。
桌面补充目录为 `Helix_论文数据补充备份_20261024`，先读其中的
`先看这里_数据集情况简介.txt`，再查 `论文数据保存索引.xlsx`。

图 10 的 40 个 BERT 柱状均值与表 1 的 20 个 GPT 绝对吞吐量已对应到原始日志，
允许最终汇总表的舍入误差。本次没有重新训练，也没有重新审查/改写全部历史选样、
误差棒或统计口径。归档分支中的 `reproducibility/server_environment_20261024.json`
记录的是退役核对当天的环境，不等同于每个历史作业动态加载库的精确版本证明。
