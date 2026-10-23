# Helix 成熟版与分支说明

更新日期：2026-10-24。本说明根据服务器源码比对、GitHub 实际分支指针和提交记录整理。

## 先选哪个版本

**成熟算法基线：`main`，算法代码基线提交 `d15bcf7`。**

这是 `Finalize Helix experiment code` 对应的成熟版，包含半步因子流水线、
active-prefix 动态 rank 和 CUDA event 非阻塞同步。它是 2026-09-11 在服务器
恢复使用的主版本。本次 `main` 只添加版本说明和 README 入口，不改变算法源文件。

**若要换机器继续实验，建议从完整复现归档
`archive/helix-paper-repro-20261024` 开始。**

该分支以 `d15bcf7` 为基础，额外保存了服务器上实际使用的多 GPU 初始化修复、
集群启动参数、BERT/GPT/CIFAR 提交脚本、重编译脚本、环境清单和论文绘图代码。
归档代码初始提交为 `f97635d`。

对照服务器的 78 个相关源码/脚本文件已逐一校验哈希。归档分支中的主要
`compression.py`、`dopt_rsag.py`、`tensorfusion.py` 与 C++ 三个源文件保留的是
已恢复的成熟实现。历史 timeline 代码只放在明确命名的历史目录/独立分支中。

这两者的区别是：`main` 保留成熟算法基线，归档分支补齐实际运行与复现所需资料。
不应仅根据文件夹名或分支日期判断哪一版适合做性能实验。

## 全部分支的含义

| 分支 | 本次核对的基线/提交 | 含义与建议用途 |
| --- | --- | --- |
| `main` | 算法基线 `d15bcf7`；之后增加文档 | 成熟算法主入口；本次不修改其算法逻辑。 |
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
