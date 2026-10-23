# Helix paper reproduction and server-retirement code archive

This branch preserves the working code and scripts recovered from the Helix
server on 2026-10-24. It is based on mature commit
`d15bcf78ae5ef4417727cfee2c3a4e3993b5016c`.

## Versions and scope

- `main` remains the mature algorithm version at `d15bcf7`.
- This archive branch includes the later multi-GPU local-rank initialization
  fix, cluster launch configuration, successful experiment submission scripts,
  rebuild instructions, and optional September profiling scripts.
- The current primary files `dear/compression.py`, `dear/dopt_rsag.py`,
  `dear/tensorfusion.py`, and the three `common/comm_core/src` source files were
  checked against the server snapshot. The primary runtime is the restored
  mature version, not the timeline-instrumented runtime.
- `archive/helix-timeline-20260706` preserves commit `3d00e8c` for the historical
  T-series instrumentation. Additional historical instrumentation source files
  are retained under `backups/` with their original version labels. They are
  not activated by the current primary runtime.
- Raw experimental data are intentionally not uploaded here. The local
  deduplicated data index identifies existing files and newly recovered logs.
  `paper_data_sources.csv` records relative source paths and checksums.

No new training runs were performed for this archive. Code syntax checks and
source comparisons do not replace compiling and testing on a new GPU cluster.

## Recover the code

```bash
git clone --branch archive/helix-paper-repro-20261024 \
  https://github.com/VV97-hub/Dear.git
cd Dear
git submodule update --init --recursive
```

The archived gitlink commits are Horovod
`26fe8813a25bbef5b2dd98676916f7b350050fb0` and signSGD
`5bebcd0f6920dc484db61c4637aea46db96d0871`. Neither upstream repository is
modified by this backup.

## Rebuild the C++/CUDA communication extension

On the original cluster, the command was:

```bash
cd common/comm_core
bash compile.sh
```

`compile.sh` loads the original `py38-hvd`, CUDA 11.8, GCC 11.4, and OpenMPI
4.1.5 modules, then executes `python3 setup.py clean` and
`python3 setup.py install`. On a different cluster, adapt the module names and
CUDA, MPI, and NCCL paths before running it. Do not assume an old committed
`.so` is compatible with a new Python/GPU environment.

`server_environment_20261024.json` records the environment observed on the
retirement date and the installed communication extension's path and hash.
It is not a reconstruction of the exact library versions dynamically loaded
by each historical paper job. Use those jobs' command lines and runtime logs
for historical provenance.

## Experiments and figures

- BERT and GPT launchers remain under `dear/`.
- Optional profiling launchers are in `dear/measurement_20260918/`; their data
  are outside the paper-only data backup requested for this retirement.
- Plotting and verification code is under `plotting/`. These are preserved
  historical generators. Some use the original Windows/WSL absolute data
  paths; set their root/input/output paths for the new machine.
- Final figure 7 contains dense and Helix only; figure 8 contains means and
  sample points without error bars. Historical generators may contain unused
  functions for earlier figures. The local final submission PDF/source ZIP
  remains the authority for the published composition and labels.
- Figures 5/6 use instrumented visible windows. Figure 12 uses host-side
  group-ready-to-RS-issue markers, not GPU-kernel overlap measurements.

The source mappings for BERT scale-out use iterations 20–99 (80 values), and
GPT throughput uses iterations 20–79 (60 values). The backup audit matches all
40 BERT mean values and all 20 GPT absolute throughput values to the final
stored summary tables, allowing their reported rounding.
