"""Read the user's INI configuration -> Slurm -> MPI. Python 3.8+."""
import argparse
import configparser
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
SCHEMA = {
    'job': dict(name=str, nodes=int, nodelist=str, exclude=str, gpus_per_node=int, cpus_per_task=int,
                partition=str, qos=str, time_limit=str, output_root=str),
    'paths': dict(project_root=str, env_script=str, nsys_package=str),
    'training': dict(model=str, batch_size=int, sentence_len=int,
                     num_warmup_batches=int, num_batches_per_iter=int,
                     num_iters=int, nstreams=int, threshold=int, loss_log_every=int),
    'compression': dict(compressor=str, density=float, rank=int, warmup=int,
                       refresh_k=int, min_numel=int, rank_schedule=str,
                       rank_overrides=str, rank_reset_on_change=bool,
                       active_prefix_enabled=bool, embedding_policy=str,
                       stable_rank_levels=str, update_norm_stable_tol=float,
                       update_norm_critical_tol=float, update_norm_patience=int,
                       update_norm_smoothing=float, update_norm_debug_every=int),
    'runtime': dict(event_sync=bool, fp16=bool, rdma=bool, mgwfbp=bool,
                    asc=bool, nccl_debug=str),
    'capture': dict(nsys_enabled=bool, nvtx_enabled=bool, autograd=bool,
                    start_step=int, steps=int, tail_steps=int, trace=str, sample=str),
}
OPTIONAL_FIELDS = {'job': {'cpus_per_task', 'nodelist', 'exclude'}}


def absolute(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base/path).resolve())


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None,
                                      inline_comment_prefixes=('#', ';'))
    with Path(path).open(encoding='utf-8-sig') as stream:
        parser.read_file(stream)
    if parser.defaults():
        raise ValueError('不使用 [DEFAULT]；请在各分组明确填写参数。')
    if set(parser.sections()) != set(SCHEMA):
        raise ValueError('配置分组必须是：' + ', '.join(SCHEMA))
    result = {}
    for section, fields in SCHEMA.items():
        missing = set(fields) - set(parser[section]) - OPTIONAL_FIELDS.get(section, set())
        extra = set(parser[section]) - set(fields)
        if missing or extra:
            raise ValueError('[%s] 缺少 %s；未知参数 %s' % (section, sorted(missing), sorted(extra)))
        result[section] = {}
        for key, kind in fields.items():
            if key in OPTIONAL_FIELDS.get(section, set()):
                raw = parser.get(section, key, fallback='').strip()
                omitted = not raw or (key == 'cpus_per_task' and raw.lower() == 'auto')
                result[section][key] = None if omitted else kind(raw)
                continue
            result[section][key] = (parser.getboolean(section, key) if kind is bool
                                    else kind(parser.get(section, key).strip()))
    paths = result['paths']
    paths['project_root'] = (str(HERE.parents[1]) if paths['project_root'] == 'auto'
                             else absolute(paths['project_root'], Path(path).resolve().parent))
    paths['env_script'] = absolute(paths['env_script'], HERE)
    paths['nsys_package'] = absolute(paths['nsys_package'], Path(paths['project_root']))
    result['job']['output_root'] = absolute(result['job']['output_root'], Path(paths['project_root']))
    validate(result)
    return result


def validate(c):
    j, t, p, cap = (c[k] for k in ('job', 'training', 'compression', 'capture'))
    if j['nodes'] < 2 or j['gpus_per_node'] != 1:
        raise ValueError('当前入口支持至少两个节点、每节点一个 rank 和一张 GPU。')
    if j['cpus_per_task'] is not None and j['cpus_per_task'] < 1:
        raise ValueError('cpus_per_task 应为正数或 auto，也可省略。')
    for key in ('name', 'partition', 'qos'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', j[key]):
            raise ValueError('job.' + key + ' 只能使用字母、数字、下划线、点和横线。')
    if not re.fullmatch(r'(?:\d+-)?\d+:\d{2}:\d{2}', j['time_limit']):
        raise ValueError('time_limit 格式应为 HH:MM:SS 或 D-HH:MM:SS。')
    if any(int(v) >= 60 for v in j['time_limit'].split(':')[-2:]):
        raise ValueError('time_limit 的分钟和秒必须小于 60。')
    if t['model'] not in ('bert', 'bert_base'):
        raise ValueError('本配置入口目前只接入 BERT benchmark。')
    for key in ('batch_size', 'sentence_len', 'num_batches_per_iter', 'num_iters', 'nstreams'):
        if t[key] < 1:
            raise ValueError('training.' + key + ' 必须为正数。')
    if min(t['num_warmup_batches'], t['threshold'], t['loss_log_every']) < 0:
        raise ValueError('训练预热、threshold 和 loss_log_every 不能为负。')
    if p['compressor'] not in ('halfrankk', 'none') or not 0 < p['density'] <= 1:
        raise ValueError('compressor 应为 halfrankk/none；density 应在 (0,1]。')
    if p['rank'] < 1 or min(p['warmup'], p['refresh_k'], p['min_numel'], p['update_norm_debug_every']) < 0:
        raise ValueError('压缩 rank 必须为正数，warmup/refresh_k/min_numel/debug_every 不能为负。')
    if p['rank_schedule'] not in ('fixed', 'aggressive', 'gentle', 'cosine', 'warmup_decay', 'update_norm_stable'):
        raise ValueError('未知 rank_schedule。')
    if p['embedding_policy'] not in ('off', 'word', 'broad'):
        raise ValueError('未知 embedding_policy。')
    if (p['update_norm_patience'] < 1 or not 0 <= p['update_norm_smoothing'] <= 1
            or min(p['update_norm_stable_tol'], p['update_norm_critical_tol']) < 0):
        raise ValueError('更新范数的 patience、smoothing 或阈值不合法。')
    levels = [int(x.strip()) for x in p['stable_rank_levels'].split(',') if x.strip()]
    if not levels or min(levels) < 1:
        raise ValueError('stable_rank_levels 应为逗号分隔的正整数。')
    if c['runtime']['nccl_debug'] not in ('VERSION', 'WARN', 'INFO', 'TRACE'):
        raise ValueError('未知 NCCL 日志级别。')
    if c['runtime']['rdma']:
        raise ValueError('本入口沿用原 rdma=0 的 MPI 启动分支；更换 MPI transport 需单独调整启动实现。')
    if min(cap['start_step'], cap['steps']) < 0 or cap['tail_steps'] < 1:
        raise ValueError('采集 start/steps 不能为负；tail_steps 至少为 1。')
    if cap['sample'] not in ('none', 'process-tree'):
        raise ValueError('sample 支持 none（关闭 CPU 采样）或 process-tree（本实验的 CPU 热点）。')
    if cap['nsys_enabled']:
        if not cap['nvtx_enabled'] or cap['steps'] < 1:
            raise ValueError('Nsight 采集需要 nvtx_enabled=1，steps>0。')
        if t['loss_log_every']:
            raise ValueError('Nsight 算子测量要求 loss_log_every=0。')
        total = total_steps(c)
        if cap['start_step'] + cap['steps'] + cap['tail_steps'] > total:
            raise ValueError('采集窗口超过总训练调用数 %d。' % total)
        if cap['start_step'] <= t['num_warmup_batches']:
            raise ValueError('采集必须晚于训练预热，请增大 start_step。')
        if p['compressor'] == 'halfrankk' and cap['start_step'] <= p['warmup']:
            raise ValueError('采集必须晚于压缩 warmup，请增大 start_step。')
        if not {'cuda', 'nvtx'} <= set(cap['trace'].split(',')):
            raise ValueError('当前算子测量需要 trace 包含 cuda,nvtx。')


def total_steps(c):
    t = c['training']
    return 1 + t['num_warmup_batches'] + t['num_batches_per_iter'] * t['num_iters']


def benchmark_args(c):
    result = []
    mapping = {
        'training': {'model': 'model', 'batch_size': 'batch-size', 'sentence_len': 'sentence-len',
                     'num_warmup_batches': 'num-warmup-batches', 'num_batches_per_iter': 'num-batches-per-iter',
                     'num_iters': 'num-iters', 'nstreams': 'nstreams', 'threshold': 'threshold',
                     'loss_log_every': 'loss-log-every'},
        'compression': {'compressor': 'compressor', 'density': 'density', 'rank': 'compress-rank',
                        'warmup': 'compress-warmup', 'refresh_k': 'compress-refresh-k',
                        'min_numel': 'compress-min-numel', 'rank_schedule': 'rank-schedule',
                        'rank_overrides': 'compress-rank-overrides', 'embedding_policy': 'embedding-policy',
                        'stable_rank_levels': 'stable-rank-levels', 'update_norm_stable_tol': 'update-norm-stable-tol',
                        'update_norm_critical_tol': 'update-norm-critical-tol',
                        'update_norm_patience': 'update-norm-patience', 'update_norm_smoothing': 'update-norm-smoothing',
                        'update_norm_debug_every': 'update-norm-debug-every'},
    }
    for section, fields in mapping.items():
        for key, flag in fields.items():
            result.extend(['--' + flag, str(c[section][key])])
    result.extend(['--active-prefix-enabled', str(int(c['compression']['active_prefix_enabled']))])
    if c['compression']['rank_reset_on_change']:
        result.append('--rank-reset-on-change')
    for key in ('fp16', 'rdma', 'mgwfbp', 'asc'):
        if c['runtime'][key]:
            result.append('--' + key)
    cap = c['capture']
    for flag, value in [('nvtx-capture-start', cap['start_step']),
                        ('nvtx-capture-steps', cap['steps'] if cap['nsys_enabled'] else 0),
                        ('nvtx-capture-tail', cap['tail_steps']),
                        ('nvtx-autograd', int(cap['autograd'] and cap['nsys_enabled']))]:
        result.extend(['--' + flag, str(value)])
    # 旧 overlap 测量不传启用开关；其有效默认值为关闭。
    result.extend(['--overlap-console', '0'])
    return result


def preflight(c, submitting=False):
    root = Path(c['paths']['project_root'])
    required = [root/'dear/bert_benchmark.py', root/'dear/helix_capture.py',
                root/'dear/bert-base-uncased-local', root/'dear/wikitext-local',
                Path(c['paths']['env_script'])]
    for p in required:
        if not p.exists():
            raise ValueError('缺少文件或数据目录：' + str(p))
    if submitting:
        if not shutil.which('sbatch'):
            raise ValueError('未找到 sbatch。请在服务器登录节点提交。')
        if c['capture']['nsys_enabled'] and not Path(c['paths']['nsys_package']).is_file():
            raise ValueError('Nsight 安装包不存在：' + c['paths']['nsys_package'])


def sbatch_command(c, run_dir, config_path):
    j = c['job']
    command = ['sbatch', '--parsable', '--export=ALL', '--job-name=' + j['name'],
               '--nodes=' + str(j['nodes']), '--ntasks-per-node=1', '--gres=gpu:1']
    if j['cpus_per_task'] is not None:
        command.append('--cpus-per-task=' + str(j['cpus_per_task']))
    for key in ('nodelist', 'exclude'):
        if j.get(key):
            command.append('--' + key + '=' + j[key])
    command.extend(['--partition=' + j['partition'],
            '--qos=' + j['qos'], '--time=' + j['time_limit'],
            '--chdir=' + str(Path(c['paths']['project_root'])/'dear'),
            '--output=' + str(run_dir/'job_%j.out'), str(HERE/'job.sbatch'),
            str(Path(config_path).resolve()), str(HERE), str(run_dir)])
    return command


def preview(c, run_dir, config_path):
    print('最终生效配置（由 conf 读取；不使用旧冒烟脚本的参数）：')
    for section in SCHEMA:
        print('\n[' + section + ']')
        for key, value in c[section].items():
            if value is None:
                shown = 'auto（不传 --cpus-per-task）' if key == 'cpus_per_task' else ''
            else:
                shown = int(value) if isinstance(value, bool) else value
            print('%s = %s' % (key, shown))
    cap, t = c['capture'], c['training']
    print('\n总训练调用：%d；rank 数：%d；全局 batch：%d' %
          (total_steps(c), c['job']['nodes'], c['job']['nodes'] * t['batch_size']))
    if cap['nsys_enabled']:
        start, end = cap['start_step'], cap['start_step'] + cap['steps'] - 1
        offset = 1 + t['num_warmup_batches']
        print('核心 source_step：%d--%d；尾步：%d--%d；对应 Iter #%d--%d（可能是部分 batch）' %
              (start, end, end + 1, end + cap['tail_steps'],
               (start-offset)//t['num_batches_per_iter'], (end-offset)//t['num_batches_per_iter']))
        if cap['sample'] == 'process-tree':
            print('CPU 热点采样：开启；每个 rank 启动前会检查所在计算节点的 perf 权限。')
    else:
        print('Nsight 关闭：直接启动 Python，不解压 CLI，不生成 nsys 报告。')
    print('结果目录：' + str(run_dir))
    print('提交命令：' + shlex.join(sbatch_command(c, run_dir, config_path)))


def submit(config_path, dry_run):
    c = load_config(config_path)
    preflight(c, submitting=not dry_run)
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + c['job']['name'] + '_' + uuid.uuid4().hex[:6]
    run_dir = Path(c['job']['output_root'])/name
    preview(c, run_dir, config_path)
    if dry_run:
        print('\n预览完成：没有提交任务，也没有创建结果目录。')
        return
    run_dir.mkdir(parents=True)
    command = sbatch_command(c, run_dir, config_path)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError('Slurm 提交失败：' + completed.stderr.strip())
    job_id = completed.stdout.strip().split(';')[0]
    print('\n已提交 Job ID：' + job_id)
    print('日志：' + str(run_dir/('job_' + job_id + '.out')))
    print('报告：' + str(run_dir/'reports'))


def job_environment(c, path, run_dir):
    env = os.environ.copy()
    root = Path(c['paths']['project_root'])
    run_dir = Path(run_dir)
    cap = c['capture']
    env.update(HELIX_PROJECT_DIR=str(root), HELIX_ENV_SCRIPT=c['paths']['env_script'],
               HELIX_REPORT_DIR=str(run_dir/'reports'), HELIX_CONFIG_FILE=str(Path(path).resolve()),
               HELIX_RUN_DIR=str(run_dir),
               HELIX_LAUNCHER=str(HERE/'submit_experiment.py'), HELIX_BOOTSTRAP=str(HERE/'bootstrap_nsys.sh'),
               HELIX_NSYS_ENABLED=str(int(cap['nsys_enabled'])), HELIX_NVTX=str(int(cap['nvtx_enabled'])),
               HELIX_NSYS_TRACE=cap['trace'], HELIX_NSYS_SAMPLE=cap['sample'],
               DEAR_EVENT_SYNC=str(int(c['runtime']['event_sync'])), DEAR_OVERLAP_NEEDS_SYNC='0',
               NCCL_DEBUG=c['runtime']['nccl_debug'], UCX_WARN_UNUSED_ENV_VARS='n',
               NSYS_PACKAGE=c['paths']['nsys_package'], PYTHONUNBUFFERED='1')
    if cap['nsys_enabled']:
        env['NSYS_LOCAL_ROOT'] = '/tmp/helix_nsys_' + env['SLURM_JOB_ID']
        env['NSYS_BIN'] = env['NSYS_LOCAL_ROOT'] + '/target-linux-x64/nsys'
    return env


JOB_BODY = r'''
set -euo pipefail
set +u
source "$HELIX_ENV_SCRIPT"
set -u
mkdir -p "$HELIX_REPORT_DIR"
export HELIX_HOSTFILE="$HELIX_REPORT_DIR/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames "$SLURM_JOB_NODELIST" | awk '{print $0 " slots=1"}' > "$HELIX_HOSTFILE"
printf 'Helix config: %s\n' "$HELIX_CONFIG_FILE"
if [[ "$HELIX_NSYS_ENABLED" == 1 ]]; then
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        bash "$HELIX_BOOTSTRAP"
fi
exec "$PYTHON" "$HELIX_LAUNCHER" --launch "$HELIX_CONFIG_FILE" --run-dir "$HELIX_RUN_DIR"
'''


def run_job(path, run_dir):
    c = load_config(path)
    if int(os.environ.get('SLURM_NNODES', '0')) != c['job']['nodes']:
        raise ValueError('实际 Slurm 节点数与配置不一致；停止启动。')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('run-job 必须在 Slurm 作业内运行。')
    os.chdir(Path(c['paths']['project_root'])/'dear')
    os.execvpe('bash', ['bash', '-c', JOB_BODY], job_environment(c, path, run_dir))


def mpi_command(c, env):
    prefix = env.get('MPIPATH', '')
    mpi = env.get('MPIRUN') or (str(Path(prefix)/'bin/mpirun') if prefix else 'mpirun')
    command = [mpi]
    if prefix:
        command.extend(['--prefix', prefix])
    command.extend(['-np', str(c['job']['nodes']), '--bind-to', 'none', '--map-by', 'slot',
                    '--hostfile', env['HELIX_HOSTFILE']])
    if env.get('HELIX_MPI_TCP_IF_INCLUDE'):
        command.extend(['--mca', 'btl_tcp_if_include', env['HELIX_MPI_TCP_IF_INCLUDE']])
    for key in ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'HELIX_NVTX', 'HELIX_REPORT_DIR', 'NSYS_BIN',
                'HELIX_NSYS_TRACE', 'HELIX_NSYS_SAMPLE',
                'DEAR_EVENT_SYNC', 'DEAR_OVERLAP_NEEDS_SYNC', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA',
                'NCCL_IB_GID_INDEX', 'NCCL_IB_DISABLE', 'NCCL_DEBUG', 'NCCL_LAUNCH_MODE',
                'UCX_MEMTYPE_CACHE', 'UCX_WARN_UNUSED_ENV_VARS', 'SLURM_JOB_ID', 'PYTHONUNBUFFERED'):
        if key in env:
            command.extend(['-x', key])
    cap = c['capture']
    if cap['nsys_enabled']:
        # rank wrapper 接受配置指定的 trace/sample，其余捕获方式与已测入口相同。
        command.extend(['bash', str(HERE/'profile_rank.sh')])
    command.extend([env.get('PYTHON', 'python'), '-u', 'bert_benchmark.py'])
    command.extend(benchmark_args(c))
    return command


def launch(path):
    c = load_config(path)
    if int(os.environ.get('SLURM_NNODES', '0')) != c['job']['nodes']:
        raise ValueError('当前 conf 的节点数与实际 Slurm 分配不一致；停止启动。')
    command = mpi_command(c, os.environ)
    print('MPI command: ' + shlex.join(command), flush=True)
    os.execvpe(command[0], command, os.environ.copy())


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Helix：从 conf 预览或提交实验')
    parser.add_argument('config', nargs='?', default=str(HERE/'experiment.conf'))
    parser.add_argument('--dry-run', action='store_true', help='只检查并显示最终参数，不提交作业')
    parser.add_argument('--run-job', metavar='CONFIG', help=argparse.SUPPRESS)
    parser.add_argument('--launch', metavar='CONFIG', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.run_job:
            if not args.run_dir:
                raise ValueError('run-job 缺少结果目录。')
            run_job(args.run_job, args.run_dir)
        elif args.launch:
            launch(args.launch)
        else:
            submit(args.config, args.dry_run)
    except (ValueError, OSError, RuntimeError, configparser.Error) as error:
        print('错误：' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
