"""CPU-only checks for live configuration, no snapshot files, and launch."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'dear/measurement_nvtx'
spec = importlib.util.spec_from_file_location('submit_experiment', HERE/'submit_experiment.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ConfigSubmitTests(unittest.TestCase):
    def setUp(self):
        self.c = launcher.load_config(HERE/'experiment.conf')

    def test_server_defaults_and_source_window_are_preserved(self):
        t, p = self.c['training'], self.c['compression']
        self.assertEqual((self.c['job']['nodes'], t['batch_size'], t['sentence_len']), (4, 64, 64))
        self.assertEqual((t['num_warmup_batches'], t['num_batches_per_iter'], t['num_iters']), (10, 10, 1000))
        self.assertEqual((p['warmup'], p['rank_schedule'], p['stable_rank_levels']),
                         (5000, 'update_norm_stable', '16,12,8'))
        self.assertEqual(launcher.total_steps(self.c), 10011)
        args = launcher.benchmark_args(self.c)
        def value(flag): return args[args.index(flag) + 1]
        self.assertEqual(value('--num-warmup-batches'), '10')
        self.assertEqual(value('--compress-warmup'), '5000')
        self.assertEqual(value('--nvtx-capture-start'), '5211')
        self.assertEqual(value('--nvtx-capture-steps'), '40')
        self.assertNotIn('--overlap-profile', args)

    def test_typo_and_invalid_windows_fail_before_submission(self):
        cases = [('capture', 'start_step', 5000), ('capture', 'steps', 10000),
                 ('capture', 'nvtx_enabled', False), ('training', 'loss_log_every', 1),
                 ('job', 'gpus_per_node', 2), ('job', 'cpus_per_task', 0), ('job', 'time_limit', '00:99:00'),
                 ('capture', 'sample', 'system-wide')]
        for section, key, value in cases:
            c = copy.deepcopy(self.c); c[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): launcher.validate(c)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.conf'
            p.write_text((HERE/'experiment.conf').read_text(encoding='utf-8').replace(
                'num_iters = 1000', 'num_itres = 1000'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'num_itres'): launcher.load_config(p)

    def test_cpu_sampling_scope_is_checked_and_forwarded_to_mpi(self):
        for mode in ('none', 'process-tree'):
            with self.subTest(mode=mode):
                self.c['capture']['sample'] = mode
                launcher.validate(self.c)
                env = dict(SLURM_JOB_ID='CPU_TEST')
                with patch.dict(os.environ, env):
                    effective = launcher.job_environment(self.c, HERE/'experiment.conf', '/result')
                self.assertEqual(effective['HELIX_NSYS_SAMPLE'], mode)
                effective['HELIX_HOSTFILE'] = '/hosts'
                command = launcher.mpi_command(self.c, effective)
                self.assertIn('HELIX_NSYS_SAMPLE', command)

    def test_cpu_count_can_be_omitted_or_explicitly_requested(self):
        text = (HERE/'experiment.conf').read_text(encoding='utf-8')
        original = next(line for line in text.splitlines() if line.startswith('cpus_per_task ='))
        for setting, expected in [('', None), ('# '+original, None),
                                  ('cpus_per_task =', None), ('cpus_per_task = auto', None),
                                  ('cpus_per_task = 1', 1), ('cpus_per_task = 8', 8)]:
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as tmp:
                conf = Path(tmp)/'test.conf'
                conf.write_text(text.replace(original, setting), encoding='utf-8')
                c = launcher.load_config(conf)
                self.assertEqual(c['job']['cpus_per_task'], expected)
                command = launcher.sbatch_command(c, Path(tmp)/'results', conf)
                cpu_flags = [arg for arg in command if arg.startswith('--cpus-per-task=')]
                self.assertEqual(cpu_flags, [] if expected is None else ['--cpus-per-task='+str(expected)])

    def test_node_selection_is_optional_and_hostlist_expressions_are_preserved(self):
        text = (HERE/'experiment.conf').read_text(encoding='utf-8')
        for field in ('nodelist', 'exclude'):
            text = '\n'.join(line for line in text.splitlines()
                             if not line.startswith(field+' =')) + '\n'
        cases = [('', []), ('nodelist =\nexclude =\n', []),
                 ('nodelist = g0014,g0022\n', ['--nodelist=g0014,g0022']),
                 ('exclude = g[0017-0018]\n', ['--exclude=g[0017-0018]']),
                 ('nodelist = g[0014,0022]\nexclude = g0017,g0018\n',
                  ['--nodelist=g[0014,0022]', '--exclude=g0017,g0018'])]
        for settings, expected in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp:
                conf = Path(tmp)/'test.conf'
                conf.write_text(text.replace('[job]', '[job]\n'+settings), encoding='utf-8')
                c = launcher.load_config(conf)
                command = launcher.sbatch_command(c, Path(tmp)/'results', conf)
                flags = [arg for arg in command if arg.startswith(('--nodelist=', '--exclude='))]
                self.assertEqual(flags, expected)
                self.assertEqual(command[-3], str(conf.resolve()))

    def test_baseline_bypasses_nsys_and_capture_but_keeps_training(self):
        for key in ('nsys_enabled', 'nvtx_enabled', 'autograd'): self.c['capture'][key] = False
        args = launcher.benchmark_args(self.c)
        self.assertEqual(args[args.index('--nvtx-capture-steps')+1], '0')
        self.assertEqual(args[args.index('--num-iters')+1], '1000')
        env = dict(HELIX_HOSTFILE='/tmp/hosts', PYTHON='/conda/bin/python', MPIPATH='/mpi')
        command = launcher.mpi_command(self.c, env)
        self.assertNotIn(str(HERE/'profile_rank.sh'), command)
        self.assertEqual(command[command.index('-np')+1], '4')
        self.assertEqual(command[command.index('--prefix')+1], '/mpi')

    def test_dry_run_has_no_submission_or_filesystem_writes(self):
        with patch.object(launcher, 'preflight'), patch.object(launcher, 'preview'), \
             patch.object(launcher.subprocess, 'run') as submit, patch.object(Path, 'mkdir') as mkdir, \
             redirect_stdout(io.StringIO()):
            launcher.submit(HERE/'experiment.conf', True)
        submit.assert_not_called(); mkdir.assert_not_called()

    def test_submission_passes_original_conf_and_creates_no_snapshot_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf = Path(tmp)/'test.conf'
            text = (HERE/'experiment.conf').read_text(encoding='utf-8')
            conf.write_text(text.replace('output_root = dear/logs/nvtx', 'output_root = '+Path(tmp).as_posix()), encoding='utf-8')
            done = subprocess.CompletedProcess([], 0, '12345;cluster\n', '')
            with patch.object(launcher, 'preflight'), patch.object(launcher, 'preview'), \
                 patch.object(launcher.subprocess, 'run', return_value=done) as submit, \
                 redirect_stdout(io.StringIO()):
                launcher.submit(conf, False)
            command = submit.call_args.args[0]
            self.assertIn('--nodes=4', command)
            self.assertIn('--time=02:00:00', command)
            self.assertEqual(Path(command[-3]), conf.resolve())
            run_dir = Path(command[-1])
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(list(run_dir.iterdir()), [])
            conf.write_text(text.replace('num_iters = 1000', 'num_iters = 900'), encoding='utf-8')
            self.assertEqual(launcher.load_config(command[-3])['training']['num_iters'], 900)

    @unittest.skipUnless(os.name == 'posix', 'Bash integration runs on Linux')
    def test_rank_wrapper_passes_configured_trace_and_preserves_target_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fake = root/'nsys'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_NSYS_ARGS"\n')
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update(NSYS_BIN=str(fake), HELIX_REPORT_DIR=str(root), HELIX_NSYS_TRACE='nvtx,cuda',
                       HELIX_NSYS_SAMPLE='none', TEST_NSYS_ARGS=str(root/'args'),
                       OMPI_COMM_WORLD_RANK='3', SLURM_JOB_ID='CPU_TEST')
            completed = subprocess.run(['bash', str(HERE/'profile_rank.sh'), '/python path/bin/python',
                                        'bert_benchmark.py', '--compress-rank', '16'], env=env)
            self.assertEqual(completed.returncode, 0)
            args = (root/'args').read_text().splitlines()
            self.assertIn('--trace=nvtx,cuda', args)
            self.assertIn('--capture-range=cudaProfilerApi', args)
            self.assertIn('--output='+str(root/'helix_CPU_TEST_rank3'), args)
            self.assertEqual(args[-4:], ['/python path/bin/python', 'bert_benchmark.py', '--compress-rank', '16'])

    @unittest.skipUnless(os.name == 'posix', 'Bash integration runs on Linux')
    def test_cpu_sampling_preflight_stops_on_failed_native_sampling(self):
        for supported in (False, True):
            with self.subTest(supported=supported), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fake = root/'nsys'
                state = 'OK' if supported else 'Fail'
                fake.write_text('#!/bin/bash\nif [[ "$1" == status ]]; then\n'
                                'echo "- CPU Profiling Environment (process-tree): '+state+'"\n'
                                'else\nprintf "%s\\n" "$@" > "$TEST_NSYS_ARGS"\nfi\n')
                fake.chmod(0o755)
                env = os.environ.copy()
                env.update(NSYS_BIN=str(fake), HELIX_REPORT_DIR=str(root), HELIX_NSYS_SAMPLE='process-tree',
                           TEST_NSYS_ARGS=str(root/'args'), OMPI_COMM_WORLD_RANK='0')
                completed = subprocess.run(['bash', str(HERE/'profile_rank.sh'), 'python', 'bert_benchmark.py'],
                                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(completed.returncode, 0 if supported else 1)
                self.assertEqual((root/'args').exists(), supported)
                if supported:
                    self.assertIn('--sample=process-tree', (root/'args').read_text())

    @unittest.skipUnless(os.name == 'posix', 'Bash integration runs on Linux')
    def test_job_shell_and_mpi_use_dynamic_node_count_without_gpus(self):
        # Fake scheduler/MPI binaries record calls; no sbatch, CUDA or GPU work.
        for profiling in (False, True):
            with self.subTest(profiling=profiling), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'dear').mkdir(); binaries = root/'bin'; binaries.mkdir()
                fake_env = root/'env.sh'
                fake_env.write_text('export PYTHON='+sys.executable+'\nexport MPIRUN='+str(binaries/'mpirun')+'\n')
                scripts = {
                    'scontrol': '#!/bin/bash\nprintf "node1\\nnode2\\nnode3\\nnode4\\n"\n',
                    'srun': '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_SRUN_ARGS"\n',
                    'mpirun': '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_MPI_ARGS"\nprintf "%s\\n" "$HELIX_NVTX:$NCCL_DEBUG" > "$TEST_MPI_ENV"\n',
                }
                for name, source in scripts.items():
                    f = binaries/name; f.write_text(source); f.chmod(0o755)
                text = (HERE/'experiment.conf').read_text(encoding='utf-8')
                text = text.replace('project_root = auto', 'project_root = '+str(root))
                text = text.replace('env_script = env_sczd744.sh', 'env_script = '+str(fake_env))
                if not profiling:
                    for key in ('nsys_enabled', 'nvtx_enabled', 'autograd'):
                        text = text.replace(key+' = 1', key+' = 0')
                conf = root/'experiment.conf'; conf.write_text(text, encoding='utf-8')
                env = os.environ.copy()
                env.update(PATH=str(binaries)+os.pathsep+env['PATH'], SLURM_NNODES='4',
                           SLURM_JOB_ID='CPU_TEST', SLURM_JOB_NODELIST='node[1-4]',
                           TEST_SRUN_ARGS=str(root/'srun_args'), TEST_MPI_ARGS=str(root/'mpi_args'),
                           TEST_MPI_ENV=str(root/'mpi_env'))
                completed = subprocess.run(['bash', str(HERE/'job.sbatch'), str(conf), str(HERE), str(root/'result')],
                                           env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                args = (root/'mpi_args').read_text().splitlines()
                self.assertEqual(args[args.index('-np')+1], '4')
                self.assertEqual(args[args.index('--compress-warmup')+1], '5000')
                self.assertEqual((root/'srun_args').exists(), profiling)
                if profiling:
                    self.assertIn('--nodes=4', (root/'srun_args').read_text())
                    self.assertIn('--ntasks=4', (root/'srun_args').read_text())
                    self.assertIn(str(HERE/'profile_rank.sh'), args)
                else:
                    self.assertNotIn(str(HERE/'profile_rank.sh'), args)
                self.assertEqual((root/'mpi_env').read_text().strip(), ('1' if profiling else '0')+':INFO')
                self.assertEqual({p.name for p in (root/'result').iterdir()}, {'reports'})


if __name__ == '__main__': unittest.main()
