import os
import datetime
import subprocess
import os
import numpy as np
import time
# 批量执行horovod_mpi_cj.sh 或者 launch.sh脚本，收集不同配置下的训练速度数据，并生成报告。所以称之为benchmarks.py
DEBUG=0
# 一、初始化和配置
# w/o tf
#LOGPREFIX='sc22-notf'
#a2amethods=['wfbp', 'bytescheduler', 'dear']

# w/ tf
LOGPREFIX='sc22-tf'
# a2amethods=['horovod', 'pytorch-ddp', 'mgwfbp', 'dear']
a2amethods=['dear']

psmethods=[]
methods=a2amethods+psmethods

tasks=[('resnet50', 64), ('densenet201', 32), ('inceptionv4', 64), ('bert_base', 64), ('bert', 32)]
#tasks=[('resnet50', 64)]

#nworkers=[8, 16, 32, 64] 
nworkers=[4]

rdmas=[0, 1]
#rdmas=[0]

NUM_OF_TRIES=1
exp_log='exp.log'

def configs():
    cfg = {}
    cfg['LOGPREFIX'] = LOGPREFIX
    cfg['methods'] = methods
    cfg['tasks'] = tasks
    cfg['nworkers'] = nworkers
    cfg['rdmas'] = rdmas
    cfg['NUM_OF_TRIES'] = NUM_OF_TRIES
    cfg['exp_log'] = exp_log
    return cfg

# 二、命令生成（gen_cmd 函数）
# 根据输入（RDMA标志、方法、任务（模型+批量大小）、工作节点数）生成shell命令和日志文件路径。
# 根据方法确定工作文件夹、压缩器（默认为'none'）、阈值（默认为0）和其他标志。
def gen_cmd(rdma, method, task, nworker):
    LOGHOME = os.path.abspath(os.getcwd())
    folder = method 
    compressor = 'none'
    threshold = 0
    ismgwfbp = 1
    if method in a2amethods: # 与PS相对
        isasc = 0
        nstreams = 1
        if method in ['signum', 'eftopk', 'bspa2a', 'wfbp', 'wfbp2', 'dgcsampling']:
            if method in ['signum', 'eftopk', 'dgcsampling']:
                compressor = method 
            folder = 'mgwfbp'
            ismgwfbp = 0
            if method == 'wfbp' or method == 'wfbp2':
                threshold = 0 
                if method == 'wfbp2':
                    nstreams = 2
            else:
                threshold = 536870912
        if method in ['mgwfbp', 'mgwfbp2', 'asc', 'asc2']:
            folder = 'mgwfbp'
            if method in ['mgwfbp2', 'asc2']:
                nstreams = 2
            if method.find('asc')>=0:
                isasc = 1
        logfile = '%s/logs/%s/rdma-%d-method-%s-dnn-%s-bs-%d-nworker-%d-compressor-%s-thres-%d.log' % (LOGHOME, LOGPREFIX, rdma, method, task[0], task[1], nworker, compressor, threshold)
        #cmd = 'cd %s;'% folder
        #cmd += 'rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d asc=%d nstreams=%d ./horovod_mpi_cj.sh >> %s 2>&1' % (rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, isasc, nstreams, logfile)
        script_path = '/data/run01/sczd744/dear_pytorch-master/%s/horovod_mpi_cj.sh'% folder
        script_dir = os.path.dirname(script_path)
        #cmd = 'rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d asc=%d nstreams=%d %s >> %s 2>&1' % (rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, isasc, nstreams, script_path, logfile)
        cmd = 'cd %s; rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d asc=%d nstreams=%d ./horovod_mpi_cj.sh %s >> %s 2>&1' % (script_dir,rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, isasc, nstreams, script_path, logfile)
    else: #PS，使用launch.sh文件执行
        threshold=0
        bytescheduler=0
        if method == 'bspps':
            threshold=536870912
        if method == 'byteschedulerps':
            bytescheduler=1
        logfile = '%s/logs/%s/rdma-%d-method-%s-dnn-%s-bs-%d-nworker-%d-compressor-%s-thres-%d.log' % (LOGHOME, LOGPREFIX, rdma, method, task[0], task[1], nworker, compressor, threshold)
        folder='byteps'
        cmd = 'cd %s;'% folder
        cmd += 'debug=0 rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d bytescheduler=%d ./launch.sh >> %s 2>&1' % (rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, bytescheduler, logfile)
    return cmd, logfile

# 三、实验跟踪和执行
# check_if_finished(cmd)：检查命令是否已在exp_log中记录（避免重复运行）。
def check_if_finished(cmd):
    if not os.path.isfile(exp_log):
        with open(exp_log, 'w+') as f:
            f.write('')
            return False
    with open(exp_log, 'r') as f:
        for line in f.readlines():
            if line.find(cmd) >= 0:
                return True
        return False

# flag_as_finished(cmd)：完成后将命令追加到exp_log。
def flag_as_finished(cmd):
    with open(exp_log, 'a+') as f:
        f.write(cmd+'\n')

# 执行命令，写日志文件。通过subprocess.check_output运行命令。cmd是命令字符串，logfile是日志文件路径。
def execute(cmd, logfile):
    print('%s' % cmd)
    if DEBUG:
        return 0,0

    # 添加这部分：自动赋予脚本执行权限
    if 'horovod_mpi_cj.sh' in cmd:
        # 从cmd中提取实际的脚本路径
        import re
        match = re.search(r'cd\s+(\S+);', cmd)
        if match:
            script_dir = match.group(1)
            script_path = os.path.join(script_dir, 'horovod_mpi_cj.sh')
            if os.path.exists(script_path):
                os.chmod(script_path, 0o755)

        #os.chmod(r'/data/home/sczd744/run/dear_pytorch-master/dear/horovod_mpi_cj.sh', 0o744)
        # script_path = '/data/run01/sczd744/dear_pytorch-master/pytorch-ddp/horovod_mpi_cj.sh'
        #if os.path.exists(script_path):
            #os.chmod(script_path, 0o755)
    #if 'launch.sh' in cmd:
        #os.chmod(r'/data/home/sczd744/run/dear_pytorch-master/dear/horovod_mpi_cj.sh', 0o744)
    
    finished = check_if_finished(cmd)
    # ... 后续代码
    
    if not finished:
        with open(logfile, 'w+') as f:
            x = datetime.datetime.now()
            f.write('#Date: %s\n#CMD: %s\n' % (x.strftime("%b %d %Y %H:%M:%S"), cmd))
        for i in range(NUM_OF_TRIES):
            try:
                #subprocess.check_output(cmd, shell=True)
                subprocess.check_output(cmd, shell=True, env=os.environ.copy())
            except Exception as e:
                print('cmd: %s ERROR: %s' % (cmd, e))
        flag_as_finished(cmd)
    speed = extract_log(logfile)
    return speed

# 逐行读取日志文件。查找包含'Total '和'GPU(s)'的行，提取浮点速度值。计算所有此类速度的均值和标准差。
def extract_log(logfile):
    with open(logfile) as f:
        speeds = []
        for line in f.readlines():
            if line.find('Total ') >= 0 and line.find('GPU(s)') >= 0:
                speed = float(line.split(': ')[-1].split()[0])
                speeds.append(speed)
        mean = np.mean(speeds)
        std = np.std(speeds)
        return mean, std
    return 0, 0

# 四、报告生成和写入
# 初始化报告数据结构。为每个RDMA标志、任务和方法创建
def init_reports():
    reports = {}
    for rdma in rdmas:
        reports[rdma] = {}
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            reports[rdma][task_str] = {}
            for method in methods:
                reports[rdma][task_str][method] = []
    return reports

# 将报告打印到控制台并保存为JSON文件。
def write_reports(reports):
    import json
    print('==== All Reports ======')
    for rdma in rdmas:
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            for method in methods:
                print('rdma:%d,%s,%s'%(rdma, method, ','.join(reports[rdma][task_str][method])))
    with open('reports.json', 'w') as fp:
        json.dump(reports, fp)

# 五、主函数
# 遍历所有RDMA标志、任务、方法和工作节点数，生成命令，执行实验，收集速度数据，并更新报告结构。
def main():
    reports = init_reports()
    for rdma in rdmas:
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            for method in methods:
                for nworker in nworkers:
                    cmd, logfile = gen_cmd(rdma, method, task, nworker)
                    speed = execute(cmd, logfile)
                    #speed_str = '%.3f+-%.3f' % speed
                    speed_str = '%.3f' % speed[0]
                    reports[rdma][task_str][method].append(speed_str)
                    print('Speed: ', speed)
                    print
                    if method in psmethods:
                        killcmd='cd byteps;./stop.sh'
                        subprocess.check_output(killcmd, shell=True)
                        time.sleep(1)
                print('rdma:%d,%s,%s'%(rdma, method, ','.join(reports[rdma][task_str][method])))
    write_reports(reports)


if __name__ == '__main__':
    main()
