from __future__ import print_function

import argparse
import numpy as np
from transformers import BertTokenizer, BertForPreTraining, BertConfig
from transformers import AdamW
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import os
from compression import compressors
from datasets import load_dataset
import dopt_rsag as hvd
from torch.utils.data import Dataset, DataLoader
from transformers.utils import logging
import math
logging.set_verbosity_error()


hvd.init()

os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'

# 全局变量
seed = 0 + hvd.rank()
np.random.seed(seed)
torch.manual_seed(seed)
step = 0


#import dopt_rsag_bo as hvd

import timeit
from profiling import benchmark
import time

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='bert',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size')
parser.add_argument('--sentence-len', type=int, default=128,
                    help='input sentence len')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=2000,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')
parser.add_argument('--threshold', type=int, default=536870912, help='Set threshold if mgwfbp is False')
parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')
parser.add_argument('--compressor', type=str, default='none', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
parser.add_argument('--density', type=float, default=1, help='Density for sparsification')
parser.add_argument('--exclude-parts', type=str, default='', help='choices: reducescatter, allgather')
# 动态rank增加：
parser.add_argument('--compress-rank', type=int, default=8)
parser.add_argument('--compress-warmup', type=int, default=1000)
parser.add_argument('--compress-min-numel', type=int, default=16384,
                    help='do not apply low-rank compression when tensor.numel() is below this threshold')
# rank_schedule 用字符串表示预设方案，不在命令行里写dict
parser.add_argument('--rank-schedule', type=str, default=None,
                    choices=[None, 'aggressive', 'gentle','cosine','warmup_decay'])
parser.add_argument('--local-rank', type=int, default=0)

# TODO 打印上面四个值；方便查看

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = str(args.nstreams)

# rank动态变化预设方案映射
RANK_SCHEDULES = {
    None:        None,
    'aggressive': {0: 8, 300: 6, 5000: 4},
    'gentle':     {0: 8, 5000: 6, 20000: 4},
    # 自定义复杂机制：用函数表达任意逻辑
    'cosine':     lambda step: max(1, round(4 * 0.5 * (1 + math.cos(math.pi * step / 10000)))),
    # step=0 → rank=4，step=5000 → rank=2，step=10000 → rank=1，余弦平滑衰减
    
    'warmup_decay': lambda step: (
        4 if step < args.compress_warmup          # 热身阶段高rank
        else max(1, 4 - step // 3000)  # 之后线性衰减
    ),
}


print(
    "PID", os.getpid(),
    "OMPI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK"),
    "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"),
    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
    "current_device", torch.cuda.current_device() if torch.cuda.is_available() else None,
    flush=True
)

# 动态rank的config打印
print("===== Dynamic_rank Training Config =====")
print(f"compress_rank: {args.compress_rank}")
print(f"compress_warmup: {args.compress_warmup}")
print(f"compress_min_numel: {args.compress_min_numel}")
print(f"rank_schedule: {args.rank_schedule}")
print(f"local_rank: {args.local_rank}")
print("========================================")

if args.cuda:
    # Horovod: pin GPU to local rank.
    #print('local rank: ', hvd.local_rank())
    # torch.cuda.set_device(hvd.rank()%4) # GPU全都是节点内时可以用，跨节点时不可用，会导致rank和GPU不对应的
    pass



cudnn.benchmark = True

DATAPATH='/datasets/shshi'
pretrained_path='%s/pretrained'%DATAPATH

#tokenizer = BertTokenizer.from_pretrained(pretrained_path)
# 新增数据代码
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased-local')
# 这是加载本地的 parquet 文件
dataset = load_dataset("parquet", data_files="./wikitext-local/train-00000-of-00001.parquet", split="train")
# dataset = load_dataset("ag_news", split="train[:1%]") 更小的数据集

texts = [x["text"] for x in dataset if len(x["text"].strip()) > 0]

import nltk
nltk.data.path.append('/data/home/sczd744/run/dear_pytorch-master/dear/wikitext-local/nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    if hvd.rank() == 0:
        nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# ===== 新增：构造 documents =====
documents = []
current_doc = []

for text in texts:
    if text.strip() == "":
        if len(current_doc) > 1:
            documents.append(current_doc)
        current_doc = []
    else:
        sentences = sent_tokenize(text)
        current_doc.extend(sentences)

# 最后一个文档
if len(current_doc) > 1:
    documents.append(current_doc)

# model = BertForPreTraining.from_pretrained(pretrained_path) # 原文的bert结构
# model = BertForPreTraining(config) 
# model = BertForPreTraining.from_pretrained('./bert-base-uncased-local') # 读取已经训练过的副本
config = BertConfig.from_pretrained('./bert-base-uncased-local')  # 只读结构，不加载权重
model = BertForPreTraining(config)                                 # 权重随机初始化
vocab_size = model.config.vocab_size

if args.cuda:
    model.cuda()

model.train()   # 保证训练行为正确

# dataloader类
class BertPretrainingDataset(Dataset):
    def __init__(self, documents, tokenizer, max_len, dataset_size=10000):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset_size = dataset_size  # 虚拟长度，控制每个epoch的步数

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # 原来 create_batch 里单条样本的逻辑，搬过来
        while True:
            doc = self.documents[np.random.randint(0, len(self.documents))]
            if len(doc) < 2:
                continue

            i = np.random.randint(0, len(doc) - 1)
            sent_a = doc[i]

            if np.random.rand() < 0.5:
                sent_b = doc[i + 1]
                label = 0
            else:
                if len(self.documents) > 1:
                    rand_idx = np.random.randint(0, len(self.documents) - 1)
                    if self.documents[rand_idx] is doc:
                        rand_idx = len(self.documents) - 1
                    rand_doc = self.documents[rand_idx]
                else:
                    rand_doc = doc
                sent_b = rand_doc[np.random.randint(0, len(rand_doc))]
                label = 1

            encoded = self.tokenizer(
                sent_a, sent_b,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt',
                return_overflowing_tokens=False
            )

            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask'][0]
            token_type_ids = encoded['token_type_ids'][0]

            # MLM mask
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < 0.15) & (input_ids != self.tokenizer.pad_token_id)

            labels = input_ids.clone()
            labels[~mask_arr] = -100

            rand2 = torch.rand(input_ids.shape)
            mask_80 = mask_arr & (rand2 < 0.8)
            input_ids[mask_80] = self.tokenizer.mask_token_id

            mask_10 = mask_arr & (rand2 >= 0.8) & (rand2 < 0.9)
            random_tokens = torch.randint(0, self.tokenizer.vocab_size, input_ids.shape)
            input_ids[mask_10] = random_tokens[mask_10]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'next_sentence_label': torch.tensor(label, dtype=torch.long),
                'masked_lm_labels': labels
            }

max_len = args.sentence_len
batch_size = args.batch_size

# 新增：构建 Dataset 和 DataLoader
dataset_train = BertPretrainingDataset(documents, tokenizer, max_len, dataset_size=50000)

# 分布式训练需要 DistributedSampler 保证每个rank数据不重叠
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(
    dataset_train,
    num_replicas=hvd.size(),
    rank=hvd.rank(),
    shuffle=True
)

train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2,        # 后台预加载线程数
    pin_memory=args.cuda, # GPU训练时加速数据传输
    drop_last=True        # 丢弃最后不足一个batch的数据
)

# 构造一个可以循环取数据的迭代器
data_iter = iter(train_loader)


""" 
# 数据生成老版本，会产生nan的问题
input_ids = (torch.rand(batch_size, max_len) * 2000).long()
attention_masks = torch.rand(batch_size, max_len).long()
token_type_ids = torch.rand(batch_size, max_len).long()
position_ids = (torch.rand(batch_size, max_len) * 10).long()
next_sentence_label = torch.rand(batch_size, 1).long()
masked_lm_labels = torch.rand(batch_size, max_len).long()

input_ids = torch.randint(0, vocab_size, (batch_size, max_len))
attention_masks = torch.ones(batch_size, max_len).long()
token_type_ids = torch.zeros(batch_size, max_len).long()
position_ids = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).long()
next_sentence_label = torch.randint(0, 2, (batch_size,))
masked_lm_labels = torch.randint(0, vocab_size, (batch_size, max_len))  # 改成合法词表 id
"""

""" 随机配置batch数据，不如dataloader
def create_batch(documents, tokenizer, batch_size, max_len):
    input_ids_list = []
    attention_masks_list = []
    token_type_ids_list = []
    masked_lm_labels_list = []
    next_sentence_labels_list = []

    while len(input_ids_list) < batch_size:
        # ===== 新的 NSP 采样 =====
        # 随机选一个文档
        doc = documents[np.random.randint(0, len(documents))]

        # 保证文档长度足够，不够就跳过，继续采
        if len(doc) < 2:
            continue

        idx = np.random.randint(0, len(doc) - 1)
        sent_a = doc[idx]

        if np.random.rand() < 0.5:
            # 正样本（下一句）
            sent_b = doc[idx + 1]
            label = 0
        else:
            # 负样本（随机句）
            if len(documents) > 1:
                rand_idx = np.random.randint(0, len(documents) - 1)
                if documents[rand_idx] is doc:
                    rand_idx = len(documents) - 1
                rand_doc = documents[rand_idx]
            else:
                rand_doc = doc  # fallback（不会影响运行）
                
            sent_b = rand_doc[np.random.randint(0, len(rand_doc))]
            label = 1

        encoded = tokenizer(
            sent_a,
            sent_b,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
            return_overflowing_tokens=False  # 加这一行
        )

        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        token_type_ids = encoded['token_type_ids'][0]

        # ===== MLM mask =====
        rand = torch.rand(input_ids.shape)
    
        mask_arr = (rand < 0.15)
        # 不mask padding
        mask_arr = mask_arr & (input_ids != tokenizer.pad_token_id)

        labels = input_ids.clone()
        labels[~mask_arr] = -100

        # 80% → MASK
        rand2 = torch.rand(input_ids.shape)
        mask_80 = mask_arr & (rand2 < 0.8)
        input_ids[mask_80] = tokenizer.mask_token_id

        # 10% → random token
        mask_10 = mask_arr & (rand2 >= 0.8) & (rand2 < 0.9)
        random_tokens = torch.randint(
            0,
            tokenizer.vocab_size,
            input_ids.shape,
            device=input_ids.device
        )
        input_ids[mask_10] = random_tokens[mask_10]

# 剩下10%保持不变

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        masked_lm_labels_list.append(labels)
        next_sentence_labels_list.append(label)

    return (
        torch.stack(input_ids_list),
        torch.stack(attention_masks_list),
        torch.stack(token_type_ids_list),
        torch.tensor(next_sentence_labels_list, dtype=torch.long),
        torch.stack(masked_lm_labels_list)
    )
"""
        
class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        # print(type(prediction_scores),type(masked_lm_labels)) # Debug
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss
criterion = BertPretrainingCriterion(model.config.vocab_size)

seq_layernames, layerwise_times = None, None



optimizer = torch.optim.AdamW(model.parameters(),
        lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-6 # args.adam_epsilon  - default is 1e-8.
        )
# optimizer = optim.SGD(model.parameters(), lr=5e-6, momentum=0.9, weight_decay=0.01)

# optimizer = AdamW(model.parameters(), lr=5e-5) 
#使用SGD优化器（学习率2e-5）

#compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
# Horovod: wrap optimizer with DistributedOptimizer.
#optimizer = hvd.DistributedOptimizer(optimizer,
#                                     named_parameters=model.named_parameters(),
#                                     compression=compression,
#                                     op=hvd.Average)
if hvd.size() > 1:
    optimizer = hvd.DistributedOptimizer(optimizer, model=model, compression=compressors[args.compressor]
                                         (device=torch.device('cuda', args.local_rank),
    rank=args.compress_rank,
    rank_schedule=RANK_SCHEDULES[args.rank_schedule],
    warmup_steps=args.compress_warmup,
    min_compression_numel=args.compress_min_numel,), 

    is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=1, threshold=args.threshold, writer=None, gradient_path='./', fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, exclude_parts=args.exclude_parts)
    
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    #hvd.broadcast_optimizer_state(optimizer, root_rank=0)
# 如果分布式（Horovod size > 1），则用Horovod的DistributedOptimizer包装：
# 支持梯度压缩、稀疏化（密度<1时使用指定压缩器）、阈值、MG-WFBP等高级功能。广播模型参数和优化器状态以同步多GPU。


# 清零梯度、前向传播计算预测分数和序列关系分数、计算损失、反向传播、优化步骤、CUDA同步。

""" 原本benchmark_step开头：用create_batch配置batch的代码。
def benchmark_step():
    global step
    # --------构造数据集----------
    batch = create_batch(documents, tokenizer, batch_size, max_len)

    if args.cuda:
        batch = tuple(item.cuda() for item in batch)

    input_ids, attention_masks, token_type_ids, next_sentence_label, masked_lm_labels = batch
"""


"""压缩引入了梯度噪声，等效于给每一步加了一个随机扰动，学习率越大这个扰动的影响越大。
PowerSGD 类方法的经验做法是压缩开始时把 lr 降到原来的 0.3-0.5 倍，再用 cosine decay 继续衰减。"""
BASE_LR = 5e-5 #  与optimizer定义时保持一致
WARMUP_STEPS = args.compress_warmup   # 和压缩的 warmup 对齐
"""
# 压缩步骤时降低一次学习率lr
def _adjust_lr(step):
    if step != WARMUP_STEPS:          # 只在切换点触发一次
        return
    new_lr = BASE_LR * 0.5
    for group in optimizer.param_groups:
        group['lr'] = new_lr
    if hvd.rank() == 0:
        print(f"[LR] step={step}, compression activated, lr: {BASE_LR:.2e} -> {new_lr:.2e}")
"""

"""
# cosine decay
def _adjust_lr(step):
    if step < WARMUP_STEPS:
        return
    # 压缩阶段：从 0.5*BASE_LR 开始做 cosine decay
    compress_step = step - WARMUP_STEPS
    total_compress_steps = args.num_iters * args.num_batches_per_iter - WARMUP_STEPS
    cosine_decay = 0.5 * (1 + math.cos(math.pi * compress_step / total_compress_steps))
    new_lr = BASE_LR * 0.5 * cosine_decay
    for group in optimizer.param_groups:
        group['lr'] = new_lr
"""
def _adjust_lr(step):
    # 暂时停用压缩阶段的 lr / optimizer state 调整，先单独观察压缩算法本身。
    return

        
            
def benchmark_step():
    global step, data_iter

    # 从 DataLoader 取一个 batch，耗尽时重新循环
    try:
        batch = next(data_iter)
    except StopIteration:
        sampler.set_epoch(step)   # 每轮重新shuffle
        data_iter = iter(train_loader)
        batch = next(data_iter)

    if args.cuda:
        batch = {k: v.cuda() for k, v in batch.items()}

    input_ids         = batch['input_ids']
    attention_masks   = batch['attention_mask']
    token_type_ids    = batch['token_type_ids']
    next_sentence_label = batch['next_sentence_label']
    masked_lm_labels  = batch['masked_lm_labels']

    # 压缩切换后降低学习率并做cosine衰减，缓解压缩噪声带来的优化震荡
    if args.compressor != 'none':
        _adjust_lr(step)

    optimizer.zero_grad()

    # 测试NaN的来源 SHAN with torch.autograd.detect_anomaly():
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
    prediction_scores = outputs.prediction_logits
    seq_relationship_score = outputs.seq_relationship_logits

    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_label)
    if hvd.rank() == 0 and step % 5 == 0:
        print("step:",step)
        print(f"Loss: {loss.item():.4f}")

    loss.backward()
    
    optimizer.step()

    """ # 这里可以打印一轮训练中：forward、backward、optimizer_step、total等时间
    timings = {
        "zero_grad": t1 - t0,
        "forward": t2 - t1,
        "backward": t3 - t2,
        "optimizer_step": t4 - t3,
        "total": t4 - t0,
    }
    print("\n=== Benchmark_time Results ===")
    for key, value in timings.items():
        print(f"{key:20s}: {value*1000:.2f} ms")
    print("=" * 25)
    """

    
    step += 1

benchmark_step()



def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

# 运行warmup batches（默认10次），不计入基准。
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

log('BERT Large Pretraining, Sentence len: %d' % max_len)
log('Running benchmark...')
img_secs = []
iter_times = []
# 运行基准迭代（默认5次，每迭代默认10 batches），使用timeit测量时间，计算每GPU的句子/秒吞吐量。
for x in range(args.num_iters):
    total_time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    sen_sec = args.batch_size * args.num_batches_per_iter / total_time
    log('Iter #%d: %.1f sentences/sec per GPU' % (x, sen_sec))
    img_secs.append(sen_sec)
    iter_times.append(total_time / args.num_batches_per_iter)

# Results 输出每个迭代的吞吐量、平均迭代时间（带置信区间）、单GPU和总多GPU的句子/秒性能。
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Iteraction time: %.3f +-%.3f' % (np.mean(iter_times), 1.96*np.std(iter_times)))
log('Sen/sec per %s: %.1f +-%.1f' % ('GPU', img_sec_mean, img_sec_conf))
log('Total sen/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), 'GPU', hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
