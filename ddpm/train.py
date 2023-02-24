import math
import time
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DenoiseDiffusion, cosine_schedule, linear_schedule, UNet
from data import DiffusionSet
from utils import set_seed, save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"


# config file - (overrides the parameters given here)
# CFG = './config/pretrain.py'  # 'path/to/config/file.py'
CFG = None
# =============================================================
# Parameters
# =============================================================

# model
IN_CHANNELS = 3
OUT_CHANNELS = None
DIM = 16
DIM_MULTS = (1, 2, 4, 8)
ATTNS = (False, False, True, True)
N_BLOCKS = 1
GROUPS = 4

# logging
LOGDIR = Path('./data/runs')
LOAD = LOGDIR/'ckpt.pt'  # or None
PRINT_INTERVAL = 10

# dataset
TIMESTEPS = 100
IMG_SIZE = 64

# training
BATCH_SIZE = 32
GRAD_ACC_STEPS = 1  # used to simulate larger batch sizes
MAX_ITERS = 100_000  # total number of training iterations
EVAL_ITERS = 100
EVAL_ONLY = False  # if True, script exits right after the first eval
SAVE_EVERY = False  # save unique checkpoint at every eval interval.
GRADIENT_CLIP = None  # 5
# adamw optimizer
LR = 6e-4  # max learning rate
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95
# learning rate decay settings
DECAY_LR = True  # whether to decay the learning rate
WARMUP_ITERS = 2000  # how many steps to warm up for
LR_DECAY_ITERS = MAX_ITERS  # should be ~= max_iters per Chinchilla
MIN_LR = LR/10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# init

# warning!!! executes codes in config file directly with no safety!
if CFG is not None:
    with open(CFG, 'r') as fp: exec(fp.read())  # import cfg settings
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)
set_seed(108)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True  # optimize backend algorithms
extras = {}

# =============================================================
# Dataset, Dataloader init
# =============================================================
trainset = DiffusionSet(IMG_SIZE, 'train', TIMESTEPS)
evalset = DiffusionSet(IMG_SIZE, 'test', TIMESTEPS)
print(f'Total training samples = {len(trainset)}')

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
)
evalloader = DataLoader(
    evalset, batch_size=BATCH_SIZE, shuffle=True,
)

# =============================================================
# Model, Optimizer, Criterion init and Checkpoint load
# =============================================================
net = UNet(
    IN_CHANNELS, OUT_CHANNELS, 
    DIM, DIM_MULTS, ATTNS, 
    N_BLOCKS, GROUPS,
)
net_state, optim_state, itr, best, kwargs = load_checkpoint(LOAD)
if net_state is not None:
    net.load_state_dict(net_state)
net.to(DEVICE)
optimizer = optim.AdamW(net.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
if optim_state is not None:
    optimizer.load_state_dict(optim_state)
criterion = nn.MSELoss()
print(f'Total model parameters = {net.n_params} = {net.n_params/1e6}M')

# =============================================================
# Learning Rate Decay Scheduler (cosine with warmup)
# =============================================================
def get_lr(iter):
    """Refs:
            https://github.com/karpathy/nanoGPT/blob/master/train.py
    """
    # 1) linear warmup for warmup_iters steps
    if iter < WARMUP_ITERS:
        return LR * iter / WARMUP_ITERS
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (LR - MIN_LR)

# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
trainloss, valloss, log_trainloss = 0, 0, 0
trainloader = iter(trainloader)
net.train()
optimizer.zero_grad(set_to_none=True)
# set_to_none -> instead of filling grad with zero tensor set it to None
# reduce memory consumptionn + increases speed
print('Training ...')
start_time = time.perf_counter()
for itr in range(itr, MAX_ITERS + 1):
    # =============================================================
    # Validation
    # =============================================================
    if (itr%EVAL_INTERVAL == 0) or EVAL_ONLY:
        print('Evaluating ...')
        net.eval()
        valloss = 0
        with torch.no_grad():
            for inp, tar in tqdm(evalloader, total=len(evalloader)):
                inp, tar = inp.to(DEVICE), tar.to(DEVICE)
                logits = net(inp)
                loss = criterion(logits.reshape(-1, tokenizer.n_vocab), tar.reshape(-1))
                valloss += loss.item()
        net.train()

        valloss = valloss/EVAL_ITERS
        trainloss = trainloss/EVAL_INTERVAL

        # =============================================================
        # Saving and Logging
        # =============================================================
        if EVAL_ONLY:
            log_data = f'val loss: {valloss}, \t time: {(end_time - start_time)/60}M'
            print(f'{"-"*150}\n{log_data}\n{"-"*150}')
            break

        print('Saving checkpoint ...')
        ckpt_name = LOGDIR/'ckpt.pt' if not SAVE_EVERY else LOGDIR/f'ckpt_{itr}.pt'
        save_checkpoint(
            net, optimizer, itr, valloss, trainloss, best, ckpt_name, **extras,
        )

        if valloss < best:
            best = valloss
            save_checkpoint(
                net, optimizer, itr, valloss, trainloss, best, LOGDIR/'best.pt', **extras,
            )

        write_pred(inp[0], logits[0], tokenizer, LOGDIR/'predictions.txt', label=f'iteration = {itr}')

        logfile = LOGDIR/'log.txt'
        log_data = f"iteration: {itr}/{MAX_ITERS}, \tval loss: {valloss}, \ttrain loss: {trainloss}, \tbest loss: {best}"
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')
        end_time = time.perf_counter()
        log_data = f'{log_data}, \t time: {(end_time - start_time)/60}M'
        print(f'{"-"*150}\n{log_data}\n{"-"*150}')

        trainloss = 0
        start_time = time.perf_counter()
        print('Training ...')

    # =============================================================
    # Training
    # =============================================================
    # forward, loss, backward with grad. accumulation
    loss_ = 0
    for step in range(GRAD_ACC_STEPS):
        inp, tar = next(trainloader)
        inp, tar = inp.to(DEVICE), tar.to(DEVICE)

        logits = net(inp)
        loss = criterion(logits.reshape(-1, tokenizer.n_vocab), tar.reshape(-1))
        loss.backward()

        loss_ += loss.item()

    # optimize params
    loss_ = loss_/GRAD_ACC_STEPS
    trainloss += loss_
    log_trainloss += loss_
    if GRADIENT_CLIP is not None:
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)

    # cosine scheduler with warmup learning rate decay
    if DECAY_LR:
        lr = get_lr(itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # print info.
    if itr%PRINT_INTERVAL == 0:
        log_data = f"iteration: {itr}/{MAX_ITERS}, \ttrain loss: {log_trainloss/PRINT_INTERVAL}"
        print(log_data)
        log_trainloss = 0

# =============================================================
# END
# =============================================================