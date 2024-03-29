import math
import time
from pathlib import Path
import sys

from tqdm import tqdm
import torch

from model import VQVAE
from data import load_cifar, data_loaders
from utils import set_seed, save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"


# config file - (overrides the parameters given here)
CFG = None
if len(sys.argv) > 1: CFG = str(sys.argv[1])
# =============================================================
# Parameters
# =============================================================
# model
IN_CH = 3
RES_LAYERS = 2
HIDDEN_CH = 256
NUM_EMB = 8*8*10
BETA = 0.25
# logging
LOGDIR = Path('./data/runs')
CKPT = LOGDIR/'ckpt.pt'  # or None
PRINT_INTERVAL = 10
# training
GRAD_ACC_STEPS = 1  # used to simulate larger batch sizes
MAX_ITERS = 100_000

EVAL_INTERVAL = 500
EVAL_ITERS = 100
EVAL_ONLY = False  # if True, script exits right after the first eval
SAVE_EVERY = False  # save unique checkpoint at every eval interval.
GRADIENT_CLIP = None  # 5
# adam optimizer
BATCH_SIZE = 64
LR = 2e-4
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# =============================================================
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)
# set_seed(108)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True  # optimize backend algorithms

# warning!!! executes codes in config file directly with no safety!
if (CFG is not None) and Path(CFG).is_file():
    print(f'Reading configuration from {CFG} ...')
    with open(CFG, 'r') as fp: exec(fp.read())  # import cfg settings

# =============================================================
# Dataset, Dataloader init
# =============================================================
trainset, evalset = load_cifar()
trainloader, evalloader = data_loaders(trainset, evalset, BATCH_SIZE)
# =============================================================
# Load Checkpoint
# =============================================================
vqvae_ckpt, itr, best, kwargs = load_checkpoint(CKPT)

# =============================================================
# VQVAE(Model, Optimizer, Criterion) init and checkpoint load
# =============================================================
vqvae = VQVAE(
    IN_CH, RES_LAYERS, HIDDEN_CH, NUM_EMB, 
    BETA, LR, DEVICE,
    vqvae_ckpt, inference=False,
)

# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
trainloss, valloss, log_trainloss = 0, 0, 0
vqvae.train()
vqvae.zero_grad(set_to_none=True)
trainloader_, evalloader_ = iter(trainloader), iter(evalloader)
print('Training ...')
start_time = time.perf_counter()
for itr in range(itr, MAX_ITERS + 1):
    # =============================================================
    # Validation
    # =============================================================
    if (itr%EVAL_INTERVAL == 0) or EVAL_ONLY:
        print('Evaluating ...')
        vqvae.eval()
        valloss = 0
        with torch.no_grad():
            for _ in tqdm(range(EVAL_ITERS)):
                try: data, lbl = next(evalloader_)
                except StopIteration:
                    evalloader_ = iter(evalloader)
                    data, lbl = next(evalloader_)
                data = data.to(DEVICE)
                (ze, z, zq, pred), loss = vqvae.forward(data)
                valloss += loss.item()
        vqvae.train()

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
            vqvae.get_ckpt(), itr, valloss, trainloss, best, ckpt_name,
        )

        if valloss < best:
            best = valloss
            save_checkpoint(
                vqvae.get_ckpt(), itr, valloss, trainloss, best, LOGDIR/'best.pt',
            )
        
        write_pred(pred, LOGDIR/'predictions', str(itr), scale=2)

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
        try: data, lbl = next(trainloader_)
        except StopIteration:
            trainloader_ = iter(trainloader)
            data, lbl = next(trainloader_)
        data = data.to(DEVICE)
        (ze, z, zq, pred), loss = vqvae.forward(data)
        loss.backward()
        loss_ += loss.item()

    # optimize params
    loss_ = loss_/GRAD_ACC_STEPS
    trainloss += loss_
    log_trainloss += loss_
    vqvae.optimize(GRADIENT_CLIP, new_lr=None, set_to_none=True)

    # print info.
    if itr%PRINT_INTERVAL == 0:
        log_data = f"iteration: {itr}/{MAX_ITERS}, \ttrain loss: {log_trainloss/PRINT_INTERVAL}"
        print(log_data)
        log_trainloss = 0

# =============================================================
# END
# =============================================================