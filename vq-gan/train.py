import math
import time
from pathlib import Path
import sys

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from model import VQGAN
from data import ImageSet, load_cifar, data_loaders
from utils import set_seed, save_checkpoint, load_checkpoint, write_pred, LossManager


__author__ = "__Girish_Hegde__"


# config file - (overrides the parameters given here)
CFG = None
if len(sys.argv) > 1: CFG = str(sys.argv[1])
# =============================================================
# Parameters
# =============================================================
# model
IN_CH = 3
DOWNSAMPLING_FACTOR = 4
HIDDEN_CH = 128
NUM_EMB = 256
# IN_CH = 3
# RES_LAYERS = 2
# HIDDEN_CH = 256
# NUM_EMB = 8*8*10

PERCEPTUAL_LOSS = False
BETA = 0.75
# dataset
TRAIN_IMG_DIR = '../../landscapes_256/train'
EVAL_IMG_DIR = '../../landscapes_256/test'
EXT = '.png'
# logging
LOGDIR = Path('./data/runs_128')
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
BATCH_SIZE = 16
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
trainset, evalset = ImageSet(TRAIN_IMG_DIR, EXT, 0.5), ImageSet(EVAL_IMG_DIR, EXT, 0.5)
trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True)
evalloader = DataLoader(evalset, BATCH_SIZE, shuffle=True)
# trainset, evalset = load_cifar()
# trainloader, evalloader = data_loaders(trainset, evalset, BATCH_SIZE)

# =============================================================
# Load Checkpoint
# =============================================================
net_ckpt, itr, best, kwargs = load_checkpoint(CKPT)

# =============================================================
# VQGAN(Model, Optimizer, Criterion) init and checkpoint load
# =============================================================
vqgan = VQGAN(
    IN_CH, DOWNSAMPLING_FACTOR, HIDDEN_CH, NUM_EMB, 
    PERCEPTUAL_LOSS, BETA, LR, DEVICE,
    net_ckpt, inference=False,
)
# vqgan = VQGAN(
#     IN_CH, RES_LAYERS, HIDDEN_CH, NUM_EMB, 
#     PERCEPTUAL_LOSS, BETA, LR, DEVICE,
#     net_ckpt, inference=False,
# )
# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
loss_manager = LossManager(
    'train_loss', 'val_loss', 'log_loss', 
    'train_recon_loss', 'val_recon_loss', 'log_recon_loss', 
    'train_disc_loss', 'val_disc_loss', 'log_disc_loss',
    metric='val_recon_loss',
    best=best,
)
vqgan.train()
vqgan.zero_grad(set_to_none=True)
trainloader_, evalloader_ = iter(trainloader), iter(evalloader)
print('Training ...')
start_time = time.perf_counter()
for itr in range(itr, MAX_ITERS + 1):
    # =============================================================
    # Validation
    # =============================================================
    if (itr%EVAL_INTERVAL == 0) or EVAL_ONLY:
        print('Evaluating ...')
        vqgan.eval()
        with torch.no_grad():
            for _ in tqdm(range(EVAL_ITERS)):
                try: data = next(evalloader_)
                except StopIteration:
                    evalloader_ = iter(evalloader)
                    data = next(evalloader_)
                data = data.to(DEVICE)
                (ze, z, zq, pred, lbl), (recon_loss, emb_loss, gan_loss, loss), disc_loss = vqgan.forward(data, backward=False)
                loss_manager.accumulate(
                    val_loss = loss.item(), val_recon_loss = recon_loss.item(), val_disc_loss = disc_loss.item(), 
                )
        vqgan.train()

        loss_manager.average(
            'val_loss', 'val_recon_loss', 'val_disc_loss', 
            'train_loss', 'train_recon_loss', 'train_disc_loss'
        )
        # =============================================================
        # Saving and Logging
        # =============================================================
        if EVAL_ONLY:
            log_data = loss_manager.get_str('val_loss', 'val_recon_loss', 'val_disc_loss') + f'\ttime: {(end_time - start_time)/60}M'
            print(f'{"-"*150}\n{log_data}\n{"-"*150}')
            break

        print('Saving checkpoint ...')
        ckpt_name = LOGDIR/'ckpt.pt' if not SAVE_EVERY else LOGDIR/f'ckpt_{itr}.pt'
        save_checkpoint(
            vqgan.get_ckpt(), itr, 
            loss_manager.losses['val_recon_loss'], loss_manager.losses['train_recon_loss'], loss_manager.best, 
            ckpt_name, 
            disc_loss = loss_manager.losses['val_disc_loss'],
        )

        if loss_manager.update_best():
            save_checkpoint(
                vqgan.get_ckpt(), itr, 
                loss_manager.losses['val_recon_loss'], loss_manager.losses['train_recon_loss'], loss_manager.best, 
                LOGDIR/'best.pt', 
                disc_loss = loss_manager.losses['val_disc_loss'],
            )
        
        write_pred(pred[:10], LOGDIR/'predictions', str(itr))

        logfile = LOGDIR/'log.txt'
        loss_str = loss_manager.get_str('val_loss', 'val_recon_loss', 'val_disc_loss', 'train_loss', 'train_recon_loss', 'train_disc_loss', spacer=' ')
        log_data = f"iteration: {itr}/{MAX_ITERS}, \t{loss_str} \tbest: {loss_manager.best}"
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')
        end_time = time.perf_counter()
        log_data = f'{log_data}, \ttime: {(end_time - start_time)/60}M'
        print(f'{"-"*150}\n{log_data}\n{"-"*150}')

        start_time = time.perf_counter()
        print('Training ...')

    # =============================================================
    # Training
    # =============================================================
    # forward, loss, backward with grad. accumulation
    for step in range(GRAD_ACC_STEPS):
        try: data = next(trainloader_)
        except StopIteration:
            trainloader_ = iter(trainloader)
            data = next(trainloader_)
        data = data.to(DEVICE)
        (ze, z, zq, pred, lbl), (recon_loss, emb_loss, gan_loss, loss), disc_loss = vqgan.forward(data)
        loss_manager.accumulate(
            train_loss = loss.item(), train_recon_loss = recon_loss.item(), train_disc_loss = disc_loss.item(), 
            log_loss = loss.item(), log_recon_loss = recon_loss.item(), log_disc_loss = disc_loss.item(), 
        )

    # optimize params
    vqgan.optimize(GRADIENT_CLIP, new_lr=None, set_to_none=True)

    # print info.
    if itr%PRINT_INTERVAL == 0:
        loss, recon_loss, disc_loss = loss_manager.average('log_loss', 'log_recon_loss', 'log_disc_loss')
        log_data = f"itr: {itr}/{MAX_ITERS}, \ttrain_loss: {loss}, \trecon_loss: {recon_loss}, \tdisc_loss: {disc_loss}"
        print(log_data)
# =============================================================
# END
# =============================================================