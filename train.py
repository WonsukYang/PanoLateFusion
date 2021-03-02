import argparse
import datetime
import math
import random
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AVSDModel
from dataloader import PanoAVQADataset

parser = argparse.ArgumentParser()
parser = PanoAVQADataset.add_cmdline_args(parser)
parser = AVSDModel.add_cmdline_args(parser)

parser.add_argument_group('Training related arguments')
parser.add_argument('-num_epochs', default=10, type=int, help="Epochs")
parser.add_argument('-batch_size', default=128, type=int, help="Batch size")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-lr_decay_rate', default=0.9997592083, type=float, help="Decay for lr")
parser.add_argument('-min_lr', default=5e-5, type=float, help="Minimum learning rate")
parser.add_argument('-weight_init', default='xavier', choices=['xavier', 'kaiming'], help="Weight initialization")
parser.add_argument('-weight_decay', default=0.00075, help="Weight decay for l2 regularization")
parser.add_argument('-gpuid', default=0, nargs='+', type=int, help='GPU id to use')
parser.add_argument('-cuda', default=True, type=bool, help='Using CUDA')
parser.add_argument('-seed', default=1234, type=int, help="Random seed")

parser.add_argument_group('Checkpoint related arguments')
parser.add_argument('-load_path', default='', type=str, help="Loading path")
parser.add_argument('-save_path', default='checkpoint/', type=str, help="Path to save checkpoints")
parser.add_argument('-log_step', default=500, type=int)
parser.add_argument('-save_step', default=10, type=int)
parser.add_argument('-log_dir', default='runs/', type=str, help="Path to save tensorboard logs")

args = parser.parse_args()

# set tensorboard
writer = SummaryWriter(log_dir=args.log_dir)

# set directory for checkpoints
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    args.save_path += start_time

# set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0"
else:
    device = "cpu"

# Load useful arguments
setattr(args, "max_ques_len", 20)
setattr(args, "max_ans_len", 8)
setattr(args, "vocab_size", 1807)
setattr(args, "device", device)
model_args = args

# read saved model and args
if args.load_path != '':
    components = torch.load(args.load_path)
    model_args = components['model_args']

#### Load dataset

dataset = PanoAVQADataset(model_args, split='train')
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate_fn)
                        
dataset_val = PanoAVQADataset(model_args, split='val')
dataloader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=dataset_val.collate_fn)
#### Load model
model = AVSDModel(model_args)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)

setattr(args, "iter_per_epochs", math.ceil(len(dataset) / args.batch_size))

if args.load_path != '':
    model.load_state_dict(components['model'])

if args.cuda and torch.cuda.is_available():
    if torch.cuda.device_count() > 1 and isinstance(args.cuda, list):
        print("Setting up for multi-gpu training...")
        model = nn.DataParallel(model, device_ids=args.gpuid)
    model = model.cuda()

os.makedirs(args.save_path, exist_ok=True)

# print arguments 
for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

running_loss = 0.0
train_start = datetime.datetime.utcnow()
print('Training starts: {}' \
    .format(datetime.datetime.strftime(train_start,
            '%d-%b-%Y-%H:%M:%S')))

log_loss = []
for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
    model.train()
    for i, batch in enumerate(tqdm(dataloader, desc="Dataloader")):
        optimizer.zero_grad()
        if args.cuda:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
    
        output = model(batch)

        cur_loss = criterion(output, batch['ans_idx'])
        cur_loss.backward()

        optimizer.step()

        train_loss = cur_loss.item()
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.item()
        else:
            running_loss = cur_loss.item()

        # if optimizer.param_groups[0]['lr'] > args.min_lr:
        #     scheduler.step()

        if i % args.log_step == 0:
            ### validation step ###
            
            # model.eval()
            model.evaluate()
            with torch.no_grad():
                validation_losses = []
                for _, val_batch in enumerate(dataloader_val):
                    if args.cuda:
                        for key in val_batch:
                            if isinstance(val_batch[key], torch.Tensor):
                                val_batch[key] = val_batch[key].cuda()
                    
                    types = val_batch['ques_type']
                    val_output = model(val_batch)

                    cur_loss = criterion(val_output, val_batch['ans_idx'])
                    validation_losses.append(cur_loss.item())

                validation_loss = np.mean(validation_losses)
                iteration = (epoch - 1) * args.iter_per_epochs + i

                writer.add_scalar('val_loss', validation_loss, iteration)
                writer.add_scalar('train_loss', train_loss, iteration)
                writer.add_scalar('running_loss', running_loss, iteration)
                
                log_loss.append((epoch,
                                iteration,
                                running_loss,
                                train_loss,
                                validation_loss,
                                optimizer.param_groups[0]['lr']))

    if epoch % args.save_step == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args
        }, os.path.join(args.save_path, '{}_model_epoch_{}.pth'.format(model_args.lang_model, epoch)))

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args
}, os.path.join(args.save_path, f'{model_args.lang_model}_model_final.pth'))
np.save(os.path.join(args.save_path, 'log_loss'), log_loss)
