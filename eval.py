import os
import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import PanoAVQADataset
from model import AVSDModel
from utils.eval_utils import process_results, scores_to_ranks, get_gt_ranks

parser = argparse.ArgumentParser()
parser = PanoAVQADataset.add_cmdline_args(parser)
parser = AVSDModel.add_cmdline_args(parser)

## evaluation related arguments
parser.add_argument_group('Evaluation related arguments')
parser.add_argument('-load_path', default='./checkpoint/rnn_model_final.pth')
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-seed', default=1234, type=int)
parser.add_argument('-gpuid', default=0, nargs='+', type=int)
parser.add_argument('-save_result', action='store_true')
parser.add_argument('-save_path', default='ranks.json', type=str)
args = parser.parse_args()

# load useful argument (required for running decoder)
setattr(args, 'max_ans_len', 8)
setattr(args, 'weight_init', 'xavier')
setattr(args, 'max_ques_len', 20)
setattr(args, 'vocab_size', 1807)
torch.manual_seed(args.seed)

if args.gpuid >= 0:
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.gpuid)

# load checkpoint
checkpoint = torch.load(args.load_path)
model_args = checkpoint['model_args']

dataset = PanoAVQADataset(args, 'test')
dataloader = DataLoader(dataset,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

model = AVSDModel(model_args)

print("Loading model...")

### only for bert maybe? ###
# from collections import OrderedDict 
# state_dict = OrderedDict()
# for k, v in checkpoint['model'].items():
#     if k.startswith('module.'):
#         _k = k[7:]
#     state_dict[_k] = v

model.load_state_dict(checkpoint['model'])
if args.gpuid >= 0:
    model.cuda()

print("Evaluation staring at : {}".format(
    datetime.strftime(datetime.utcnow(),
                    '%d-%b-%Y-%H:%M:%S'))
)

# evaluate model
results = []
question_types = []
with torch.no_grad():
    model.evaluate()
    # model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        question_types.append(batch['ques_type']) # ques_type must be a tensor
        output = model(batch)
        loss = F.cross_entropy(output, batch['ans_idx'])
        print(loss.item())
        ranks = scores_to_ranks(output.data)
        gt_ranks = get_gt_ranks(ranks, batch['ans_idx'].data)
        results.append(gt_ranks)

    question_types = torch.cat(question_types, 0)
    results = torch.cat(results, 0) # create torch.Tensor from a list of gt_ranks
    process_results(results, question_types)

if args.save_result:
    print("Saving results on {}".format(args.save_path))
    with open(args.save_path, 'w') as f:
        json.dump(results, f)
