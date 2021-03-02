import os
import json
import pickle

from tqdm import tqdm
from collections import defaultdict

import torch
from torch import Tensor
from torch.utils.data import Dataset

import torch.nn.functional as F

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data

def load_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def mask_tensor(x: Tensor):
    '''Returns a copy of input tensor, 
    replacing any non-zero entry to one
    '''
    mask = x.detach().clone()
    mask[mask != 0] = 1
    return mask 

class PanoAVQADataset(Dataset):
    '''
    Given: 
        - list of questions (unique) (*) and their lengths (as tokens)
        - list of answers (unique) (**) and their lengths (as tokens)
        - list containing: 
                        - indices of options (**)
                        - index of answer wrt (**)
                        - index of question (*)
                        - metadata (split, clip_id, type)
        - audio features
        - visual features  
    '''

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group('Dataset related arguments')
        parser.add_argument('-visual_feat_dir', default='data/visual_feat', type=str)
        parser.add_argument('-audio_feat_dir', default='data/audio_feat_old', type=str)
        parser.add_argument('-qa_data_dir', default='data', type=str)
        parser.add_argument('-num_workers', default=2, type=int)
        parser.add_argument('-ans_list_fname', default='data/answers_{}.pkl', type=str)
        parser.add_argument('-ques_list_fname', default='data/questions_{}.pkl', type=str)
        parser.add_argument('-vis_mode', default='i3d_center', choices=['i3d_center',
                                                                        'i3d_er'], type=str)
        parser.add_argument('-aud_mode', default='top_3_feat', choices=['top_3_feat',
                                                                        'top_5_feat',
                                                                        'orig_feat',
                                                                        'pool_2_feat',
                                                                        'pool_4_feat'], type=str)
        parser.add_argument('-channel_opt', default='stereo', choices=['mono',
                                                                       'stereo'], type=str)
        return parser

    def __init__(self, args, split):
        super().__init__()
        
        assert split in ('train', 'val', 'test')

        self.args = args
        self.split = split

        qa_data = load_json(os.path.join(args.qa_data_dir, f"{split}.json"))
        self.data = self.organize_data(qa_data)        

    def __len__(self):
        return len(self.data['ans_idx'])

    def __getitem__(self, idx):
        item = {}

        item['ques'] = self.data['ques'][idx]
        
        item['ques_type'] = self.data['ques_type'][idx]
        
        item['ans_idx'] = self.data['ans_idx'][idx]
        item['options'] = self.data['options'][idx]
        
        if self.args.lang_model == 'rnn':
            item['ques_len'] = self.data['ques_len'][idx]
            item['opt_len'] = self.data['opt_len'][idx]
        else:
            item['ques_mask'] = self.data['ques_mask'][idx]
            item['opt_mask'] = self.data['opt_mask'][idx]
        
        item['aud_feat'] = self.data['aud_feat'][idx]
        item['vis_feat'] = self.data['vis_feat'][idx]
        
        if self.split == 'test':
            item['clip_id'] = self.data['clip_id'][idx]
            
        return item

    def organize_data(self, data):
        '''Organize data in so much that no further processing is necessary on retrieval
        BERT specific: ques_mask, opt_mask  
        RNN specific: ques_len, opt_len
        '''

        question_list = load_pickle(os.path.join(self.args.ques_list_fname \
                                            .format(self.args.lang_model)))
        answer_list = load_pickle(os.path.join(self.args.ans_list_fname \
                                            .format(self.args.lang_model))) 

        if self.args.lang_model == 'rnn':
            question_len_list = load_pickle(os.path.join(self.args.ques_list_fname \
                                                .format(self.args.lang_model + '_len')))
            answer_len_list = load_pickle(os.path.join(self.args.ans_list_fname \
                                                    .format(self.args.lang_model + '_len')))
            # turn question_list and answer_list into torch.Tensor
            # to allow list indexing
            question_list = torch.Tensor(question_list).long()
            answer_list = torch.Tensor(answer_list).long()

        res = defaultdict(list)

        for d in tqdm(data):
        
            # process option data
            options = [answer_list[i].squeeze(0) for i in d['ans_idx']]
            if self.args.lang_model == 'bert':
                res['opt_mask'].append(torch.stack([mask_tensor(opt) for opt in options]))
            else:
                ### remove extraneous padding using ans_idx?
                opt_len = [min(answer_len_list[i], self.args.max_ans_len) for i in d['ans_idx']]
                if 0 in opt_len:
                    continue
                res['opt_len'].append(opt_len)
            res['options'].append(torch.stack(options))    ### torch.index_select?

            # process question data
            question = question_list[d['ques_idx']].squeeze(0)
            res['ques'].append(question)
            if self.args.lang_model == 'bert':
                res['ques_mask'].append(mask_tensor(question))
            else:
                res['ques_len'].append(min(question_len_list[d['ques_idx']], self.args.max_ques_len))
            
            clip_id = d['clip']
            if self.split == 'test':
                res['clip_id'].append(clip_id)

            res['ques_type'].append(0 if d['type'] == 's' else 1)
            res['ans_idx'].append(d['gt_index'])

            # load audio feature
            aud_feat = load_pickle(f"{self.args.audio_feat_dir}/{self.args.aud_mode}/{clip_id}.pkl")
            if self.args.channel_opt == 'stereo':
                aud_feat = torch.Tensor(aud_feat[1:]) \
                                    .permute(1, 0, 2) \
                                    .reshape(-1, self.args.audio_feature_size)
                aud_feat = torch.mean(aud_feat, dim=0, keepdim=True)
                aud_feat = F.normalize(aud_feat, dim=1, p=2).view(self.args.audio_feature_size, )    
            else:
                raise NotImplementedError("Dataloader does not support mono channel option")
            
            res['aud_feat'].append(aud_feat)

            # load visual feature
            vis_feat = load_pickle(f"{self.args.visual_feat_dir}/{self.args.vis_mode}/{clip_id}.pkl")
            vis_feat = torch.Tensor(vis_feat).view(-1, self.args.visual_feature_size)
            vis_feat = F.normalize(vis_feat, dim=1, p=2).view(self.args.visual_feature_size, )
            
            res['vis_feat'].append(vis_feat)
            
        return res
    
    def collate_fn(self, batch):
        merged = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged:
            if key == 'clip_id':
                out[key] = merged[key]
            elif key in ('ques_type', 'ans_idx', 'ques_len', 'opt_len'):
                out[key] = torch.Tensor(merged[key]).long()
            else:
                out[key] = torch.stack(merged[key], 0)

        if self.args.lang_model == 'rnn':
            out['ques'] = out['ques'][:, :torch.max(out['ques_len'])]
            out['options'] = out['options'][:, :, :torch.max(out['opt_len'])]
        return out

if __name__ == "__main__":
    # PanoAVQA dataset test code goes here
    import argparse
    from pprint import pprint
    from model import AVSDModel
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser = PanoAVQADataset.add_cmdline_args(parser)
    parser = AVSDModel.add_cmdline_args(parser)
    args = parser.parse_args()

    setattr(args, 'dropout', 0.5)
    setattr(args, 'weight_init', 'xavier')
    setattr(args, 'max_ques_len', 20)
    setattr(args, 'lang_model', 'rnn')
    setattr(args, 'audio_feature_size', 4096)
    setattr(args, 'visual_feature_size', 4096)
    setattr(args, 'max_ans_len', 8)
    setattr(args, 'vocab_size', 1807)
    print("Loading PanoAVQA dataset...")
    dataset = PanoAVQADataset(args, 'val')
    
    print("\nTraining...\n")
    # dataloader = DataLoader(dataset, 
    #                     batch_size=8, 
    #                     shuffle=True,
    #                     collate_fn=dataset.collate_fn)

    model = AVSDModel(args)

    print(model)