import torch
import torch.nn as nn
import torch.nn.init as init

from utils import DynamicRNN
from transformers import BertModel

class Encoder(nn.Module):

    @staticmethod
    def add_cmdline_args(parser):
        parser.add_argument_group("Arguments related to encoder")
        parser.add_argument("-visual_feature_size", default=4096, type=int)
        parser.add_argument("-audio_feature_size", default=4096, type=int)
        parser.add_argument("-embed_size", default=300, type=int)
        parser.add_argument("-dropout", default=0.5, type=int)
        parser.add_argument("-num_layers", default=2, type=int)
        parser.add_argument("-rnn_hidden_size", default=768, type=int)
        parser.add_argument('-lang_model', default='bert', choices=['bert', 'rnn'], type=str)

        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.dropout = nn.Dropout(p=args.dropout)

        if self.args.lang_model == 'bert':
            self.ques_net = BertModel.from_pretrained("bert-base-uncased",
                                                        return_dict=True)
        else:
            self.word_embed = nn.Embedding(args.vocab_size, 
                                        args.embed_size,
                                        padding_idx=0)
            self.ques_net = DynamicRNN(nn.LSTM(args.embed_size,
                                   args.rnn_hidden_size,
                                   args.num_layers,
                                   batch_first=True,
                                   dropout=args.dropout))

        fusion_size = args.audio_feature_size \
                  + args.visual_feature_size \
                  + args.rnn_hidden_size 
        
        self.fusion = nn.Linear(fusion_size, args.rnn_hidden_size)
        
        self.weight_init()

    def weight_init(self):
        if self.args.weight_init == 'xavier':
            init.xavier_uniform_(self.fusion.weight.data)
        elif self.args.weight_init == 'kaiming':
            init.kaiming_uniform_(self.fusion.weight.data)
        init.zeros_(self.fusion.bias.data)

    def forward(self, batch):
        # batch_size * 1 * aud_feat_size
        audio = batch['aud_feat']

        # batch_size * vid_feat_size
        video = batch['vis_feat']
        
        # batch_size * max_ques_len
        ques = batch['ques']
                
        if self.args.lang_model == 'bert':
            ques_embed = self.ques_net(input_ids=ques, 
                                       attention_mask=batch['ques_mask']).last_hidden_state
            ques_embed = ques_embed[:, 0] # output of [CLS] token
        else: 
            ques_embed = self.word_embed(ques)
            ques_embed = self.ques_net(ques_embed, batch['ques_len'])

        fused_vector = torch.cat((audio, video, ques_embed), 1)

        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector))

        return fused_embedding

    def evaluate(self):
        self.dropout.eval()
        