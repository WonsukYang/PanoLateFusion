import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from utils import DynamicRNN

class Decoder(nn.Module):

    def __init__(self, args, encoder):
        super().__init__()
        self.args = args

        if args.lang_model == 'bert':
            # Few notes on option_net -
            # in original paper, decoder and encoder shares word_embedding
            # but each has its own rnn 
            # NOT SURE if the code below makes ques_net and option_net
            # to share the embedding layer...
            
            # self.option_net = BertModel.from_pretrained('bert-base-uncased',
            #                                             return_dict=True)
            # self.option_net.embeddings = encoder.ques_net.embeddings
            self.option_net = encoder.ques_net
        else:
            self.word_embed = encoder.word_embed
            self.option_net = DynamicRNN(nn.LSTM(args.embed_size, args.rnn_hidden_size,
                                                batch_first=True))
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch, enc_out):
        options = batch['options']
        batch_size, num_options, max_opt_len = options.size()
        scores = []

        if self.args.lang_model == 'rnn':
            options_len = batch['opt_len']
            options = options.contiguous().view(-1, num_options * max_opt_len)
            options = self.word_embed(options)
            options = options.view(batch_size, num_options, max_opt_len, -1)
        else:
            opt_mask = batch['opt_mask']
        
        for opt_id in range(num_options):

            if self.args.lang_model == 'rnn':
                opt = options[:, opt_id, :, :]

                opt_len = options_len[:, opt_id]
                opt_embed = self.option_net(opt, opt_len)
            else:
                opt = options[:, opt_id, :]
                mask = opt_mask[:, opt_id, :]
                opt_embed = self.option_net(input_ids=opt,
                                            attention_mask=mask).last_hidden_state
                opt_embed = opt_embed[:, 0] 
            scores.append(torch.sum(opt_embed * enc_out, 1))        

        scores = torch.stack(scores, 1)
        return scores