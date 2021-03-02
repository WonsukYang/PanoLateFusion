import torch
import torch.nn as nn
import torch.nn.functional as feature

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class DynamicRNN(nn.Module):

    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(sorted_seq_input,
                                                lengths=sorted_len,
                                                batch_first=True)

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
        else:
            hx = None
        _, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        rnn_output = h_n[-1].index_select(dim=0, index=bwd_order)
        return rnn_output

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        if isinstance(sorted_len, Variable):
            sorted_len = sorted_len.data
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
