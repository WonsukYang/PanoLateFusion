from .encoder import Encoder
from .decoder import Decoder

import torch.nn as nn

class AVSDModel(nn.Module):
    @staticmethod
    def add_cmdline_args(parser):
        parser = Encoder.add_cmdline_args(parser)
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args, self.encoder)

    def forward(self, batch):
        enc_out = self.encoder(batch)
        dec_out = self.decoder(batch, enc_out)

        return dec_out

    def evaluate(self):
        self.encoder.evaluate()