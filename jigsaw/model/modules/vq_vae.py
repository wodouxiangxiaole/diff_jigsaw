import torch
import torch.nn as nn
import numpy as np
from jigsaw.model.modules.pn2 import PN2
from jigsaw.model.modules.utils.quantizer import VectorQuantizer
from chamferdist import ChamferDistance
import os
from pytorch3d import transforms
import matplotlib.cm as cm


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super(VQVAE, self).__init__()
        self.pn2 = PN2(cfg)
        self.cfg = cfg
        self.encoder = self.pn2.encode
        self.vector_quantization = VectorQuantizer(
            cfg.ae.n_embeddings,
            cfg.ae.embedding_dim,
            cfg.ae.beta
        )
        self.decoder = self.pn2.decode


    def forward(self, part_pcs, verbose=False):
        """
        part_pcs = (batch, L, C)

        x.shape = (batch, C, L)
        """
        x = part_pcs.permute(0, 2, 1)

        z_e, xyz = self.encoder(x)

        B, L, C = z_e.shape

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e.reshape(B, 4 * L, -1)
        )

        z_q = z_q.reshape(B, L, -1)

        x_hat = self.decoder(z_q)

        output_dict = {
            'embedding_loss': embedding_loss,
            'pc_offset': x_hat,
            'perplexity': perplexity,
            "xyz": xyz,
            "z_q": z_q
        }

        return output_dict
    

    def encode(self, part_pcs):
        """
        x.shape = (batch, C, L)
        """

        x = part_pcs.permute(0, 2, 1)
        z_e, xyz = self.encoder(x)
        
        B, L, C = z_e.shape
        _, z_q, _, _, _ = self.vector_quantization(
            z_e.reshape(B, 4 * L, -1)
        )
        z_q = z_q.reshape(B, L, -1)
        output_dict = {
            "z_q": z_q,
            "xyz": xyz
        }

        return output_dict
    
    def decode(self, z_q):
        x_hat = self.decoder(z_q)
        return x_hat    