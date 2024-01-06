import torch.nn as nn
import torch
from torch.nn import functional as F
from jigsaw.model.model_utils import (
    timestep_embedding,
    PositionalEncoding,
    EmbedderNerf
)
from jigsaw.model.transformer import EncoderLayer


class DiffModel(nn.Module):
    """
    Transformer-based diffusion model
    """

    def __init__(self, cfg):
        super(DiffModel, self).__init__()
        self.cfg = cfg

        self.model_channels = cfg.model.embed_dim
        self.out_channels = cfg.model.out_channels
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads

        if cfg.model.ref_part:
            self.ref_part_emb = nn.Embedding(2, cfg.model.embed_dim)

        self.activation = nn.SiLU()
        self.transformer_layers = nn.ModuleList(
            [EncoderLayer(self.model_channels, self.num_heads, 0.1, self.activation) for x in range(self.num_layers)])

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.model_channels),
        )

        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_trans = EmbedderNerf(**embed_kwargs)
        self.trans_emb = lambda x, eo=embedder_trans: eo.embed(x)

        embed_kwargs['input_dims'] = 4
        embedder_rots = EmbedderNerf(**embed_kwargs)
        self.rots_emb = lambda x, eo=embedder_rots: eo.embed(x)


        embed_pos_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        # Pos embedding for positions of points xyz
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)

        embed_scale_kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_scale = EmbedderNerf(**embed_scale_kwargs)
        self.scale_embedding = lambda x, eo=embedder_scale: eo.embed(x)

        self.shape_embedding = nn.Linear(
            cfg.model.num_dim + embedder_scale.out_dim + embedder_rots.out_dim + embedder_trans.out_dim, 
            self.model_channels
        )

        self.pos_fc = nn.Linear(
            embedder_pos.out_dim + embedder_scale.out_dim + embedder_rots.out_dim + embedder_trans.out_dim,
            self.model_channels
        )

        # Pos encoding for indicating the sequence. which part is for the reference
        self.pos_encoding = PositionalEncoding(self.model_channels)

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels // 2)
        self.output_linear3 = nn.Linear(self.model_channels // 2, self.out_channels)
    

    def _gen_mask(self, L, N, B, mask):
        self_block = torch.ones(L, L, device=mask.device)  # Each L points should talk to each other
        self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]

        flattened_mask = mask.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)  # shape [B, N*L]
        flattened_mask = flattened_mask.unsqueeze(1)  # shape [B, 1, N*L]
        gen_mask = flattened_mask * flattened_mask.transpose(-1, -2)  # shape [B, N*L, N*L]
        return self_mask, gen_mask
    

    def _gen_cond(self, timesteps, x, xyz, latent, scale):
        """
        Generate the condition for the transformer
        timesteps: B, 1
        x: B, N, 7
        xyz: B, N, L, 3
        latent: B, N, L, 64
        """
        B, N, L, _ = latent.shape

        x = x.flatten(0, 1)  # (B*N, 7)
        trans_param = x[..., :3]
        rot_param = x[..., 3:]

        trans_emb = self.trans_emb(trans_param).unsqueeze(1).repeat(1, L, 1) # (B*N, L, C_trans)
        rot_emb = self.rots_emb(rot_param).unsqueeze(1).repeat(1, L, 1) # (B*N, L, C_rot)
        scale = scale.flatten(0, 1)  # (B*N, 1)
        scale_emb = self.scale_embedding(scale).unsqueeze(1).repeat(1, L, 1) # (B*N, 1, C)
        
        # Generate shape embedding [B*N, L, C_latent + C_scale + C_rot + C_trans]
        latent = latent.flatten(0, 1)  # (B*N, L, 64)
        latent = torch.cat((latent, scale_emb, rot_emb, trans_emb), dim=2) # (B*N, L, C)
        shape_emb = self.shape_embedding(latent)

        # Generate position embedding [B*N, L, C_pos + C_scale + C_rot + C_trans]
        xyz = xyz.flatten(0, 1)  # (B*N, L, 3)
        xyz_pos_emb = self.pos_embedding(xyz)
        xyz_pos_emb = torch.cat((xyz_pos_emb, scale_emb, rot_emb, trans_emb), dim=2)
        xyz_pos_emb = self.pos_fc(xyz_pos_emb)

        # Generate time embedding [B, N, C]
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)

        return shape_emb, xyz_pos_emb, time_emb


    def _out(self, data_emb, B, N, L):
        out = data_emb.reshape(B, N, L, self.model_channels)
        # Avg pooling
        out = out.mean(dim=2)
        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)
        return out_dec


    def _add_ref_part_emb(self, B, N, L, shape_emb, ref_part):
        """
        shape_emb: B*N, L 256
        ref_part_valids: B, N
        """
        shape_emb = shape_emb.reshape(B, N, L, -1)

        ref_part_emb = self.ref_part_emb.weight[0].repeat(B, N, 1)
        ref_part_emb[torch.arange(B), ref_part] = self.ref_part_emb.weight[1]

        shape_emb = shape_emb + ref_part_emb.unsqueeze(2).repeat(1, 1, L, 1)
        return shape_emb.reshape(B*N, L, self.model_channels)


    def forward(self, trans, rots, timesteps, latent, xyz, part_valids, scale, ref_part):
        """
        Latent already transform

        forward pass 
        x : (B, N, 3)
        timesteps : (B, 1)
        latent : (B, N, L, 4)
        xyz : (B, N, L, 3)
        mask: B, N
        scale: B, N
        """

        B, N, L, _ = latent.shape
        
        x = torch.cat([trans, rots], dim=-1)

        shape_emb, pos_emb, time_emb = self._gen_cond(timesteps, x, xyz, latent, scale)
        self_mask, gen_mask = self._gen_mask(L, N, B, part_valids)

        if self.cfg.model.ref_part:
            shape_emb = self._add_ref_part_emb(B, N, L, shape_emb, ref_part)

        data_emb = shape_emb.reshape(B, N*L, -1) + \
                    pos_emb.reshape(B, N*L, -1) + time_emb 

        for layer in self.transformer_layers:
            data_emb = layer(data_emb, self_mask, gen_mask)

        # data_emb (B, N*L, C)
        out_dec = self._out(data_emb, B, N, L)

        return out_dec

