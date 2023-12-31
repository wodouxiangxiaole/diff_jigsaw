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

        # # init learnable embedding
        # self.num_queries = cfg.data.max_num_part
        # self.learnable_embedding = nn.Embedding(self.num_queries, self.model_channels)

        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': 7,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)

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
            cfg.model.num_dim + embedder_scale.out_dim, 
            self.model_channels
        )

        self.param_fc = nn.Linear(embedder_obj.out_dim, self.model_channels)
        self.pos_fc = nn.Linear(
            embedder_pos.out_dim + embedder_scale.out_dim, 
            self.model_channels
        )

        # Pos encoding for indicating the sequence. which part is for the reference
        self.pos_encoding = PositionalEncoding(self.model_channels)

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels // 2)
        self.output_linear3 = nn.Linear(self.model_channels // 2, self.out_channels)
        
        # # mlp out for translation N, 256 -> N, 3
        # self.mlp_out_trans = nn.Sequential(
        #     nn.Linear(self.model_channels, self.model_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.model_channels, self.model_channels // 2),
        #     nn.SiLU(),
        #     nn.Linear(self.model_channels // 2, 3),
        # )

        # # mlp out for rotation N, 256 -> N, 4
        # self.mlp_out_rot = nn.Sequential(
        #     nn.Linear(self.model_channels, self.model_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.model_channels, self.model_channels // 2),
        #     nn.SiLU(),
        #     nn.Linear(self.model_channels // 2, 4),
        # )


    def _gen_mask(self, L, N, B, mask):
        self_block = torch.ones(L, L, device=mask.device)  # Each L points should talk to each other
        self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]

        flattened_mask = mask.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)  # shape [B, N*L]
        flattened_mask = flattened_mask.unsqueeze(1)  # shape [B, 1, N*L]
        gen_mask = flattened_mask * flattened_mask.transpose(-1, -2)  # shape [B, N*L, N*L]
        return self_mask, gen_mask
    

    def _gen_cond(self, timesteps, x, xyz, latent, scale):
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)

        x = x.flatten(0, 1)  # (B*N, 7)

        xyz = xyz.flatten(0, 1)  # (B*N, L, 3)

        latent = latent.flatten(0, 1)  # (B*N, L, 64)

        scale = scale.flatten(0, 1)  # (B*N, 1)
        scale_emb = self.scale_embedding(scale).unsqueeze(1) # (B*N, 1, C)
        scale_emb = scale_emb.repeat(1, latent.shape[1], 1) # (B*N, L, C)

        latent = torch.cat((latent, scale_emb), dim=2) # (B*N, L, 64+C)

        shape_emb = self.shape_embedding(latent)
        xyz_pos_emb = self.pos_embedding(xyz)
        xyz_pos_emb = torch.cat((xyz_pos_emb, scale_emb), dim=2)

        x_emb = self.param_fc(self.param_embedding(x))
        xyz_pos_emb = self.pos_fc(xyz_pos_emb)
        return x_emb, shape_emb, xyz_pos_emb, time_emb


    def _out(self, data_emb, B, N, L):
        out = data_emb.reshape(B, N, L, self.model_channels)
        # Avg pooling
        out = out.mean(dim=2)
        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)
        return out_dec


    # def _out(self, data_emb, B, N, L):
    #     out = data_emb.reshape(B, N, L, self.model_channels)

    #     # Avg pooling
    #     out = out.mean(dim=2)

    #     trans = self.mlp_out_trans(out)
    #     rots = self.mlp_out_rot(out)

    #     return torch.cat([trans, rots], dim=-1)


    def _add_ref_part_emb(self, B, x_emb, ref_part):
        """
        x_emb: B, N, 256
        ref_part_valids: B, N
        """
        x_emb = x_emb.reshape(B, -1, self.model_channels)
        ref_part_emb = self.ref_part_emb.weight[0].repeat(B, x_emb.shape[1], 1)
        ref_part_emb[torch.arange(B), ref_part] = self.ref_part_emb.weight[1]

        x_emb = x_emb + ref_part_emb
        return x_emb.reshape(-1, self.model_channels)


    def forward(self, x, timesteps, latent, xyz, part_valids, scale, ref_part):
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

        x_emb, shape_emb, pos_emb, time_emb = self._gen_cond(timesteps, x, xyz, latent, scale)
        self_mask, gen_mask = self._gen_mask(L, N, B, part_valids)

        if self.cfg.model.ref_part:
            x_emb = self._add_ref_part_emb(B, x_emb, ref_part)

        x_emb = x_emb.reshape(B, N, 1, -1)
        x_emb = x_emb.repeat(1, 1, L, 1)
        condition_emb = shape_emb.reshape(B, N*L, -1) + \
                            pos_emb.reshape(B, N*L, -1) + time_emb 
        
        # B, N*L, C
        data_emb = x_emb.reshape(B, N*L, -1) 

        data_emb = data_emb + condition_emb

        for layer in self.transformer_layers:
            data_emb = layer(data_emb, self_mask, gen_mask)

        # data_emb (B, N*L, C)
        out_dec = self._out(data_emb, B, N, L)

        return out_dec

