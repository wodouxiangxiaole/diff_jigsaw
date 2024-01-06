import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from jigsaw.model.diffusion import DiffModel
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from chamferdist import ChamferDistance
from jigsaw.evaluation.jigsaw_evaluator import (
    calc_part_acc,
    trans_metrics,
    randn_tensor,
    rot_metrics,
)
import numpy as np
import os
from jigsaw.model.custom_diffusers import CustomDDPMScheduler
from pytorch3d import transforms



class Jigsaw3D(pl.LightningModule):
    def __init__(self, cfg):
        super(Jigsaw3D, self).__init__()
        self.cfg = cfg
        self.diffusion = DiffModel(cfg)
        self.save_hyperparameters()

        self.encoder = hydra.utils.instantiate(cfg.ae.ae_name, cfg)
        if cfg.model.scheduler == "ddpm":
            if cfg.model.DDPM_BETA_SCHEDULE == "linear":
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                    beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                    prediction_type=cfg.model.PREDICT_TYPE,
                    beta_start=cfg.model.BETA_START,
                    beta_end=cfg.model.BETA_END,
                )
            else: 
                self.noise_scheduler = CustomDDPMScheduler(
                    num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                    beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                    prediction_type=cfg.model.PREDICT_TYPE,
                    beta_start=cfg.model.BETA_START,
                    beta_end=cfg.model.BETA_END,
                    timestep_spacing=self.cfg.model.timestep_spacing
                )
        elif cfg.model.scheduler == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                prediction_type=cfg.model.PREDICT_TYPE,
                beta_start=cfg.model.BETA_START,
                beta_end=cfg.model.BETA_END,
            )
        else:
            raise NotImplementedError

        self.cd_loss = ChamferDistance()
        self.num_points = cfg.model.num_point
        self.num_channels = cfg.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.model.num_inference_steps
        )

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []

        self.metric = ChamferDistance()


    def _extract_feats(self, data_dict):
        B, P, N, C = data_dict["part_pcs"].shape
        part_pcs = data_dict["part_pcs"][data_dict['part_valids'].bool()]
        encoder_out = self.encoder.encode(part_pcs)
        latent = torch.zeros(B, P, self.num_points, self.num_channels, device=self.device)
        xyz = torch.zeros(B, P, self.num_points, 3, device=self.device)
        latent[data_dict['part_valids'].bool()] = encoder_out["z_q"]
        xyz[data_dict['part_valids'].bool()] = encoder_out["xyz"]

        return latent, xyz
        

    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        ref_part = data_dict["ref_part"]

        noise = torch.randn(gt_rots.shape, device=self.device)
        B, P, N, C = data_dict["part_pcs"].shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_rots = self.noise_scheduler.add_noise(gt_rots, noise, timesteps)

        if self.cfg.model.ref_part:
            noisy_rots[torch.arange(B), ref_part] = gt_rots[torch.arange(B), ref_part]

        latent, xyz = self._extract_feats(data_dict)

        pred_noise = self.diffusion(
            gt_trans,
            noisy_rots, 
            timesteps, 
            latent, 
            xyz, 
            data_dict['part_valids'],
            data_dict["part_scale"],
            ref_part
        )

        if self.cfg.model.PREDICT_TYPE == "epsilon":
            output_dict = {
                'predict': pred_noise,
                'gt': noise
            }
        elif self.cfg.model.PREDICT_TYPE == "sample":
            output_dict = {
                'predict': pred_noise,
                'gt': gt_rots
            }


        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['predict']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt']
        if self.cfg.model.weighted_small_pieces:
            weights = torch.where(data_dict["part_scale"][part_valids] < 0.03, 0.1, 1.0)
            mse_loss = F.mse_loss(pred_noise[part_valids], noise[part_valids], reduction='none')
            mse_loss = torch.mean(mse_loss * weights)
        elif self.cfg.model.ref_part:
            part_valids[torch.arange(part_valids.shape[0]), data_dict["ref_part"]] = False
            mse_loss = F.mse_loss(pred_noise[part_valids], noise[part_valids])
        else:
            mse_loss = F.mse_loss(pred_noise[part_valids], noise[part_valids])

        return {'mse_loss': mse_loss}


    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        
        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        
        return total_loss
    
    def _calc_val_loss(self, data_dict):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and logs
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)

    

    def _calc_metrics(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        ref_part = data_dict["ref_part"]

        B, P, N, C = data_dict["part_pcs"].shape
        noisy_rots = torch.randn(gt_rots.shape, device=self.device)


        if self.cfg.model.ref_part:
            noisy_rots[torch.arange(B), ref_part] = gt_rots[torch.arange(B), ref_part] 

        
        latent, xyz = self._extract_feats(data_dict)
        all_pred_trans_rots = []

        all_pred_trans_rots.append(torch.cat([gt_trans, noisy_rots], dim=-1).detach().cpu().numpy())

        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).repeat(len(noisy_rots)).cuda()

            pred_noise = self.diffusion(
                gt_trans,
                noisy_rots, 
                timesteps, 
                latent,
                xyz, 
                data_dict["part_valids"],
                data_dict["part_scale"],
                ref_part
            )
        
            noisy_rots = self.noise_scheduler.step(pred_noise, t, noisy_rots).prev_sample

            if self.cfg.model.ref_part:
                noisy_rots[torch.arange(B), ref_part] = gt_rots[torch.arange(B), ref_part]     
            

            all_pred_trans_rots.append(torch.cat([gt_trans, noisy_rots], dim=-1).detach().cpu().numpy())

        pts = data_dict['part_pcs']
        pred_translation = gt_trans
        pred_rots = noisy_rots

        acc = calc_part_acc(pts, trans1=pred_translation, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)

        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'], 'rmse')
        rmse_t = trans_metrics(pred_translation, gt_trans,  data_dict['part_valids'], 'rmse')

        self.acc_list.append(torch.mean(acc))
        self.rmse_r_list.append(torch.mean(rmse_r))
        self.rmse_t_list.append(torch.mean(rmse_t))
        return np.stack(all_pred_trans_rots, axis=0), acc


    def validation_step(self, data_dict, idx):

        self._calc_val_loss(data_dict)

        return self._calc_metrics(data_dict)
    

    def on_validation_epoch_end(self):
        total_acc = torch.mean(torch.stack(self.acc_list))
        total_rmse_t = torch.mean(torch.stack(self.rmse_t_list))
        total_rmse_r = torch.mean(torch.stack(self.rmse_r_list))
        self.log(f"eval/part_acc", total_acc)
        self.log(f"eval/rmse_t", total_rmse_t)
        self.log(f"eval/rmse_r", total_rmse_r)
        self.acc_list = []
        self.rmse_t_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_rmse_r


    def test_step(self, data_dict, idx):
        pred_trans_rots, acc = self.validation_step(data_dict, idx)

        T, B, _, _ = pred_trans_rots.shape

        for i in range(B):
            save_dir = os.path.join(
                self.cfg.experiment_output_path,
                "inference", 
                self.cfg.inference_dir, 
                str(data_dict['data_id'][i].item())
            )
            os.makedirs(save_dir, exist_ok=True)
            c_trans_rots = pred_trans_rots[:, i, ...]
            mask = data_dict["part_valids"][i] == 1
            c_trans_rots = c_trans_rots[:, mask.cpu().numpy(), ...]
            np.save(os.path.join(save_dir, f"predict_{acc[i]}.npy"), c_trans_rots)
            gt_transformation = torch.cat(
                [data_dict["part_trans"][i],
                    data_dict["part_rots"][i]], dim=-1
            )[mask]

            np.save(os.path.join(
                save_dir, "gt.npy"),
                gt_transformation.cpu().numpy()
            )
            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])


    def on_test_epoch_end(self):
        total_acc, total_rmse_t, total_rmse_r  = self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
