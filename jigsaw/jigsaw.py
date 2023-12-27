import torch
from torch.nn import functional as F
import lightning.pytorch as pl
import hydra
from jigsaw.model.diffusion import DiffModel
# from jigsaw.model.diffusion_cat import DiffModel
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm
from chamferdist import ChamferDistance
from jigsaw.evaluation.jigsaw_evaluator import (
    calc_part_acc,
    trans_metrics,
    randn_tensor,
    calc_cd
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

        if cfg.model.scheduler == "ddpm":
            if cfg.model.DDPM_BETA_SCHEDULE == "linear":
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                    beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                    prediction_type=cfg.model.PREDICT_TYPE,
                    beta_start=cfg.model.BETA_START,
                    beta_end=cfg.model.BETA_END,
                    clip_sample=False,
                )
            else: 
                self.noise_scheduler = CustomDDPMScheduler(
                    num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                    beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                    prediction_type=cfg.model.PREDICT_TYPE,
                    beta_start=cfg.model.BETA_START,
                    beta_end=cfg.model.BETA_END,
                    clip_sample=False,
                    timestep_spacing=self.cfg.model.timestep_spacing
                )
        elif cfg.model.scheduler == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
                beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
                prediction_type=cfg.model.PREDICT_TYPE,
                beta_start=cfg.model.BETA_START,
                beta_end=cfg.model.BETA_END,
                clip_sample=False,
            )
        else:
            raise NotImplementedError

        self.cd_loss = ChamferDistance()
        self.num_points = cfg.model.num_point
        self.num_channels = cfg.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.model.num_inference_steps
        )

        self.acc_list = []
        self.rmse_t_list = []
        self.cd_list = []

        self.metric = ChamferDistance()


    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)
        
        return part_pcs


    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        latent = data_dict['latent']
        xyz = data_dict["xyz"]
        ref_part = data_dict["ref_part"]

        noise = torch.randn(gt_trans.shape, device=self.device)

        B, P, _, _ = latent.shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans = self.noise_scheduler.add_noise(gt_trans, noise, timesteps)
        if self.cfg.model.ref_part:
            noisy_trans[torch.arange(B), ref_part] = gt_trans[torch.arange(B), ref_part]


        pred_noise = self.diffusion(noisy_trans, timesteps, 
                                    latent, xyz, 
                                    data_dict['part_valids'],
                                    data_dict["part_scale"],
                                    ref_part)

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise
        }

        return output_dict


    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']
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
    

    def on_after_backward(self):
        """Called after the backward pass but before the optimizer step."""
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"Parameter not used in forward pass: {name}")

                
    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"val_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        
        latent = data_dict['latent']
        xyz = data_dict["xyz"]

        gt_trans = data_dict['part_trans']
        noise_trans = randn_tensor(gt_trans.shape, device=self.device)

        ref_part = data_dict["ref_part"]
        B, P, _, _ = latent.shape

        all_pred_translation = []

        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = t.reshape(-1).repeat(len(noise_trans)).cuda()
            
            if t == self.cfg.model.reset_timestep:
                noise_trans = randn_tensor(gt_trans.shape, device=self.device)

            pred_noise = self.diffusion(noise_trans, timesteps, 
                                        latent, xyz, 
                                        data_dict["part_valids"],
                                        data_dict["part_scale"],
                                        ref_part)
            
            vNext = self.noise_scheduler.step(pred_noise, t, noise_trans).prev_sample
            noise_trans = vNext

            if self.cfg.model.ref_part:
                noise_trans[torch.arange(B), ref_part] = gt_trans[torch.arange(B), ref_part]        

            all_pred_translation.append(noise_trans.detach().cpu().numpy())

        pts = data_dict['part_pcs']
        pred_translation = noise_trans[..., :3]

        part_valids = data_dict['part_valids']

        # for i in range(B):
        #     scale = part_valids[i]
        #     for j in range(scale.shape[0]):
        #         if scale[j] < 0.03:
        #             part_valids[i][j] = 0

        acc = calc_part_acc(pts, trans1=pred_translation, trans2=gt_trans, 
                            valids=part_valids, 
                            chamfer_distance=self.metric)

        rmse_t = trans_metrics(pred_translation, gt_trans,  part_valids, 'rmse')

        cd = calc_cd(
            pts, pred_translation, gt_trans,
            part_valids, 
            self.metric
        )

        self.acc_list.append(torch.mean(acc))
        self.rmse_t_list.append(torch.mean(rmse_t))
        self.cd_list.append(torch.mean(cd))

        return np.stack(all_pred_translation, axis=0), acc, rmse_t, cd
    

    def on_validation_epoch_end(self):
        total_acc = torch.mean(torch.stack(self.acc_list))
        total_rmse_t = torch.mean(torch.stack(self.rmse_t_list))
        total_cd = torch.mean(torch.stack(self.cd_list))
        self.log(f"eval/part_acc", total_acc)
        self.log(f"eval/rmse_t", total_rmse_t)
        self.log(f"eval/cd", total_cd)
        self.acc_list = []
        self.rmse_t_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_cd


    def test_step(self, data_dict, idx):
        predict_translation, acc, _, _ = self.validation_step(data_dict, idx)

        T, B, _, _ = predict_translation.shape

        for i in range(B):
            save_dir = os.path.join(
                self.cfg.experiment_output_path,
                "inference", 
                self.cfg.inference_dir, 
                str(data_dict['data_id'][i].item())
            )
            os.makedirs(save_dir, exist_ok=True)
            c_translation = predict_translation[:, i, ...]
            mask = data_dict["part_valids"][i] == 1
            c_translation = c_translation[:, mask.cpu().numpy(), ...]
            np.save(os.path.join(save_dir, f"predict_{acc[i]}.npy"), c_translation)
            gt_translation = data_dict["part_trans"][i][mask]
            
            np.save(os.path.join(
                save_dir, "gt.npy"),
                gt_translation.cpu().numpy()
            )
            with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
                f.write(data_dict["mesh_file_path"][i])


    def on_test_epoch_end(self):
        total_acc, total_rmse_t, total_cd  = self.on_validation_epoch_end()
        # self.print("--------------Metrics on Test Set--------------")
        # self.print("test/part_acc", total_acc)
        # self.print("test/rmse_t", total_rmse_t)
        # self.print("------------------------------------------------")
        

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
