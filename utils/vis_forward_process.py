import hydra
import diffusers
from jigsaw.dataset.dataset import build_geometry_dataloader
from jigsaw.model.custom_diffusers import CustomDDPMScheduler
from tqdm import tqdm
import torch
import numpy as np
import os


@hydra.main(config_path="../config", config_name="global_config")
def main(cfg):
    cfg.data.batch_size = 1
    cfg.data.val_batch_size = 1
    train_loader, val_loader = build_geometry_dataloader(cfg)

    noise_scheduler = CustomDDPMScheduler(
        num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
        beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
        prediction_type=cfg.model.PREDICT_TYPE,
        beta_start=cfg.model.BETA_START,
        beta_end=cfg.model.BETA_END,
        clip_sample=False,
    )

    for data_dict in train_loader:
        gt_trans = data_dict["part_trans"][0].cuda()
        gt_rots = data_dict["part_rots"][0].cuda()
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)

        data_id = str(data_dict["data_id"].item())
        num_parts = data_dict["num_parts"]


        if data_id != "120":
            continue
        forward_translation = []
        for t in range(1000):
            t = torch.tensor(t).cuda()
            
            noise = torch.randn(size=gt_trans_and_rots.shape, device=gt_trans_and_rots.device)

            timesteps = t.reshape(-1).repeat(gt_trans_and_rots.shape[0]).cuda()
            noisy_trans = noise_scheduler.add_noise(gt_trans_and_rots, noise, timesteps)
            forward_translation.append(noisy_trans[:num_parts].detach().cpu().numpy())
        
        save_dir = f"{cfg.save_dir}/{data_id}"
        
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/translation.npy", np.array(forward_translation))      
        with open(os.path.join(save_dir, "mesh_file_path.txt"), "w") as f:
            f.write(data_dict["mesh_file_path"][0])  
        np.save(f"{save_dir}/gt.npy", gt_trans_and_rots.detach().cpu().numpy())
        forward_translation = []

    

if __name__ == "__main__":
    main()