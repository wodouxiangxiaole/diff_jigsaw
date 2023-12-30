""" 
Dataset load latent
latent got from pre-trained auto-encoder

All of the data already pre-processing, 
since I dont want each dataloader re-compute everything for now

"""


import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from pytorch3d import transforms


class GeometryLatentDataset(Dataset):
    def __init__(
            self,
            cfg,
            data_dir,
            category,
            overfit,
            data_fn
    ):

        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        # Right now is just 20
        self.max_num_part = 20

        if overfit != -1:
            self.data_files = self.data_files[:overfit] * 40

        self.data_list = []

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))
            latent = data_dict['latent']
            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            part_trans = data_dict['part_trans']
            xyz = data_dict['xyz']
            num_parts = data_dict["num_parts"].item()
            mesh_file_path = data_dict['mesh_file_path'].item()
            part_pcs = data_dict['part_pcs']
            scale = data_dict['scale']
            part_rots = data_dict["gt_quats"]

            if cfg.model.ref_part:
                ref_part = np.argmax(scale[:num_parts])
            else:
                ref_part = -1

            cur_pts, cur_quats = [], []
            for i in range(self.max_num_part):
                if i == ref_part:
                    pc, gt_quat = self._rotate_pc(part_pcs[i], True)
                else:
                    pc, gt_quat = self._rotate_pc(part_pcs[i], False)
                cur_pts.append(pc)
                cur_quats.append(gt_quat)

            part_pcs = np.stack(cur_pts, axis=0).astype(np.float32)
            part_rots = np.stack(cur_quats, axis=0).astype(np.float32)

            sample = {
                'latent': latent,
                'data_id': data_id,
                'part_valids': part_valids,
                'part_trans': part_trans,
                'xyz': xyz,
                'mesh_file_path': mesh_file_path,
                'num_parts': num_parts,
                'part_pcs': part_pcs,
                'part_scale': scale,
                'part_rots': part_rots,
                'ref_part': ref_part
            }

            self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)


    def _rotate_pc(self, pc, ref_part):
        """pc: [N, 3]"""
        if ref_part:
            # do not rotate
            rot_mat = np.eye(3)
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt
    

    def __getitem__(self, idx):
        
        return self.data_list[idx]


def build_geometry_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_dir,
        category=cfg.data.category,
        overfit=cfg.data.overfit,
        data_fn="train",
    )
    train_set = GeometryLatentDataset(**data_dict)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )


    data_dict['data_fn'] = "val"
    data_dict['data_dir'] = cfg.data.data_val_dir
    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader


def build_test_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_val_dir,
        category=cfg.data.category,
        overfit=cfg.data.overfit,
        data_fn="test",
    )

    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return val_loader


