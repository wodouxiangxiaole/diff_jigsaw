python train.py \
    experiment_name=train_6d_rot_every_batch_mlp \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    data.batch_size=40 \
    data.val_batch_size=40 \
    model.ref_part=True \
    data.rot_in_getitem=True \
    data.data_dir=../vq_jigsaw/latent_data/bottle_volume_constrained_no_rot/train/ \
    data.data_val_dir=../vq_jigsaw/latent_data/bottle_volume_constrained_no_rot/val/ \
    +trainer.devices=2 \
    +trainer.strategy=ddp
    