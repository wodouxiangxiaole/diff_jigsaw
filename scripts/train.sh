
python train.py \
    experiment_name=train_all_rot_only_rot_every_batch \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    data.batch_size=50 \
    data.val_batch_size=50 \
    model.ref_part=True \
    data.rot_in_getitem=True \
    data.data_dir=../vq_jigsaw/latent_data/bottle_volume_constrained_no_rot/train/ \
    data.data_val_dir=../vq_jigsaw/latent_data/bottle_volume_constrained_no_rot/val/ \
    +trainer.devices=2 \
    +trainer.strategy=ddp
    