
python train.py \
    experiment_name=train_all_9d_pose \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    data.batch_size=200 \
    data.val_batch_size=200 \
    model.ref_part=True \
    +trainer.devices=2 \
    +trainer.strategy=ddp
    