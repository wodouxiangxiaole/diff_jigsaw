
python train.py \
    experiment_name=train_9d_no_ref_part \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    data.batch_size=80 \
    data.val_batch_size=80 \
    +trainer.devices=2 \
    +trainer.strategy=ddp
    