python train.py \
    experiment_name=train_all_rot_only_pred_x0 \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    model.ref_part=True \
    model.PREDICT_TYPE=sample \
    data.batch_size=40 \
    data.val_batch_size=40 \
    model.PREDICT_TYPE=sample \
    +trainer.devices=2 \
    +trainer.strategy=ddp 

    