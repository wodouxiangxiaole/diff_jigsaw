
python train.py \
    experiment_name=exp2_train_all \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    data.batch_size=16 \
    data.val_batch_size=16 \
    +trainer.devices=3 \
    +trainer.strategy=ddp_find_unused_parameters_true
    