
python train.py \
    experiment_name=train_all \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    +trainer.devices=2 \
    +trainer.strategy=ddp_find_unused_parameters_true