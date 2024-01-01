python test.py experiment_name=train_all_bs24_4gpu \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    inference_dir=test_on_training_2000epoch \
    ckpt_path=output/train_all_bs24_4gpu/training/last.ckpt \
    data.val_batch_size=6


python test.py experiment_name=train_all_bs24_4gpu_fixed_encoder_add_ref_part \
    model.DDPM_BETA_SCHEDULE=squaredcos_cap_v2 \
    inference_dir=test_on_training_2000epoch \
    ckpt_path=output/train_all_bs24_4gpu_fixed_encoder_add_ref_part/training/last.ckpt \
    model.ref_part=True \
    data.val_batch_size=6

