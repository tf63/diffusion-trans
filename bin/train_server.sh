# 1 batch -> 1520 MiB
# batch_size -> 60
export CUDA_VISIBLE_DEVICES=0,1
../venv/bin/python3 ddpm_conditional.py --run_name epoch_1000_view_10-6000_nocfg --batch_size 58 --root_dir data/diffusion_fix/03001627 --n_gpus 2 --cfg_scale 0 --epoch_ini 0
# ../venv/bin/python3 ddpm_conditional.py --run_name epoch_1000_view_10-6000_nocfg --batch_size 30 --root_dir data/diffusion_fix/03001627 --n_gpus 1 --cfg_scale 0