cd /d D:\CUHK-X\Small-Model-Track\baselines\v3_ensemble
.venv\Scripts\python.exe train.py --mode cv --cv-splits 5 --epochs 80 --model midfuse --batch-size 28 --seed 42 --patience 20 --no-balanced-sampler --label-smoothing 0.05 --lr 8e-4 --ckpt-dir checkpoints --metrics-out metrics_v3_long.json > logs\train_v3_long.log 2>&1
