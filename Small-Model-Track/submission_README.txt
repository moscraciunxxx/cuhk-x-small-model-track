v19 clear-win (size<=100MB): nested OOF 0.6653 holdout 0.6436 (gate hold>=0.642, or hold>=0.637 & OOF>=0.68)
method=conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02 params={"keys": ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"], "temp": 0.5}
fold_plan={"kd": [1, 2, 3], "kd_mf_v13": [1, 2, 3], "kd_mf_a02": [1, 2, 3], "kd_eq_a02": [0, 1, 2, 3, 4]}
fold_ckpt_size_mb=93.54 (full 5-fold peek was ~141MB; midfuse shares folds [1, 2, 3])
ping_disk_saver=true.
