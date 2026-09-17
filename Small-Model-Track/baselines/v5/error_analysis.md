# MidFuse v2 error analysis (v5)

Quiet MSI run. Holdout users **{8,9,24}**; OOF from 5 GroupKFold MidFuse v2b fold ckpts.

- Holdout ckpt: `D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2\checkpoints_midfuse_v2b\best_holdout.pt`
- Fold ckpt dir: `D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2\checkpoints_midfuse_v2b`
- Reported holdout baseline: **0.537** (measured here 0.5366)
- OOF accuracy: **0.5015** (fold mean 0.5050853925069481)

## Does imbalance explain errors?

- (holdout) Positive support–recall corr (0.407); rarer classes tend to have lower recall — imbalance contributes.
- (holdout) High-vs-low support recall gap=0.274.
- (holdout) Top confusions look largely semantic/similar-motion (not pure frequency): 9_Pour_drinks→10_Stir_drinks (10); 11_Peel_fruits→10_Stir_drinks (8); 30_Do_jumping_jacks→31_Do_stretching_exercises (6); 29_Do_squats→32_Stand_up (5); 26_Play_games→7_Eat_food (5)
- (oof) Positive support–recall corr (0.439); rarer classes tend to have lower recall — imbalance contributes.
- (oof) High-vs-low support recall gap=0.189.
- (oof) Top confusions look largely semantic/similar-motion (not pure frequency): 10_Stir_drinks→9_Pour_drinks (25); 12_Sweep_the_floor→13_Mop_the_floor (24); 11_Peel_fruits→10_Stir_drinks (20); 22_Turn_pages→21_Read_documents (19); 10_Stir_drinks→11_Peel_fruits (17)

## holdout_midfuse_v2b_users_8_9_24

- n=505  accuracy=**0.5366**  errors=234
- majority baseline: class 36 (36_Walk) acc=0.1228
- corr(support, recall)=0.40695978963959445
- recall gap (high-support quartile − low-support quartile)=0.2744352465652277
- top-10 confused pairs cover 22.2% of errors
- mean conf: all=0.647 correct=0.7462998032569885 wrong=0.5318403840065002

### Top confused pairs

| true | pred | count | frac_of_true |
|------|------|------:|-------------:|
| 9_Pour_drinks → | 10_Stir_drinks | 10 | 0.56 |
| 11_Peel_fruits → | 10_Stir_drinks | 8 | 0.53 |
| 30_Do_jumping_jacks → | 31_Do_stretching_exercises | 6 | 1.00 |
| 29_Do_squats → | 32_Stand_up | 5 | 0.36 |
| 26_Play_games → | 7_Eat_food | 5 | 0.24 |
| 6_Drink_water → | 37_Take_medicine | 4 | 0.27 |
| 21_Read_documents → | 22_Turn_pages | 4 | 0.22 |
| 26_Play_games → | 19_Make_a_phone_call | 4 | 0.19 |
| 16_Fold_clothes → | 0_Wash_face | 3 | 0.50 |
| 18_Write → | 8_Take_and_use_tableware | 3 | 0.50 |
| 25_Watch_TV → | 27_Take_a_selfie | 3 | 0.50 |
| 15_Wipe_windows_and_tables → | 1_Brush_teeth | 3 | 0.33 |
| 22_Turn_pages → | 8_Take_and_use_tableware | 3 | 0.25 |
| 29_Do_squats → | 34_Sit_down | 3 | 0.21 |
| 4_Wipe_hands → | 10_Stir_drinks | 3 | 0.20 |

### Worst classes (by recall)

| class | support | recall | precision |
|-------|--------:|-------:|----------:|
| 2_Comb_hair | 8 | 0.000 | 0.000 |
| 18_Write | 6 | 0.000 | 0.000 |
| 25_Watch_TV | 6 | 0.000 | 0.000 |
| 30_Do_jumping_jacks | 6 | 0.000 | 0.000 |
| 8_Take_and_use_tableware | 0 | 0.000 | 0.000 |
| 11_Peel_fruits | 15 | 0.067 | 0.083 |
| 26_Play_games | 21 | 0.095 | 0.500 |
| 29_Do_squats | 14 | 0.143 | 0.333 |
| 24_Use_a_mobile_phone | 6 | 0.167 | 0.091 |
| 13_Mop_the_floor | 8 | 0.250 | 1.000 |
| 9_Pour_drinks | 18 | 0.333 | 0.500 |
| 22_Turn_pages | 12 | 0.333 | 0.333 |

## oof_5fold_midfuse_v2b

- n=2931  accuracy=**0.5015**  errors=1461
- majority baseline: class 36 (36_Walk) acc=0.1143
- corr(support, recall)=0.4394507780053445
- recall gap (high-support quartile − low-support quartile)=0.1889785122236876
- top-10 confused pairs cover 12.9% of errors
- mean conf: all=0.656 correct=0.7519590258598328 wrong=0.560298502445221

### Top confused pairs

| true | pred | count | frac_of_true |
|------|------|------:|-------------:|
| 10_Stir_drinks → | 9_Pour_drinks | 25 | 0.21 |
| 12_Sweep_the_floor → | 13_Mop_the_floor | 24 | 0.38 |
| 11_Peel_fruits → | 10_Stir_drinks | 20 | 0.19 |
| 22_Turn_pages → | 21_Read_documents | 19 | 0.32 |
| 10_Stir_drinks → | 11_Peel_fruits | 17 | 0.15 |
| 9_Pour_drinks → | 8_Take_and_use_tableware | 17 | 0.13 |
| 9_Pour_drinks → | 10_Stir_drinks | 17 | 0.13 |
| 34_Sit_down → | 32_Stand_up | 17 | 0.12 |
| 7_Eat_food → | 6_Drink_water | 17 | 0.11 |
| 29_Do_squats → | 32_Stand_up | 16 | 0.21 |
| 17_Tap_the_keyboard → | 21_Read_documents | 16 | 0.19 |
| 8_Take_and_use_tableware → | 9_Pour_drinks | 15 | 0.15 |
| 26_Play_games → | 24_Use_a_mobile_phone | 13 | 0.33 |
| 30_Do_jumping_jacks → | 31_Do_stretching_exercises | 13 | 0.30 |
| 37_Take_medicine → | 6_Drink_water | 13 | 0.19 |

### Worst classes (by recall)

| class | support | recall | precision |
|-------|--------:|-------:|----------:|
| 25_Watch_TV | 12 | 0.000 | 0.000 |
| 26_Play_games | 40 | 0.050 | 0.061 |
| 19_Make_a_phone_call | 43 | 0.093 | 0.111 |
| 18_Write | 38 | 0.105 | 0.098 |
| 35_Do_lunges | 26 | 0.154 | 0.250 |
| 16_Fold_clothes | 24 | 0.167 | 0.286 |
| 22_Turn_pages | 59 | 0.186 | 0.244 |
| 14_Wipe_bowls | 35 | 0.200 | 0.292 |
| 24_Use_a_mobile_phone | 48 | 0.250 | 0.185 |
| 8_Take_and_use_tableware | 97 | 0.268 | 0.255 |
| 10_Stir_drinks | 117 | 0.291 | 0.298 |
| 2_Comb_hair | 54 | 0.296 | 0.320 |

## Notes for v5 experiments

- Skip Radar/Thermal (known bad).
- Depth_Color + IR have full train/test coverage (see modality_coverage.json).
- High-ROI MidFuse tweak candidate: focal / class-focused loss on worst-recall classes from above.
