from pathlib import Path

src = Path(r"baselines/v17/v17_fuse.py").read_text(encoding="utf-8")

src = src.replace(
'''"""v17 nested late-fuse: MidFusePlus (velocity) + s2s/ms2s + v11 branches.

Clear-win overwrite+ping: holdout >= 0.624 OR (holdout >= 0.619 AND nested OOF >= 0.660).
Selection = max nested OOF; holdout last. No TTA / no leaky stackers.
Branches: stronger KD + optional KD seed2 + MidFusePlus + v15 branches.
"""''',
'''"""v18 nested late-fuse: KD alpha/seed/teacher diversity + KD-only blends.

Clear-win overwrite+ping: holdout >= 0.636 OR (holdout >= 0.631 AND nested OOF >= 0.670).
Selection = max nested OOF among multi-branch blends; holdout last.
No TTA / no leaky stackers. Prefer KD-only logit ensembles (ckpt size).
"""'''
)

src = src.replace("V16_OOF = 0.6521", "V17_OOF = 0.6636\nV16_OOF = 0.6521")
src = src.replace("V16_HOLD = 0.6139", "V17_HOLD = 0.6257\nV16_HOLD = 0.6139")
src = src.replace("WIN_OOF_STRICT = 0.660", "WIN_OOF_STRICT = 0.670")
src = src.replace("WIN_HOLD_STRICT = 0.619", "WIN_HOLD_STRICT = 0.631")
src = src.replace("WIN_HOLD_ALONE = 0.624", "WIN_HOLD_ALONE = 0.636")
src = src.replace("WIN_OOF_SOFT = 0.660", "WIN_OOF_SOFT = 0.670")
src = src.replace("WIN_HOLD_SOFT = 0.619", "WIN_HOLD_SOFT = 0.631")

old_cand = '''        ("kd_a02", ROOT / "oof_kd_a02.npz", ["kd_a02", "kd", "student", "compact"], "kd_a02"),
        ("kdv15", ROOT / "oof_kd_v15.npz", ["kd", "student", "compact"], "kdv15"),
    ]'''
new_cand = '''        ("kd_a02", ROOT / "oof_kd_a02.npz", ["kd_a02", "kd", "student", "compact"], "kd_a02"),
        ("kd_a01", ROOT / "oof_kd_a01.npz", ["kd_a01", "kd", "student", "compact"], "kd_a01"),
        ("kd_a015", ROOT / "oof_kd_a015.npz", ["kd_a015", "kd", "student", "compact"], "kd_a015"),
        ("kd_a025", ROOT / "oof_kd_a025.npz", ["kd_a025", "kd", "student", "compact"], "kd_a025"),
        ("kd_a02s7", ROOT / "oof_kd_a02s7.npz", ["kd_a02s7", "kd", "student", "compact"], "kd_a02s7"),
        ("kd_a02s123", ROOT / "oof_kd_a02s123.npz", ["kd_a02s123", "kd", "student", "compact"], "kd_a02s123"),
        ("kd_mf_a02", ROOT / "oof_kd_mf_a02.npz", ["kd_mf_a02", "kd", "student", "midfuse"], "kd_mf_a02"),
        ("kd_c_v15a02", ROOT / "oof_kd_c_v15a02.npz", ["kd_c_v15a02", "kd", "student", "compact"], "kd_c_v15a02"),
        ("kd_eq_a02", ROOT / "oof_kd_eq_a02.npz", ["kd_eq_a02", "kd", "student", "compact"], "kd_eq_a02"),
        ("kd_T1_a02", ROOT / "oof_kd_T1_a02.npz", ["kd_T1_a02", "kd", "student", "compact"], "kd_T1_a02"),
        ("kd_T4_a02", ROOT / "oof_kd_T4_a02.npz", ["kd_T4_a02", "kd", "student", "compact"], "kd_T4_a02"),
        ("kd_rich_a02", ROOT / "oof_kd_rich_a02.npz", ["kd_rich_a02", "kd", "student", "compact"], "kd_rich_a02"),
        ("kdv15", ROOT / "oof_kd_v15.npz", ["kd", "student", "compact"], "kdv15"),
    ]'''
assert old_cand in src, "candidates block not found"
src = src.replace(old_cand, new_cand)

old_blend = '''        # v17: exhaustive KD multi-blends on v16-validated pool + conf-weighted.
        # New diversity branches (kd3/a02/mf_v13) registered as solos for ablation;
        # they showed large OOF-hold gaps and are not auto-combinatorially expanded.
        _kd_pool = [k for k in ("kd", "kd2", "kd_c", "kd_alt") if k in P]
        from itertools import combinations
        for r in range(2, len(_kd_pool) + 1):
            for combo in combinations(_kd_pool, r):
                tag = "_".join(combo)
                bases_spec[f"eq_{tag}"] = (list(combo), "eq")
                bases_spec[f"pow_{tag}"] = (list(combo), "pow")
                bases_spec[f"conf_{tag}"] = (list(combo), "conf")
        # limited pairwise: each new branch with eq_kd_kd_c (best 2-mix peek)
        for nb in ("kd3", "kd_a02", "kd_mf_v13"):
            if nb in P and "kd" in P and "kd_c" in P:
                bases_spec[f"eq_kd_kd_c_{nb}"] = (["kd", "kd_c", nb], "eq")
                bases_spec[f"conf_kd_kd_c_{nb}"] = (["kd", "kd_c", nb], "conf")
'''

new_blend = '''        # v18: exhaustive KD-only multi-blends (eq/pow/conf) on all available KD
        # students. Cap combo size at 6 to keep nested search tractable; also
        # register full-pool eq/pow/conf. Non-KD kept for ablation only.
        from itertools import combinations
        _kd_all = [
            k for k in (
                "kd", "kd2", "kd_c", "kd_alt", "kd3", "kd_a02", "kd_mf_v13",
                "kd_a01", "kd_a015", "kd_a025", "kd_a02s7", "kd_a02s123",
                "kd_mf_a02", "kd_c_v15a02", "kd_eq_a02", "kd_T1_a02", "kd_T4_a02",
                "kd_rich_a02",
            ) if k in P
        ]
        print(f"v18 KD pool ({len(_kd_all)}): {_kd_all}", flush=True)
        _max_r = min(6, len(_kd_all))
        for r in range(2, _max_r + 1):
            for combo in combinations(_kd_all, r):
                tag = "_".join(combo)
                bases_spec[f"eq_{tag}"] = (list(combo), "eq")
                bases_spec[f"pow_{tag}"] = (list(combo), "pow")
                bases_spec[f"conf_{tag}"] = (list(combo), "conf")
        if len(_kd_all) >= 2:
            tag = "_".join(_kd_all)
            bases_spec[f"eq_{tag}"] = (list(_kd_all), "eq")
            bases_spec[f"pow_{tag}"] = (list(_kd_all), "pow")
            bases_spec[f"conf_{tag}"] = (list(_kd_all), "conf")
'''
assert old_blend in src, "blend block not found"
src = src.replace(old_blend, new_blend)

old_solo = '''    for solo in ("kd", "kd2", "kd_c", "kd_alt", "kd3", "kd_mf_v13", "kd_a02", "kdv15", "mfw", "s2s", "ms2s"):
        if solo in P:
            bases_spec[f"solo_{solo}"] = ([solo], "eq")
'''
new_solo = '''    for solo in (
        "kd", "kd2", "kd_c", "kd_alt", "kd3", "kd_mf_v13", "kd_a02",
        "kd_a01", "kd_a015", "kd_a025", "kd_a02s7", "kd_a02s123",
        "kd_mf_a02", "kd_c_v15a02", "kd_eq_a02", "kd_T1_a02", "kd_T4_a02",
        "kd_rich_a02", "kdv15", "mfw", "s2s", "ms2s",
    ):
        if solo in P:
            bases_spec[f"solo_{solo}"] = ([solo], "eq")
'''
assert old_solo in src, "solo block not found"
src = src.replace(old_solo, new_solo)

old_meta = '''    if "kd_a02" in P:
        branches_meta["kd_a02"] = ("kd", ROOT / "checkpoints_kd_a02", "compact")
    if "kdv15" in P:
'''
new_meta = '''    if "kd_a02" in P:
        branches_meta["kd_a02"] = ("kd", ROOT / "checkpoints_kd_a02", "compact")
    for _alias, _subdir, _default in [
        ("kd_a01", "checkpoints_kd_a01", "compact"),
        ("kd_a015", "checkpoints_kd_a015", "compact"),
        ("kd_a025", "checkpoints_kd_a025", "compact"),
        ("kd_a02s7", "checkpoints_kd_a02s7", "compact"),
        ("kd_a02s123", "checkpoints_kd_a02s123", "compact"),
        ("kd_mf_a02", "checkpoints_kd_mf_a02", "midfuse"),
        ("kd_c_v15a02", "checkpoints_kd_c_v15a02", "compact"),
        ("kd_eq_a02", "checkpoints_kd_eq_a02", "compact"),
        ("kd_T1_a02", "checkpoints_kd_T1_a02", "compact"),
        ("kd_T4_a02", "checkpoints_kd_T4_a02", "compact"),
        ("kd_rich_a02", "checkpoints_kd_rich_a02", "compact"),
    ]:
        if _alias in P:
            branches_meta[_alias] = ("kd", ROOT / _subdir, _default)
    if "kdv15" in P:
'''
assert old_meta in src, "meta block not found"
src = src.replace(old_meta, new_meta)

src = src.replace("submission_v17_candidate.csv", "submission_v18_candidate.csv")
src = src.replace("submission_v17.csv", "submission_v18.csv")
src = src.replace("submission_v17_probs.npz", "submission_v18_probs.npz")
src = src.replace("v17 clear-win:", "v18 clear-win:")
src = src.replace("(gate hold>=0.624, or hold>=0.619 & OOF>=0.660)", "(gate hold>=0.636, or hold>=0.631 & OOF>=0.670)")
src = src.replace("NO overwrite - track submission remains v16", "NO overwrite - track submission remains v17")
src = src.replace(
    '"v16_nested_oof": V16_OOF, "v16_holdout": V16_HOLD,',
    '"v17_nested_oof": V17_OOF, "v17_holdout": V17_HOLD, "v16_nested_oof": V16_OOF, "v16_holdout": V16_HOLD,',
)

marker = '("kd_a02", ROOT / "holdout_kd_a02.npz", ["kd_a02", "kd", "student", "compact"]),'
assert marker in src, "holdout marker not found"
insert = '''("kd_a02", ROOT / "holdout_kd_a02.npz", ["kd_a02", "kd", "student", "compact"]),
        ("kd_a01", ROOT / "holdout_kd_a01.npz", ["kd_a01", "kd", "student", "compact"]),
        ("kd_a015", ROOT / "holdout_kd_a015.npz", ["kd_a015", "kd", "student", "compact"]),
        ("kd_a025", ROOT / "holdout_kd_a025.npz", ["kd_a025", "kd", "student", "compact"]),
        ("kd_a02s7", ROOT / "holdout_kd_a02s7.npz", ["kd_a02s7", "kd", "student", "compact"]),
        ("kd_a02s123", ROOT / "holdout_kd_a02s123.npz", ["kd_a02s123", "kd", "student", "compact"]),
        ("kd_mf_a02", ROOT / "holdout_kd_mf_a02.npz", ["kd_mf_a02", "kd", "student", "midfuse"]),
        ("kd_c_v15a02", ROOT / "holdout_kd_c_v15a02.npz", ["kd_c_v15a02", "kd", "student", "compact"]),
        ("kd_eq_a02", ROOT / "holdout_kd_eq_a02.npz", ["kd_eq_a02", "kd", "student", "compact"]),
        ("kd_T1_a02", ROOT / "holdout_kd_T1_a02.npz", ["kd_T1_a02", "kd", "student", "compact"]),
        ("kd_T4_a02", ROOT / "holdout_kd_T4_a02.npz", ["kd_T4_a02", "kd", "student", "compact"]),
        ("kd_rich_a02", ROOT / "holdout_kd_rich_a02.npz", ["kd_rich_a02", "kd", "student", "compact"]),'''
src = src.replace(marker, insert)

out = Path(r"baselines/v18/v18_fuse.py")
out.write_text(src, encoding="utf-8")
print("wrote", out, "bytes", len(src))
assert "WIN_HOLD_ALONE = 0.636" in src
assert "kd_a01" in src
assert "submission_v18.csv" in src
assert "V17_OOF = 0.6636" in src
print("sanity ok")
