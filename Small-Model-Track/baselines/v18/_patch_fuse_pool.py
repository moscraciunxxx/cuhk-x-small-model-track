from pathlib import Path
p = Path(r"baselines/v18/v18_fuse.py")
src = p.read_text(encoding="utf-8")
old = '''        print(f"v18 KD pool ({len(_kd_all)}): {_kd_all}", flush=True)
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
new = '''        print(f"v18 KD pool ({len(_kd_all)}): {_kd_all}", flush=True)
        # Cap pool for exhaustive search: keep up to 10 by solo OOF on non-holdout.
        # Always retain v17 anchors (kd, kd_c, kd_a02, kd_alt, kd_mf_v13) if present.
        _solo_oof = {k: float((P[k].argmax(1) == yt_nh).mean()) for k in _kd_all}
        _anchors = [k for k in ("kd", "kd_c", "kd_a02", "kd_alt", "kd_mf_v13", "kd3") if k in _kd_all]
        _ranked = sorted(_kd_all, key=lambda k: -_solo_oof[k])
        _pool = []
        for k in _anchors + _ranked:
            if k not in _pool:
                _pool.append(k)
            if len(_pool) >= 10:
                break
        print(f"v18 exhaustive pool ({len(_pool)}): {[(k, round(_solo_oof[k],4)) for k in _pool]}", flush=True)
        _max_r = min(5, len(_pool))
        for r in range(2, _max_r + 1):
            for combo in combinations(_pool, r):
                tag = "_".join(combo)
                bases_spec[f"eq_{tag}"] = (list(combo), "eq")
                bases_spec[f"pow_{tag}"] = (list(combo), "pow")
                bases_spec[f"conf_{tag}"] = (list(combo), "conf")
        if len(_pool) >= 2:
            tag = "_".join(_pool)
            bases_spec[f"eq_{tag}"] = (list(_pool), "eq")
            bases_spec[f"pow_{tag}"] = (list(_pool), "pow")
            bases_spec[f"conf_{tag}"] = (list(_pool), "conf")
        # Also pairwise/triples of leftover new branches with eq_kd_kd_c anchors
        _extras = [k for k in _kd_all if k not in _pool]
        for nb in _extras:
            if "kd" in P and "kd_c" in P:
                bases_spec[f"eq_kd_kd_c_{nb}"] = (["kd", "kd_c", nb], "eq")
                bases_spec[f"conf_kd_kd_c_{nb}"] = (["kd", "kd_c", nb], "conf")
                bases_spec[f"pow_kd_kd_c_{nb}"] = (["kd", "kd_c", nb], "pow")
'''
assert old in src, "block missing"
p.write_text(src.replace(old, new), encoding="utf-8")
print("patched pool selection")
