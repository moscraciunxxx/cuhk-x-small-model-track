import json
from pathlib import Path
m = json.loads(Path("baselines/v18/metrics.json").read_text(encoding="utf-8"))
MID = {"kd", "kd_mf_v13", "kd_mf_a02"}
SZC, SZM = 21.83, 39.84

def size(keys):
    return sum(SZM if k in MID else SZC for k in keys)

WIN_HOLD_ALONE, WIN_HOLD_SOFT, WIN_OOF_SOFT = 0.636, 0.631, 0.670
rows = []
for r in m["all_methods"]:
    if str(r["name"]).startswith("solo_"):
        continue
    keys = r["params"]["keys"]
    hold, oof = r["holdout"], r["nested_oof"]
    clear = hold >= WIN_HOLD_ALONE or (hold >= WIN_HOLD_SOFT and oof >= WIN_OOF_SOFT)
    if not clear:
        continue
    rows.append((oof, hold, size(keys), r["name"], keys, r["family"], r["params"]))

rows.sort(key=lambda x: (-x[0], -x[1]))
print("clear-win count", len(rows))
print("--- top nested any size ---")
for r in rows[:12]:
    print("oof=%.4f hold=%.4f sz=%.1f %s" % (r[0], r[1], r[2], r[3]))
under = [r for r in rows if r[2] <= 100.01]
print("--- top nested size<=100 count=%d ---" % len(under))
for r in under[:25]:
    print("oof=%.4f hold=%.4f sz=%.1f %s keys=%s" % (r[0], r[1], r[2], r[3], r[4]))
if under:
    best = under[0]
    print("BEST_UNDER_100", best[3], "oof", best[0], "hold", best[1], "sz", best[2])
