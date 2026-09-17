from pathlib import Path
p = Path(r"D:\CUHK-X\Small-Model-Track\baselines\v6\final_v6.py")
c = p.read_text(encoding="utf-8")
main = c.find("\ndef main(")
assert main > 0
# find infer_test's pandas block
start = c.rfind("    import pandas as pd", 0, main)
assert start > 0
# from start to just before def main, keep everything before pandas, replace rest of function end
fn_head_end = start
# find return st of infer_test
ret = c.find("    return st\n", start)
assert ret > 0
ret_end = ret + len("    return st\n")
new_tail = '''    import pandas as pd
    paths_out = []
    for cid in clip_ids:
        cid = str(cid)
        if cid.startswith("small_model_track_test/"):
            paths_out.append(cid if cid.endswith("/") else cid + "/")
        else:
            name = cid if cid.startswith("SM_test_") else Path(cid).name
            if not name.startswith("SM_test_"):
                # try parent
                name = Path(cid).name
            paths_out.append(f"small_model_track_test/{name}/")
    df = pd.DataFrame({"path": paths_out, "prediction": pred.astype(int)})
    df = df.sort_values("path").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} n={len(df)} defer_stats={st}", flush=True)
    print(df.head(3).to_string(), flush=True)
    return st

'''
c2 = c[:start] + new_tail + c[main:]
p.write_text(c2, encoding="utf-8")
print("patched", start, ret_end, "len", len(c), "->", len(c2))
