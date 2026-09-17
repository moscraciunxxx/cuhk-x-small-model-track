from pathlib import Path
p = Path(__file__).resolve().parent / "probe_ir_v21.py"
t = p.read_text(encoding="utf-8")
old = 'max_iter=2000, multi_class="multinomial", solver="lbfgs",'
new = 'max_iter=2000, solver="lbfgs",'
n = t.count(old)
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("replaced", n)
