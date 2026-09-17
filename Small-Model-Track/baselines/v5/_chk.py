from pathlib import Path
import re
p = Path("probe_depth_ir.py")
t = p.read_text(encoding="utf-8")
# ensure it prefers cuda - already has auto. Just verify.
print("has auto", "device == \"auto\"" in t or "args.device == \"auto\"" in t)
# reduce epochs slightly if needed - 25 is fine
# bump batch if GPU free
