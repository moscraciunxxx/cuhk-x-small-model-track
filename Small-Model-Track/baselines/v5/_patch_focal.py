from pathlib import Path
p = Path("probe_focal.py")
t = p.read_text(encoding="utf-8")
old = '    device = torch.device("cpu")  # share GPU carefully — DAM4SAM busy\n    print("device", device, flush=True)'
# handle possible encoding of emdash
import re
t2, n = re.subn(
    r'    device = torch\.device\("cpu"\).*?\n    print\("device", device, flush=True\)',
    '''    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"cuda free={free/1e9:.2f}GB total={total/1e9:.2f}GB", flush=True)
        if free < 1.2e9:
            print("low VRAM, falling back to CPU", flush=True)
            device = torch.device("cpu")
    print("device", device, flush=True)''',
    t,
    count=1,
)
if n:
    p.write_text(t2, encoding="utf-8")
    print("patched", n)
else:
    print("NO MATCH")
    for i,l in enumerate(t.splitlines()):
        if "device" in l:
            print(i, repr(l))
