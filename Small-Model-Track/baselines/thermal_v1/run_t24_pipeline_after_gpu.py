"""Wait HOTC (helios/vipt) then train T24 multi-seed with retries; infer; probe; optional extra seeds."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    def now_pt():
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
except Exception:
    def now_pt():
        return time.strftime("%Y-%m-%d %H:%M:%S")

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CACHE = ROOT / "cache" / "ir_yolo_ft_v24_t24"
STATUS = ROOT / "metrics_ir_v25_status.json"
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)

def gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().splitlines()[0]
        return int(float(out))
    except Exception:
        return -1

def blocking_procs():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % { $_.ProcessId.ToString() + '|' + $_.CommandLine }"],
            text=True, timeout=25, stderr=subprocess.DEVNULL)
    except Exception as e:
        return [f"check_failed:{e}"]
    bad = []
    for ln in out.splitlines():
        low = ln.lower()
        if "run_t24_pipeline" in low or "_wait_gpu" in low:
            continue
        if "_watch_helios" in low or "watch_helios" in low:
            continue
        if "run_helios" in low or "helios_val" in low or "hotmoe" in low:
            bad.append(ln.strip()[:200]); continue
        if "run_vipt" in low or "vipt_val" in low or ("vipt" in low and "execute" in low):
            bad.append(ln.strip()[:200]); continue
        if "helios" in low and ("--execute" in low or "train" in low):
            bad.append(ln.strip()[:200])
    return bad

def wait_gpu(max_min=360):
    print(f"WAIT_GPU start {now_pt()}", flush=True)
    deadline = time.time() + max_min * 60
    while time.time() < deadline:
        used = gpu_used(); bad = blocking_procs()
        print(f"{now_pt()} mem={used}MiB blockers={len(bad)}", flush=True)
        if bad:
            for b in bad[:2]:
                print("  block:", b, flush=True)
        if not bad and 0 <= used < 100:
            time.sleep(8)
            used = gpu_used(); bad = blocking_procs()
            if not bad and 0 <= used < 100:
                print(f"GPU_FREE mem={used} {now_pt()}", flush=True)
                return True
        time.sleep(25)
    print("TIMEOUT waiting GPU", flush=True)
    return False

def patch_status(**kw):
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8-sig")) if STATUS.exists() else {}
    except Exception:
        s = {}
    s.update(kw); s["updated_at"] = now_pt()
    STATUS.write_text(json.dumps(s, indent=2), encoding="utf-8")

def run_logged(cmd, log_name, env_extra=None, retries=1):
    log = LOGDIR / f"{log_name}.log"
    err = LOGDIR / f"{log_name}.err"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if env_extra:
        env.update(env_extra)
    last_rc = 1
    for attempt in range(1, retries + 1):
        print(f"RUN attempt={attempt}/{retries} {cmd} -> {log}", flush=True)
        with open(log, "w", encoding="utf-8") as lo, open(err, "w", encoding="utf-8") as le:
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lo, stderr=le, env=env)
        (LOGDIR / f"{log_name}.pid").write_text(str(p.pid), encoding="utf-8")
        rc = p.wait()
        # unsigned wrap
        if rc < 0 or rc > 255:
            rc_signed = rc - (1 << 32) if rc > 0x7FFFFFFF else rc
        else:
            rc_signed = rc
        print(f"DONE {log_name} rc={rc} signed={rc_signed} at={now_pt()}", flush=True)
        try:
            for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
                print(" ", ln, flush=True)
        except Exception:
            pass
        try:
            et = err.read_text(encoding="utf-8", errors="replace")
            if et.strip():
                print("ERR_TAIL:", flush=True)
                for ln in et.splitlines()[-30:]:
                    print(" ", ln, flush=True)
        except Exception:
            pass
        last_rc = rc_signed if rc_signed != 0 else 0
        if last_rc == 0:
            return 0
        # brief cooldown before retry
        time.sleep(5)
        if not wait_gpu(max_min=10):
            break
    return last_rc if last_rc != 0 else 1

def train_seeds(seeds, retries=3):
    patch_status(outcome=f"TRAINING_T24_SEEDS_{'_'.join(map(str,seeds))}", gpu="TRAINING")
    existing = [int(p.stem.replace("pool_seed", "")) for p in sorted(CK.glob("pool_seed*.pt"))]
    all_seeds = sorted(set(existing) | set(seeds))
    cmd = [str(PY), "-u", "train_ir_focal_ft_t24_v24.py",
           "--t", "24", "--cache-dir", str(CACHE), "--ckpt-dir", str(CK),
           "--seeds", *[str(s) for s in all_seeds]]
    # First try normal; on fail retry with CUDA_LAUNCH_BLOCKING
    rc = run_logged(cmd, f"train_focal_ft_t24_s{'_'.join(map(str,seeds))}", retries=1)
    if rc != 0:
        print("RETRY with CUDA_LAUNCH_BLOCKING=1", flush=True)
        rc = run_logged(cmd, f"train_focal_ft_t24_s{'_'.join(map(str,seeds))}_retry",
                        env_extra={"CUDA_LAUNCH_BLOCKING": "1"}, retries=max(retries - 1, 1))
    return rc

def infer_test(tag, t, cache, ckpt, bs=4):
    patch_status(outcome=f"INFER_{tag}", gpu="INFER")
    cmd = [str(PY), "-u", "infer_ir_t24_test.py", "--t", str(t),
           "--cache-dir", str(cache), "--ckpt-dir", str(ckpt), "--bs", str(bs)]
    return run_logged(cmd, f"infer_{tag}")

def probe():
    patch_status(outcome="PROBE_T24_MULTISEED", gpu="CPU_PROBE")
    return run_logged([str(PY), "-u", "probe_ir_v25_t24_multiseed.py"], "probe_t24_multiseed")

def main():
    print(f"pipeline start {now_pt()} root={ROOT}", flush=True)
    if not wait_gpu():
        patch_status(outcome="TIMEOUT_WAITING_HOTC", gpu="BLOCKED")
        return 2

    rc = train_seeds([2024, 888], retries=3)
    if rc != 0:
        patch_status(outcome="TRAIN_FAIL", train_rc=rc)
        return rc

    # ff T16 test logits if missing
    ff_ck = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24"
    ff_cache = ROOT / "cache" / "ir_yolo_ft_v24"
    if ff_ck.exists() and not (ff_ck / "test_logits_ens.npy").exists() and (ff_cache / "test_x_t16_s112.npy").exists():
        infer_test("ff_t16", 16, ff_cache, ff_ck, bs=6)

    rc = infer_test("t24", 24, CACHE, CK, bs=4)
    if rc != 0:
        patch_status(outcome="INFER_FAIL", infer_rc=rc)
        return rc

    rc = probe()
    if rc == 0:
        print("WIN", flush=True)
        return 0

    try:
        s = json.loads(STATUS.read_text(encoding="utf-8-sig"))
        ens_solo = float(s.get("t24_train", {}).get("ens_solo_hold") or s.get("progress", {}).get("focal_ft_t24_ens") or 0)
        honest = float(s.get("progress", {}).get("best_honest") or 0)
    except Exception:
        ens_solo, honest = 0.0, 0.0
    print(f"after phaseA ens_solo={ens_solo} honest={honest} probe_rc={rc}", flush=True)

    if ens_solo >= 0.695 or honest >= 0.752:
        if wait_gpu(max_min=30):
            extra = [s for s in (7, 123, 2025) if not (CK / f"pool_seed{s}.pt").exists()]
            if extra:
                print("PHASE_B extra seeds", extra, flush=True)
                if train_seeds(extra, retries=2) == 0:
                    infer_test("t24", 24, CACHE, CK, bs=4)
                    rc = probe()
                    if rc == 0:
                        print("WIN after extra seeds", flush=True)
                        return 0

    print("MISS keep ir_v7; pipeline end", now_pt(), flush=True)
    patch_status(outcome="MISS_AFTER_PIPELINE", keep_ir_v7=True, wrote_csv=False)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
