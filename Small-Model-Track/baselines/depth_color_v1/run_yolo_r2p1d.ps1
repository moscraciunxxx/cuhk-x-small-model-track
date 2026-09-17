$ErrorActionPreference = "Stop"
Set-Location "D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1"
$py = ".\.venv\Scripts\python.exe"
& $py -u train_v2.py --modality Depth_Color --arch r2p1d --t 16 --size 112 --holdout-only --epochs 50 --batch-size 10 --lr 5e-4 --patience 12 --seed 42 --cache-dir "cache/depth_color_yolo" *> "logs\train_yolo_r2p1d.log"
& $py -u -c @"
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, r'D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1')
from model_r2p1d import build_model_r2p1d, model_size_mb

ROOT = Path(r'D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1')
TRACK = Path(r'D:\CUHK-X\Small-Model-Track')
cache = ROOT / 'cache' / 'depth_color_yolo'
ckpt = ROOT / 'checkpoints' / 'depth_color_r2p1d' / 'holdout_train.pt'
# train_v2 stores under modality_arch
ckpt2 = ROOT / 'checkpoints' / 'depth_color_r2p1d' / 'holdout_train.pt'
for c in [ROOT/'checkpoints'/'Depth_Color_r2p1d'/'holdout_train.pt', ROOT/'checkpoints'/'depth_color_r2p1d'/'holdout_train.pt']:
    if c.exists():
        ckpt = c
        break
# find
cands = list((ROOT/'checkpoints').rglob('holdout_train.pt'))
print('ckpts', cands)
ckpt = [c for c in cands if 'r2p1d' in str(c) and 'yolo' not in str(c)] 
# prefer latest depth_color_r2p1d after this run
ckpt = sorted(cands, key=lambda p: p.stat().st_mtime)[-1]
print('using', ckpt)
blob = torch.load(ckpt, map_location='cpu', weights_only=False)
print('val', blob.get('val_acc'), blob.get('val_f1'))
meta = json.loads((cache/'test_meta.json').read_text(encoding='utf-8'))
empty = set(json.loads((cache/'test_empty.json').read_text(encoding='utf-8')))
X = np.memmap(cache/'test_x_t16_s112.npy', dtype=np.uint8, mode='r', shape=(len(meta),16,112,112,3))
device = torch.device('cuda')
model = build_model_r2p1d(40, in_ch=3, base=64).to(device)
model.load_state_dict(blob['model'])
model.eval()
print('size_mb', model_size_mb(model))
logits = np.zeros((len(meta),40), np.float32)
bs=32
with torch.no_grad():
  for i in range(0,len(meta),bs):
    arr = X[i:i+bs].astype(np.float32)/255.0
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0,1,4,2,3))).to(device)
    logits[i:i+len(x)] = model(x).cpu().numpy()
preds = logits.argmax(1)
# fallback midfuse for empty
fb={}
for p in [TRACK/'baselines'/'skeleton_imu_v2'/'submission_skeleton_imu_v2_ensemble.csv']:
  with open(p) as f:
    for row in csv.DictReader(f):
      fb[row['path'].rstrip('/')+'/']=int(row['prediction'])
rows=[]
nfb=0
for i,m in enumerate(meta):
  path=m['path'] if m['path'].endswith('/') else m['path']+'/'
  if m.get('empty') or m['sample_id'] in empty:
    pred=fb.get(path, int(preds[i])); nfb+=1
  else:
    pred=int(preds[i])
  rows.append((path,pred))
out = ROOT/'submission_depth_yolo_r2p1d.csv'
with open(out,'w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['path','prediction']); w.writerows(rows)
np.save(ckpt.parent/'test_logits_yolo_r2p1d.npy', logits)
print('wrote', out, 'fallback', nfb, 'val_acc', blob.get('val_acc'))
# honest blend with midfuse
mid=np.load(ROOT/'cache'/'midfuse_test_logits.npy')
honest=json.loads((ROOT/'logs'/'honest_blend_report.json').read_text()) if (ROOT/'logs'/'honest_blend_report.json').exists() else {}
print('prev honest', honest)
"@ *> "logs\infer_yolo_r2p1d.log"
