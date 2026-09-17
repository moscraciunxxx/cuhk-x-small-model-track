import os, json
from pathlib import Path
from collections import defaultdict
import numpy as np

TRAIN_DC = Path(r'D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Depth_Color')
TRAIN_IR = Path(r'D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\IR')
TEST_ROOT = Path(r'D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test')
META = json.loads(Path(r'D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2\cache\train_meta.json').read_text(encoding='utf-8'))

def count_pngs(d):
    if not d.is_dir():
        return 0
    n = 0
    for p in d.iterdir():
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            n += 1
    return n

def modality_trial_dir(mod_root, action, user_id, trial):
    return mod_root / action / f'user{user_id}' / trial

stats = {
  'n_meta': len(META),
  'depth': {'empty':0,'missing_dir':0,'ok':0,'n_frames':[]},
  'ir': {'empty':0,'missing_dir':0,'ok':0,'n_frames':[]},
}
by_user_depth_empty = defaultdict(int)
by_action_depth_empty = defaultdict(int)
by_user_ir_empty = defaultdict(int)

for m in META:
    action = m['action_name']; uid=m['user_id']; trial=m['trial']
    for key, root in [('depth', TRAIN_DC), ('ir', TRAIN_IR)]:
        d = modality_trial_dir(root, action, uid, trial)
        if not d.is_dir():
            stats[key]['missing_dir'] += 1
            n=0
        else:
            n = count_pngs(d)
            if n==0:
                stats[key]['empty'] += 1
            else:
                stats[key]['ok'] += 1
                stats[key]['n_frames'].append(n)
        if key=='depth' and n==0:
            by_user_depth_empty[uid]+=1
            by_action_depth_empty[action]+=1
        if key=='ir' and n==0:
            by_user_ir_empty[uid]+=1

test_ids = sorted([p.name for p in TEST_ROOT.iterdir() if p.is_dir() and p.name.startswith('SM_test_')])
test_stats = {'n_clips': len(test_ids), 'depth': {'empty':0,'missing':0,'ok':0,'n_frames':[]}, 'ir': {'empty':0,'missing':0,'ok':0,'n_frames':[]}}
for tid in test_ids:
    for key, sub in [('depth','Depth_Color'),('ir','IR')]:
        d = TEST_ROOT / tid / sub
        if not d.is_dir():
            test_stats[key]['missing'] += 1
            n=0
        else:
            n = count_pngs(d)
            if n==0:
                test_stats[key]['empty'] += 1
            else:
                test_stats[key]['ok'] += 1
                test_stats[key]['n_frames'].append(n)

def summarize(s):
    fr = s['n_frames']
    out = {k:s[k] for k in s if k!='n_frames'}
    if fr:
        a=np.array(fr)
        out['frames_mean']=float(a.mean()); out['frames_median']=float(np.median(a)); out['frames_min']=int(a.min()); out['frames_max']=int(a.max())
    tot = s['ok']+s['empty']+s.get('missing_dir',s.get('missing',0))
    out['empty_or_missing_rate'] = (s['empty']+s.get('missing_dir',s.get('missing',0)))/max(tot,1)
    return out

result = {
  'train': {'n_meta': stats['n_meta'], 'depth': summarize(stats['depth']), 'ir': summarize(stats['ir']),
            'depth_empty_by_user': dict(sorted(by_user_depth_empty.items())),
            'ir_empty_by_user': dict(sorted(by_user_ir_empty.items())),
            'depth_empty_top_actions': dict(sorted(by_action_depth_empty.items(), key=lambda x:-x[1])[:10])},
  'test': {'n_clips': test_stats['n_clips'], 'depth': summarize(test_stats['depth']), 'ir': summarize(test_stats['ir'])}
}
outp = Path(r'D:\CUHK-X\Small-Model-Track\baselines\v5\modality_coverage.json')
outp.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
