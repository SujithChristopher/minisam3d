import sys, torch, numpy as np
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from minisam3d.data import PseudoLabelDataset
from minisam3d.losses import evaluate
from minisam3d.model import Student
dev = 'cuda'
for name in ['v3rot_loso_kabil', 'r18b_loso_kabil']:
    ck = torch.load(f'runs/{name}/best.pt', map_location=dev, weights_only=False)
    m = Student(ck['K'], width=ck['width'], in_ch=ck['in_ch'], dec_ch=ck['dec_ch'], head3d=ck['head3d'],
                num_rot=ck['num_rot'], backbone=ck.get('backbone', 'mbv2')).to(dev).eval()
    m.load_state_dict(ck['model'])
    for take in ['kabil_game', 'kabil_woc']:
        ds = PseudoLabelDataset(f'data/noark_pack/{take}', 256, 256, train=False, gray=True)
        js = ds.js
        r = evaluate(m, DataLoader(ds, 64, num_workers=8), js, torch.tensor([256, 256], device=dev), dev)
        jm = dict(zip(js['names'], r['mm_joint'])); jr = dict(zip(js['rot_names'], r['deg_joint']))
        print(f"{name:18}{take:11} n{len(ds)} px {r['px_err']:.2f} mm {r['mpjpe_mm']:.1f} rot {r['rot_deg']:.1f} | "
              f"wrist {np.mean([jm['left_wrist'], jm['right_wrist']]):.0f}mm elbow {np.mean([jm['left_elbow'], jm['right_elbow']]):.0f}mm "
              f"shoulder {np.mean([jm['left_shoulder'], jm['right_shoulder']]):.0f}mm | trunk {jr['trunk']:.1f} "
              f"forearm {np.mean([jr['left_forearm'], jr['right_forearm']]):.1f}deg", flush=True)
