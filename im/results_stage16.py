# results_stage16.py: 段 16 の実験(死蔵と準備率: v6 の ρ_eff の仕組みとの整合の確かめ)。python -W ignore results_stage16.py > results_stage16.txt
# 一国、標本 2000 記入、300 年の後半 200 年、乱数の種 8 個。富の集中の度合いを、段 9 の運(初めは平等)で変える。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, eq_init='equal')
SEEDS = range(8)
SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(300)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(100, 300)
    H = e.H
    dep = np.maximum(H.M, 0.0)
    o = np.argsort(-dep / H.heads); cw = np.cumsum(H.heads[o]) / H.heads.sum()
    s1 = float(dep[o][cw <= 0.01].sum() / dep.sum())
    return dict(phi=g('top10_inc')[n].mean(), s1=s1, hoardD=np.nanmean(g('hoard')[n] / g('deposits')[n]) if 'hoard' in rec[0] else 0.0,
                cf=np.nanmean(g('cf')[n]) if 'cf' in rec[0] else 1.0, loans=np.mean(g('loans')[n] / g('Y')[n]), E=g('E')[n].mean(),
                prem=g('prem')[n].mean(), rL=g('r_loan')[n].mean(), books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / g('Y')))


def sweep(lab, kw):
    jobs = [(dict(kw, luck_sigma=sg), s) for sg in SIGMAS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print(f'  {lab}')
    print('    運 σ | 所得の上位10%(Φ) | 預金の上位1% | 死蔵/預金 | 準備の縛り(貸せる割合) | 上乗せ金利 | 貸出の金利 | 企業の貸出/所得 | 雇用')
    rows = []
    for i, sg in enumerate(SIGMAS):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        mm = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        rows.append(mm)
        print(f"    {sg:.2f} | {mm['phi']:.3f} | {mm['s1']:.3f} | {mm['hoardD']:.3f} | {mm['cf']:.2f} | {mm['prem']:.4f} | {mm['rL']:.4f} | {mm['loans']:.3f} | {mm['E']:.0f}"
              + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"    (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")
    return rows


def kink_fit(x, y):
    """y = a max(0, x − θ) の最小二乗(θ を格子で探す)。"""
    best = None
    for th in np.linspace(min(x), max(x), 200):
        z = np.maximum(0, np.array(x) - th)
        if z.sum() == 0:
            continue
        a = float(z @ np.array(y) / (z @ z))
        err = float(((np.array(y) - a * z) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, a, th)
    ss = float(((np.array(y) - np.mean(y)) ** 2).sum())
    return best[1], best[2], 1 - best[0] / ss if ss > 0 else np.nan


if __name__ == '__main__':
    print('H16 富の集中(運 σ)と、死蔵・準備の縛り・貸出。死蔵 = 一人あたりの預金の平均の 5 倍を超えた分(銀行の外の現金、利子なし、準備を吸い取る)')
    print('   (a) 書き込んだ ρ_eff の形(上乗せ金利 μ softplus(Φ − θ)、μ = 1、θ = 0.15。死蔵なし)、(b) 死蔵 + 準備は求められるだけ(今の im)、')
    print('   (c) 死蔵 + 準備の量は所得の 0.35 倍に固定(準備率 0.1)')
    ra = sweep('(a) 書き込んだ ρ_eff の形', dict(mu_prem=1.0, theta_prem=0.15))
    rb = sweep('(b) 死蔵 + 準備は内生', dict(hoard=True))
    rc = sweep('(c) 死蔵 + 準備は固定', dict(hoard=True, reserves='fixed', res_supply=0.35))
    print('  折れ線の当てはめ(y = a max(0, x − θ)):')
    a, th, r2 = kink_fit([r['s1'] for r in rb], [r['hoardD'] for r in rb])
    print(f'    (b) 死蔵/預金 と 預金の上位1% の取り分: a = {a:.2f}、θ = {th:.3f}、R² = {r2:.3f}(ノート 3 の S10 では a = 0.998、θ = 0.128、R² = 0.9999)')
    a, th, r2 = kink_fit([r['phi'] for r in rb], [r['hoardD'] for r in rb])
    print(f'    (b) 死蔵/預金 と Φ(所得の上位10%): a = {a:.2f}、θ = {th:.3f}、R² = {r2:.3f}')
    a, th, r2 = kink_fit([r['phi'] for r in rc], [1 - r['cf'] for r in rc])
    print(f'    (c) 準備の縛り(1 − 貸せる割合)と Φ: a = {a:.2f}、θ = {th:.3f}、R² = {r2:.3f}')
    a, th, r2 = kink_fit([r['phi'] for r in ra], [r['prem'] for r in ra])
    print(f'    (a) 上乗せ金利と Φ: a = {a:.2f}、θ = {th:.3f}、R² = {r2:.3f}(書き込んだ形)')
