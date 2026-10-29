# results_stage12.py: 段 12 の実験(人間の有限の注意: 一年に決め直す家計の割合)。python -W ignore results_stage12.py > results_stage12.txt
# 標本 2000 記入、乱数の種 12 個。打撃は同じ種の打撃のない歴史と比べる。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
import im3
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005)
SEEDS = range(12)
ATT = (1.0, 0.5, 0.25)


def slump(args):
    """10 年の需要の不足(労働者の消費の割合 0.9 → 0.75、200 年目から)。打撃のある歴史とない歴史。"""
    kw, seed = args
    out = []
    for sh in (True, False):
        e = im1.Im1(seed=seed, **dict(BASE, **kw))
        rec = []
        try:
            for t in range(240):
                if sh and t == 200:
                    e.H.cw = (0.75, 0.2)
                if sh and t == 210:
                    e.H.cw = (0.9, 0.2)
                rec.append(e.step())
        except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
            return None
        out.append(np.array([r['E'] for r in rec]))
    a, z = out
    d = a / z - 1
    return dict(bottom=d[200:215].min(), cum=d[200:240].sum(), after=d[210:240].mean(), half=float(np.argmax(d[210:] > d[200:215].min() / 2)) if (d[210:] > d[200:215].min() / 2).any() else 30.0)


def euphoria(args):
    """土地のユーフォリア(段 2 の S2): 300 年目から 10 年、土地に置きたい割合 +0.15、そのあと利上げ。土地の実質の値段の頂と底。"""
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, lam_T=0.6, **kw))
    rec = []
    try:
        for t in range(360):
            if t == 300:
                e.p.set(theta_shock=0.15)
            if t == 310:
                e.p.set(theta_shock=0.0, r_cb=0.06)
            if t == 315:
                e.p.set(r_cb=0.02)
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    rp = np.array([r['pT'] / r['PN'] for r in rec])
    E = np.array([r['E'] for r in rec])
    return dict(peak=rp[300:340].max() / rp[280:300].mean(), bottom=rp[300:340].min() / rp[280:300].mean(), Eb=E[300:340].min() / E[280:300].mean(),
                wo=sum(r['wo_h'] for r in rec[300:340]) / np.mean([r['Y'] for r in rec[280:300]]))


def stocks(args):
    """需要で値段を決める株式の市場(段 5 の X3-4、富の弾力性 0.5)。400 年の後半 300 年。"""
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, eq_market=True, eq_pricing='demand', **kw))
    try:
        rec = [e.step() for _ in range(400)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError, OverflowError):
        return dict(crash=1.0, ratio=np.nan, infl=np.nan, sdR=np.nan)
    g = lambda k: np.array([r[k] for r in rec[100:]], float)
    return dict(crash=0.0, ratio=g('pE').sum() / g('eq_fund').sum(), infl=g('infl').mean(), sdR=np.nanstd(np.log1p(g('R_eq'))))


def hot(args):
    """速く動く資本(段 6 の K0: θ_F0 = 0.1、χ_F = 0.2)。変動の為替の年の変化の標準偏差。"""
    kw, seed = args
    w = im3.World(common=dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, theta_F0=0.1, phi_K=1.0, chi_F=0.2, **kw), currency='float', capital=True,
                  Ns=2000, seeds={'A': seed, 'B': seed + 100})
    try:
        R = [w.step() for _ in range(300)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    e = np.array([r[0]['e'] for r in R[100:]])
    return dict(sde=np.std(np.diff(np.log(e))), E=np.mean([r[0]['E'] for r in R[100:]]))


def run(f, title, cols, seeds=SEEDS):
    print(title)
    jobs = [(dict(attention=a), s) for a in ATT for s in seeds]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(f, jobs))
    print('  決め直す割合 | ' + ' | '.join(c[0] for c in cols))
    for i, a in enumerate(ATT):
        rs = [r for r in res[i * len(seeds):(i + 1) * len(seeds)] if r is not None]
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {a:.2f}         | ' + ' | '.join(f"{m[k]:{f_}}±{se[k]:{f_.lstrip('+')}}" for _, k, f_ in cols) + (f'  ({len(rs)}/{len(seeds)})' if len(rs) < len(seeds) else ''))


if __name__ == '__main__':
    run(slump, 'I1 需要の不足(労働者の消費の割合 0.9 → 0.75 を 10 年)。雇用の比(同じ種で打撃のない歴史に対して)',
        [('雇用の底', 'bottom', '+.3f'), ('40 年の雇用の損の和', 'cum', '+.2f'), ('打撃のあと 30 年の平均', 'after', '+.3f'), ('打撃のあと底の半分に戻るまでの年', 'half', '.1f')])
    run(euphoria, 'I2 土地のユーフォリアと利上げ(段 2 の S2、担保の掛け目 0.6)',
        [('土地の実質の値段の頂', 'peak', '.2f'), ('底', 'bottom', '.2f'), ('雇用の底', 'Eb', '.3f'), ('貸倒れ/所得(40 年の和)', 'wo', '.4f')])
    run(stocks, 'I3 需要で値段を決める株式の市場(段 5 の X3-4。暴れた設定)',
        [('崩れた割合', 'crash', '.2f'), ('値段/基本の値打ち', 'ratio', '.2f'), ('物価', 'infl', '+.3f'), ('株の実質の利回りの揺れ', 'sdR', '.3f')])
    run(hot, 'I4 速く動く資本と変動の為替(段 6 の K0、二地域)。乱数の種 6 組',
        [('為替の年の変化の標準偏差', 'sde', '.3f'), ('A の雇用', 'E', '.0f')], seeds=range(6))
