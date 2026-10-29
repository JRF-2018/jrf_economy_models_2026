# results_stage15b.py: 段 15b(段 15 の作り直し。ノート M 第 3 章の縁故主義の形)。python -W ignore results_stage15b.py > results_stage15b.txt
# 投資のうち借りて賄う 7 割を、貸し手が承認した割合だけ実行する。事業の成功の確率は、去年のみなの投資で上がる(外部性)。
# 市場型: 成功の確率の見積りで広く承認(雑)、関係融資: 良い事業の 6 割だけを確かに(成功 95%)。失敗した事業の能力は失われ、借りた分の半分が貸倒れる。
# 一国、標本 2000 記入、乱数の種 8 個。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005)
SEEDS = range(8)


def one(args):
    kw, seed, shock = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    rec = []
    try:
        for t in range(340):
            if shock and t == 200:
                e.p.set(h_caut=shock[0])
            if shock and t == 210:
                e.p.set(h_caut=shock[1])
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(100, 200)
    out = dict(E=g('E')[n].mean(), rY=np.mean((g('Y') / g('bN'))[n]), inv=np.mean(g('inv')[n] / g('Y')[n]),
               A=np.nanmean(g('lend_A')[n]) if 'lend_A' in rec[0] else 1.0, ps=np.nanmean(g('lend_ps')[n]) if 'lend_ps' in rec[0] else np.nan,
               loss=np.nanmean(g('lend_loss')[n] / g('Y')[n]) if 'lend_loss' in rec[0] else 0.0, books=np.nanmax(g('books') / g('Y')))
    if shock:
        out.update(E_after=g('E')[300:340].mean() / g('E')[180:200].mean(), A_after=np.nanmean(g('lend_A')[300:340]),
                   E_min=g('E')[200:230].min() / g('E')[180:200].mean())
    return out


def table(title, cases, shock=None, cols=None):
    print(title)
    jobs = [(kw, s, shock) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    cols = cols or [('雇用', 'E', '.0f'), ('実質の生産', 'rY', '.0f'), ('投資/所得', 'inv', '.3f'), ('承認の割合', 'A', '.2f'), ('事業の成功の確率', 'ps', '.2f'), ('貸倒れ/所得', 'loss', '.4f')]
    print('  場合                              | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:32s} | 崩れた'); continue
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:32s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


if __name__ == '__main__':
    cases = [('貸し手を入れない(既定)', dict())]
    for p0 in (0.7, 0.8, 0.9, 0.95):
        cases += [(f'事業の環境 p0={p0}、市場型', dict(lending='market', p0_s=p0)), (f'事業の環境 p0={p0}、関係融資', dict(lending='relation', p0_s=p0))]
    table('F15-1 投資の環境(事業の成功の確率の基準 p0)と貸し手の型(外部性 κ = 0.3)。ノート M の M9: 環境が良いと市場型、悪いと関係融資が勝つか', cases)
    cases = []
    for k in (0.0, 0.3, 0.6):
        cases += [(f'外部性 κ={k}、市場型', dict(lending='market', p0_s=0.9, kappa_x=k)), (f'外部性 κ={k}、関係融資', dict(lending='relation', p0_s=0.9, kappa_x=k))]
    table('F15-2 投資の外部性の強さ(p0 = 0.9)。ノート M の M10: 外部性があると市場型の得が大きくなるか', cases)
    cols = [('雇用の底(打撃の前に対する比)', 'E_min', '.3f'), ('100〜140 年後の雇用(打撃の前に対する比)', 'E_after', '.3f'), ('100〜140 年後の承認の割合', 'A_after', '.2f')]
    for back in (0.0, -0.05):
        cases = ((f'市場型 κ=0.6、慎重さ 0 → 0.15 → {back}', dict(lending='market', p0_s=0.9, kappa_x=0.6)),
                 (f'市場型 κ=0.35、慎重さ 0 → 0.15 → {back}', dict(lending='market', p0_s=0.9, kappa_x=0.35)),
                 (f'市場型 κ=0、慎重さ 0 → 0.15 → {back}', dict(lending='market', p0_s=0.9, kappa_x=0.0)),
                 ('関係融資 κ=0.6(慎重さは効かない)', dict(lending='relation', p0_s=0.9, kappa_x=0.6)))
        table(f'F15-3 履歴(ノート M の M11): 200 年目から 10 年、市場型の貸し手の慎重さを 0.15 に上げ、そのあと {back} に戻す', cases, shock=(0.15, back), cols=cols)
