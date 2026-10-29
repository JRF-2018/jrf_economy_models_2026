# results_stage15c.py: 段 15c 貸し手と探索、株式的富のペテン。python -W ignore results_stage15c.py > results_stage15c.txt
# 段 15b の貸し手(借りて賄う投資は承認の割合だけ、事業の成功に外部性)に、(1) 研究(探索)の一部を借りて賄う道、
# (2) 市場型の貸し手が株の値段の膨らみを成功の印と読む道(株式的富のペテン)と、貸倒れの記憶による信頼の低下、を足す。
# 一国、標本 2000 記入、300 年の後半 200 年、種 8 個。研究 2%、当たりの裾 1.5。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, s_R=0.02, alpha_R=1.5)
SEEDS = range(8)
T, BURN = 300, 100


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    E = g('E')[n]
    gam = g('gamma_front')
    d = np.diff(E)
    return dict(E=E.mean(), sdE=E.std(), prod=-np.log(gam[T - 1] / gam[BURN]) / (T - 1 - BURN), L_R=g('L_R')[n].mean(),
                A=np.nanmean(g('lend_A')[n]) if 'lend_A' in rec[-1] else 1.0,
                loss=np.nanmean(g('lend_loss')[n] / g('Y')[n]) if 'lend_loss' in rec[-1] else 0.0,
                trust=np.nanmean(g('trust')[n]) if 'trust' in rec[-1] else 1.0,
                qg=np.nanmean(g('q_gap')[n]) if 'q_gap' in rec[-1] else 0.0, sdq=np.nanstd(g('q_gap')[n]) if 'q_gap' in rec[-1] else 0.0,
                drop=float((d < -0.1 * E.mean()).sum()) / (T - BURN) * 100, top10=g('top10_inc')[n].mean(),
                books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / np.maximum(g('Y'), 1.0)))


def table(title, cases, cols):
    print(title)
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print('  場合                                   | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:36s} | すべて崩れた'); continue
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:36s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


def c1_explore():
    cols = [('雇用', 'E', '.0f'), ('技術の進み', 'prod', '.4f'), ('研究の人', 'L_R', '.1f'), ('承認の割合', 'A', '.2f'), ('貸倒れ/所得', 'loss', '.4f'),
            ('雇用の揺れ', 'sdE', '.0f'), ('所得の上位10%', 'top10', '.3f')]
    cases = [('貸し手なし', dict())]
    for p0 in (0.8, 0.9):
        for lend in ('market', 'relation'):
            cases.append((f'p0 {p0}、{lend}、探索は内部の資金', dict(lending=lend, p0_s=p0)))
            cases.append((f'p0 {p0}、{lend}、探索の半分を借りる', dict(lending=lend, p0_s=p0, res_loan=0.5)))
        cases.append((f'p0 {p0}、relation、探索の半分、探索に 0.6', dict(lending='relation', p0_s=p0, res_loan=0.5, g_rel_x=0.6)))
    table('C1 関係融資は探索を減らすか: 研究の半分を借りて賄う。市場型は投資と同じ承認の割合で、関係融資は実績のない探索に 2 割だけ貸す', cases, cols)


def c2_swindle():
    cols = [('雇用', 'E', '.0f'), ('雇用の揺れ', 'sdE', '.0f'), ('急な落ち込み(100 年あたり)', 'drop', '.1f'), ('技術の進み', 'prod', '.4f'),
            ('承認の割合', 'A', '.2f'), ('貸倒れ/所得', 'loss', '.4f'), ('株の開き', 'qg', '+.2f'), ('開きの揺れ', 'sdq', '.2f'), ('信頼', 'trust', '.2f')]
    S = dict(eq_market=True, eq_pricing='demand', nu_E=0.5, p0_s=0.85)
    cases = []
    for lend in ('market', 'relation'):
        for sw, km, lab in ((0.0, 0.0, 'ペテンなし'), (0.3, 0.0, 'ペテン 0.3'), (0.3, 20.0, 'ペテン 0.3、損の記憶 20'), (0.3, 100.0, 'ペテン 0.3、損の記憶 100')):
            cases.append((f'{lend}、{lab}', dict(S, lending=lend, phi_sw=sw, k_mem=km)))
    table('C2 株式的富のペテン: 需要で値段のつく株(値上がりの外挿 0.5)。市場型の貸し手は株の値段の膨らみを成功の印と読む(関係融資は読まない)。'
          '信頼は貸倒れの記憶で下がり、家計の株の需要にも掛かる。事業の成功の確率の基準 0.85', cases, cols)


if __name__ == '__main__':
    c1_explore()
    c2_swindle()
