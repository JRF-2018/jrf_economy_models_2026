# results_stage15.py: 段 15 の実験(関係融資と市場型融資)。python -W ignore results_stage15.py > results_stage15.txt
# 段 5 の世界樹の案の企業(部門に 20 社、すべて規則的に研究する。参入は固定)に、企業の質(悪い企業は 3 割で参入し、知識が目減りし、倒産しやすい)を入れる。
# 標本 2000 記入、乱数の種 12 個。不況: 200 年目から 10 年、労働者の消費の割合 0.9 → 0.75。同じ種の不況のない歴史と比べる。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage5b import tail

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, s_R=0.02, **tail(1.5),
            firms=20, rule_share0=1.0, entry='fixed')
SEEDS = range(12)
KEYS = ('E', 'Y', 'bN', 'gamma_front', 'bad_share', 'credit_good', 'loss', 'hhi', 'defaults', 'real_wage')


def history(args):
    kw, seed = args
    out = []
    for sh in (True, False):
        e = im1.Im1(seed=seed, **dict(BASE, **kw))
        rec = []
        try:
            for t in range(260):
                if sh and t == 200:
                    e.H.cw = (0.75, 0.2)
                if sh and t == 210:
                    e.H.cw = (0.9, 0.2)
                rec.append(e.step())
        except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
            return None
        out.append({k: np.array([r.get(k, np.nan) for r in rec], float) for k in KEYS} | {'books': max(r['books'] / r['Y'] for r in rec)})
    return out


def main():
    cases = (('悪い企業なし(段 5 と同じ)', dict()),
             ('見分けない(blind)', dict(lender='blind')),
             ('関係融資(ゾンビ 1.0、追い貸しを全部)', dict(lender='relation', zombie=1.0)),
             ('関係融資(ゾンビ 0.8)', dict(lender='relation')),
             ('関係融資(ゾンビ 0.3)', dict(lender='relation', zombie=0.3)),
             ('市場型(収縮 1)', dict(lender='market')),
             ('市場型(収縮 3)', dict(lender='market', crunch=3.0)),
             ('市場型(収縮 0)', dict(lender='market', crunch=0.0)),
             ('半々(mix)', dict(lender='mix')))
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(history, jobs))
    print('R15 関係融資と市場型融資。平時 = 100〜200 年の平均(不況のない歴史)、不況 = 200〜214 年の、不況のない歴史に対する比')
    print('  貸し手                       | 平時の技術の進み | 平時の雇用 | 平時の悪い企業の占有率 | 平時の良い企業の信用 | 平時の貸倒れ/所得 | 平時の集中 | '
          '不況の雇用の底 | 不況 15 年の雇用の損 | 不況の良い企業の信用 | 不況の貸倒れ/所得(差) | 不況のあと 30 年の技術の進み(差)')
    for i, (lab, kw) in enumerate(cases):
        ps = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not ps:
            print(f'  {lab:28s} | すべて崩れた'); continue
        P = slice(100, 200); W = slice(200, 215); A = slice(210, 260)
        f = lambda fn: (np.mean([fn(a, z) for a, z in ps]), np.std([fn(a, z) for a, z in ps]) / np.sqrt(len(ps)))
        cols = [f(lambda a, z: np.log(z['gamma_front'][100] / z['gamma_front'][199]) / 99), f(lambda a, z: z['E'][P].mean()),
                f(lambda a, z: np.nanmean(z['bad_share'][P])), f(lambda a, z: np.nanmean(z['credit_good'][P])),
                f(lambda a, z: np.nanmean(z['loss'][P] / z['Y'][P])), f(lambda a, z: np.nanmean(z['hhi'][P])),
                f(lambda a, z: (a['E'][W] / z['E'][W]).min() - 1), f(lambda a, z: (a['E'][W] / z['E'][W] - 1).sum()),
                f(lambda a, z: np.nanmean(a['credit_good'][W])), f(lambda a, z: np.nanmean(a['loss'][W] / a['Y'][W]) - np.nanmean(z['loss'][W] / z['Y'][W])),
                f(lambda a, z: np.log(a['gamma_front'][210] / a['gamma_front'][259]) / 49 - np.log(z['gamma_front'][210] / z['gamma_front'][259]) / 49)]
        fmt = ['.4f', '.0f', '.2f', '.2f', '.4f', '.2f', '+.3f', '+.2f', '.2f', '+.4f', '+.4f']
        print(f'  {lab:28s} | ' + ' | '.join(f'{m:{ff}}±{s_:{ff.lstrip("+")}}' for (m, s_), ff in zip(cols, fmt))
              + (f'  ({len(ps)}/{len(SEEDS)})' if len(ps) < len(SEEDS) else ''))
    print(f"  (帳簿/所得の最大 {max(max(a['books'], z['books']) for i in range(len(cases)) for a, z in [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]):.0e})")


if __name__ == '__main__':
    main()
