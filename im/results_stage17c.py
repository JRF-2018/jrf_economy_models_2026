# results_stage17c.py: 段 17c 平行世界の拡張: もっと稀な出来事、速い量を足した物差し、適応型の閾値。python -W ignore results_stage17c.py > results_stage17c.txt
# 段 17b と同じ経済(段 17 の S17-1。標本 400 記入)。100〜300 年のうちに国債/所得が水準を超える確率を、同じ計算の量(回した年数の和)で比べる。
# 効率 = 1 / (相対の分散 × 計算の量)。素朴な推定の効率との比が 1 を超えれば、平行世界が得。
import os
import time
import numpy as np
import im1
import im_time
from concurrent.futures import ProcessPoolExecutor
from results_stage17 import DIS

T, N, REPS = 300, 80, 5
RATE = {}


def make(i):
    return im1.Im1(seed=10000 + i, **dict(DIS, dis_rate=RATE['p']))


def naive_one(args):
    level, seed, p = args
    e = im1.Im1(seed=seed, **dict(DIS, dis_rate=p))
    hit, steps = im_time.run_until(e, T, level, im_time.MEASURES['b'], 100)
    return hit, steps


def summary(name, est, cost, ref):
    est, cost = np.array(est), float(np.mean(cost))
    m, sd = est.mean(), est.std()
    rv = (sd / ref) ** 2 if ref > 0 else np.nan
    return f'{name} {m:.4f} ± {sd:.4f}(一回 {cost:.0f} 年)', rv * cost


if __name__ == '__main__':
    print('S17c 平行世界の拡張: 国債/所得が水準を超える確率(100〜300 年)。5 回くり返しの平均 ± 標準偏差。効率の比 = 素朴な推定の (相対の分散 × 計算の量) ÷ その方法の値')
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as pool:
        for prate, target in ((0.02, 1.6), (0.005, 1.2), (0.002, 0.8)):
            RATE['p'] = prate
            t0 = time.time()
            levels = tuple(np.round(np.arange(0.5, target + 1e-9, 0.1), 2))
            out = {}
            for name, fn in (('固定の閾値(国債/所得)', lambda r: im_time.splitting(make, levels, T, N, seed=r, pool=pool)[0::2]),
                             ('固定の閾値(速い量を足した物差し)', lambda r: im_time.splitting(make, levels, T, N, seed=r, pool=pool, measure='b_fwd', measure_final='b')[0::2]),
                             ('適応型(上位 3 割)', lambda r: im_time.adaptive(make, target, T, N, p0=0.3, seed=r, pool=pool)[0::2])):
                rs = [fn(r) for r in range(REPS)]
                out[name] = ([x[0] for x in rs], [x[1] for x in rs])
            budget = int(np.mean([np.mean(v[1]) for v in out.values()]))
            n_naive = max(budget // 200, 10)
            nv, nc = [], []
            for rep in range(REPS):
                res = list(pool.map(naive_one, [(target, 100000 * (rep + 1) + s, prate) for s in range(n_naive)]))
                nv.append(np.mean([r[0] for r in res])); nc.append(sum(r[1] for r in res))
            # 素朴な推定を、平行世界の 5 回ぶんの計算をまとめて使った値を「真の値の目安」にする(全部の推定の平均)
            ref = float(np.mean([np.mean(v[0]) for v in out.values()] + [np.mean(nv)]))
            line_n, eff_n = summary('素朴', nv, nc, ref)
            print(f'  災害 {prate:.1%}/年、国債/所得 > {target}(目安 {ref:.4f}): {line_n}')
            for name, (est, cost) in out.items():
                line, eff = summary(name, est, cost, ref)
                print(f'      {line} | 効率の比 {eff_n / eff if eff > 0 else np.inf:.2f}')
            print(f'      (時間 {time.time() - t0:.0f} 秒)', flush=True)
