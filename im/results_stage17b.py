# results_stage17b.py: 段 17b 平行世界(多段の分割)で、稀な財政の苦しさの確率を推定する。python -W ignore results_stage17b.py > results_stage17b.txt
# 段 17 の S17-1 と同じ経済(災害は年 2%、上乗せ金利つき、利払いの重さのルール、テイラー・ルール 1.0)。標本 400 記入(速さのため)。
import os
import numpy as np
import time
import im1
import im_time
from concurrent.futures import ProcessPoolExecutor
from results_stage17 import DIS, P_TRUE

T = 300


RATE = {'p': P_TRUE}


def make(i):
    return im1.Im1(seed=10000 + i, **dict(DIS, dis_rate=RATE['p']))


def make_rare(i):
    return im1.Im1(seed=10000 + i, **dict(DIS, dis_rate=0.005))


def naive_one(args):
    level, seed, p = args
    e = im1.Im1(seed=seed, **dict(DIS, dis_rate=p))
    hit, steps = im_time.run_until(e, T, level, im_time.MEASURES['b'], 100)
    return hit, steps


if __name__ == '__main__':
    print('S17b 平行世界(多段の分割): 100〜300 年のうちに国債/所得が水準を超える確率。危機に近づいた世界(途中の閾値を超えた世界)を複製し、乱数を替えて未来を分ける')
    print('   素朴な推定(1 本ずつ最後まで)と、同じくらいの計算(回した年数の和)の平行世界の推定を、5 回くり返して揺れを比べる')
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as pool:
        for prate, target, levels in ((P_TRUE, 0.8, (0.5, 0.6, 0.7, 0.8)), (P_TRUE, 1.0, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)), (P_TRUE, 1.2, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)),
                                      (0.005, 0.8, (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8))):
            t0 = time.time()
            est, cost = [], []
            for rep in range(5):
                p, ps, tot = im_time.splitting(make if prate == P_TRUE else make_rare, levels, T, 80, seed=rep, pool=pool)
                est.append(p); cost.append(tot)
            ts = time.time() - t0
            budget = int(np.mean(cost))
            n_naive = max(budget // 200, 10)
            t0 = time.time()
            nv = []
            for rep in range(5):
                res = list(pool.map(naive_one, [(target, 100000 * (rep + 1) + s, prate) for s in range(n_naive)]))
                nv.append(np.mean([r[0] for r in res]))
            tn = time.time() - t0
            print(f'  災害 {prate:.1%}/年、国債/所得 > {target}: 平行世界 {np.mean(est):.4f} ± {np.std(est):.4f}(5 回の標準偏差、一回 {budget} 年ぶん、段ごとの割合の例 {np.round(ps, 2).tolist()})'
                  f' | 素朴 {np.mean(nv):.4f} ± {np.std(nv):.4f}(一回 {n_naive} 本 ≈ 同じ年数) | 時間 {ts:.0f} 秒 / {tn:.0f} 秒')
