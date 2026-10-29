# results_stage17.py: 段 17 の実験(時間の層の残り: 歴史の重点標本化と、粗い時間の進め方)。python -W ignore results_stage17.py > results_stage17.txt
# 方法の確かめなので、標本は 400 記入(速さのため)。財政は既定(利払いの重さ)、g_gamma = 0。
import os
import numpy as np
import time
import im1
import im_time
from concurrent.futures import ProcessPoolExecutor

DIS = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, relief='state',
           mu_B=0.3, theta_B=0.6, fiscal='burden', burden_target=0.016, phi_pi=1.0, rbar_link=0.3, theta_resolve=0.55)
T = 300
P_TRUE = 0.02
B_EVENT = 0.8


def hist(args):
    q, seed = args
    kw = dict(DIS, dis_rate=P_TRUE, dis_q=q if q != P_TRUE else 0.0)
    e = im1.Im1(seed=seed, **kw)
    crisis, brk = 0.0, 0.0
    try:
        for t in range(T):
            r = e.step()
            if t >= 100:
                if r['b'] > B_EVENT:
                    crisis = 1.0
                brk += r['breakups']
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        crisis = 1.0
    return crisis, brk, getattr(e, 'logw', 0.0)


def is_experiment():
    print(f'S17-1 歴史の重点標本化: 災害の本当の確率は年 {P_TRUE:.0%}。100〜{T} 年のうちに国債/所得が {B_EVENT} を超える(財政の苦しさ)確率を推定する')
    print('   (初めの 100 年は、初めの値からの移り変わりで国債/所得が高いので数えない。重みは 0 年目からの災害の引き方で決まる)')
    print('   素朴な推定(本当の確率で引く)と、重点標本化(多めの確率 q で引き、尤度の比で重みをつける)。それぞれ 400 本の歴史を、80 本ずつ 5 組に分けて推定の揺れを見る')
    N, G = 400, 5
    for q in (P_TRUE, 0.05, 0.1):
        jobs = [(q, s) for s in range(N)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
            res = list(ex.map(hist, jobs))
        c = np.array([r[0] for r in res]); b = np.array([r[1] for r in res]); w = np.exp(np.array([r[2] for r in res]))
        est = [(c[g::G] * w[g::G]).mean() for g in range(G)]
        estb = [(b[g::G] * w[g::G]).mean() for g in range(G)]
        ess = w.sum() ** 2 / (w ** 2).sum()
        lab = '素朴(q = 本当の確率)' if q == P_TRUE else f'重点標本化 q = {q:.2f}'
        print(f'  {lab:26s} | 破綻の確率 {np.mean(est):.4f} ± {np.std(est) / np.sqrt(G):.4f}(組の間の標準誤差) | 破綻した歴史 {int(c.sum())} 本 | '
              f'一家離散/200年 {np.mean(estb):.0f} ± {np.std(estb) / np.sqrt(G):.0f} | 重みの平均 {w.mean():.3f} | 実効の本数 {ess:.0f}/{N} | {time.time() - t0:.0f} 秒')


LEARN = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, relief='state',
             dis_rate=0.05, learn=True, learn_omega=1.0, prior_a=0.5, prior_b=100.0)


def proj(args):
    mode, seed, T2 = args
    e = im1.Im1(seed=seed, **LEARN)
    t0 = time.time()
    if mode == 'full':
        rec = []
        for t in range(T2):
            e.step(); rec.append((t + 1, e.belief, e.prot))
        steps = T2
    else:
        inner, jump, slow = mode
        steps, rec = im_time.projective(e, T2, inner=inner, jump=jump, slow=slow)
    rec = np.array(rec)
    at = lambda yr: (float(rec[np.searchsorted(rec[:, 0], yr), 1]), float(rec[np.searchsorted(rec[:, 0], yr), 2]))
    return dict(steps=steps, secs=time.time() - t0, b100=at(100)[0], b500=at(500)[0], b1000=at(1000)[0], p1000=at(1000)[1])


def proj_experiment():
    print('S17-2 粗い時間の進め方: 災害の頻度を忘れずに学ぶ社会(年 5% の災害、初めの信念は 0.5%)。1000 年の信念と守りの資本の道を、')
    print('   全部回す場合と、「n 年回して、遅い量の傾きで m 年飛ぶ」場合で比べる。遅い量は、信念の記録(災害の回数と年数)と守りの資本、または信念の記録だけ。乱数の種 12 個')
    T2 = 1000
    print('  進め方                 | 回した年数 | 時間(秒、一本) | 信念 100 年目 | 500 年目 | 1000 年目 | 守りの資本 1000 年目')
    B = ('n_eff', 't_eff')
    for mode, lab in (('full', '全部回す'), ((10, 40, im_time.SLOW), '10 回し 40 飛ぶ、守りも飛ばす'), ((20, 80, im_time.SLOW), '20 回し 80 飛ぶ、守りも飛ばす'),
                      ((10, 40, B), '10 回し 40 飛ぶ、信念だけ'), ((10, 90, B), '10 回し 90 飛ぶ、信念だけ'), ((20, 80, B), '20 回し 80 飛ぶ、信念だけ')):
        with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
            res = list(ex.map(proj, [(mode, s, T2) for s in range(12)]))
        mm = {k: np.mean([r[k] for r in res]) for k in res[0]}
        sd = {k: np.std([r[k] for r in res]) / np.sqrt(len(res)) for k in res[0]}
        print(f"  {lab:22s} | {mm['steps']:.0f} | {mm['secs']:.1f} | {mm['b100']:.4f}±{sd['b100']:.4f} | {mm['b500']:.4f}±{sd['b500']:.4f} | "
              f"{mm['b1000']:.4f}±{sd['b1000']:.4f} | {mm['p1000']:.1f}±{sd['p1000']:.1f}")


if __name__ == '__main__':
    is_experiment()
    proj_experiment()
