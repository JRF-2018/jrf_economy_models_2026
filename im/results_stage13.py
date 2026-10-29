# results_stage13.py: 段 13 の実験(信念の区間と三つの勘定)。python -W ignore results_stage13.py > results_stage13.txt
# 標本 2000 記入、乱数の種 12 個。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage5 import batch, m

L = dict(dis_rate=0.01, learn=True)


def b1_interval():
    print('B1 信念の区間(ノート I 第 5 章)。災害の頻度が時代で変わる(段 5 の L1 と同じ: 300 年目までは毎年 1%、そこから 10%、450 年目から 1%)')
    print('   区間 = 事後の分布の 5〜95%、床 = 区間を少なくとも平均 × (1 ± 床) に広げる(構造の不確かさ)。α = 1 は最悪に備える(maxmin)、0.5 は真ん中')
    cases = (('中立(平均)、ω = 0.95', dict(L, learn_omega=0.95)),
             ('悲観(上側 90%)、ω = 0.95', dict(L, learn_omega=0.95, learn_attitude='pessimistic')),
             ('区間 α=1、床 0、ω = 0.95', dict(L, learn_omega=0.95, learn_attitude='interval')),
             ('区間 α=1、床 0.5、ω = 0.95', dict(L, learn_omega=0.95, learn_attitude='interval', amb_floor=0.5)),
             ('区間 α=0.5、床 0.5、ω = 0.95', dict(L, learn_omega=0.95, learn_attitude='interval', amb_floor=0.5, amb_alpha=0.5)),
             ('中立、忘れない(ω = 1)', dict(L, learn_omega=1.0)),
             ('区間 α=1、床 0、忘れない', dict(L, learn_omega=1.0, learn_attitude='interval')),
             ('区間 α=1、床 0.5、忘れない', dict(L, learn_omega=1.0, learn_attitude='interval', amb_floor=0.5)),
             ('区間 α=1、床 1.0、忘れない', dict(L, learn_omega=1.0, learn_attitude='interval', amb_floor=1.0)))
    ev = [(300, dict(dis_rate=0.10)), (450, dict(dis_rate=0.01))]
    out = batch(cases, 600, ev)
    print('  社会                          | 守り(静か 200-300 / 荒れ 300-450 / 静か 450-600) | 実質の生産(静か / 荒れ / 静か) | 一家離散/100年(荒れ / 静か)')
    for lab, rs in out.items():
        pq = [m(rs, lambda r, s=s: r['protect'][s].mean())[0] for s in (slice(200, 300), slice(300, 450), slice(450, 600))]
        ry = [m(rs, lambda r, s=s: (r['Y'][s] / r['bN'][s]).mean())[0] for s in (slice(200, 300), slice(300, 450), slice(450, 600))]
        bk = [m(rs, lambda r, s=s: r['breakups'][s].sum() / (s.stop - s.start) * 100) for s in (slice(300, 450), slice(450, 600))]
        print(f"  {lab:28s} | {pq[0]:.4f} / {pq[1]:.4f} / {pq[2]:.4f} | {ry[0]:.0f} / {ry[1]:.0f} / {ry[2]:.0f} | "
              f"{bk[0][0]:.0f}±{bk[0][1]:.0f} / {bk[1][0]:.0f}±{bk[1][1]:.0f}")


BASE2 = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005)
KEYS = ('acct_use', 'acct_lux', 'acct_save', 'acct_explore', 'Y', 'E', 'n_var', 'res_w', 'ip_new', 'bN')


def crisis(args):
    kw, seed = args
    out = []
    for sh in (True, False):
        e = im1.Im1(seed=seed, **dict(BASE2, **kw))
        rec = []
        for t in range(330):
            if sh and t == 300:
                e.K[0] *= 0.5
            rec.append(e.step())
        out.append({k: np.array([r.get(k, np.nan) for r in rec], float) for k in KEYS})
    return out


def b2_accounts():
    print('B2 三つの勘定(使う = 必需品、探す = 贅沢品のうち自主研究の分 + 企業の研究の賃金、貯める = 可処分所得 − 支出)。300 年目に必需品の能力が半分')
    print('   所得に対する割合の、打撃のない同じ種の歴史との差(最初の 5 年の平均、ポイント)。実質 = 必需品の値段で割った量の比')
    cases = (('研究なし', dict()), ('研究 2%', dict(research=True, s_R=0.02)), ('研究 2% + 文化(ψ=1 ε=2)', dict(research=True, s_R=0.02, psi_v=1.0, eps_v=2.0)))
    jobs = [(kw, s) for lab, kw in cases for s in range(12)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(crisis, jobs))
    print('  場合                         | 使う/所得の差 | 贅沢品/所得の差 | 探す/所得の差 | 貯める/所得の差 | 実質の使う | 実質の贅沢品 | 実質の探す | 30 年の実質の探すの損(打撃の前の一年分に対する比) | 30 年後の種類の数の比')
    W5 = slice(300, 305)
    for i, (lab, kw) in enumerate(cases):
        rs = res[i * 12:(i + 1) * 12]
        sh = lambda k: np.mean([(a[k][W5] / a['Y'][W5]).mean() - (z[k][W5] / z['Y'][W5]).mean() for a, z in rs]) * 100
        rl = lambda k: np.mean([(a[k][W5] / a['bN'][W5]).sum() / (z[k][W5] / z['bN'][W5]).sum() for a, z in rs])
        cum = np.mean([((a['acct_explore'] / a['bN'])[300:330] - (z['acct_explore'] / z['bN'])[300:330]).sum() / (z['acct_explore'][299] / z['bN'][299]) for a, z in rs])
        nv = np.mean([a['n_var'][329] / z['n_var'][329] for a, z in rs]) if 'psi_v' in kw else np.nan
        print(f"  {lab:28s} | {sh('acct_use'):+.2f} | {sh('acct_lux'):+.2f} | {sh('acct_explore'):+.2f} | {sh('acct_save'):+.2f} | "
              f"{rl('acct_use'):.3f} | {rl('acct_lux'):.3f} | {rl('acct_explore'):.3f} | {cum:+.2f} | {nv:.3f}")


if __name__ == '__main__':
    b1_interval()
    b2_accounts()
