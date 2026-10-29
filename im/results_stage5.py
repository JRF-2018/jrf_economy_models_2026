# results_stage5.py: 段 5 の実験(学習と政治)。python -W ignore results_stage5.py > results_stage5.txt
# 標本 2000 記入(JRF 2026-10-28)。財政のルールは既定の「利払いの重さ」、金利は固定。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, relief='state')
SEEDS = range(12)


def run(args):
    kw, seed, T, events, prior = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    if prior is not None:
        e.n_eff, e.t_eff = prior
    try:
        rec = []
        for t in range(T):
            for ts, ch in events:
                if ts == t:
                    e.p.set(**ch)
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError):
        return None
    keys = ('Y', 'bN', 'breakups', 'E', 'protect', 'belief', 'dis_size', 'tau_k', 'top10_nw', 'gini_nw', 'real_wage',
            'turnout_w', 'turnout_k', 'n_rentier', 'top10_inc')
    return {k: np.array([r.get(k, np.nan) for r in rec], float) for k in keys}


def batch(cases, T, events=(), prior=None):
    jobs = [(kw, s, T, events, prior(s) if callable(prior) else prior) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(run, jobs))
    out = {}
    for i, (lab, kw) in enumerate(cases):
        out[lab] = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
    return out


def m(rs, f):
    v = np.array([f(r) for r in rs])
    return v.mean(), v.std() / np.sqrt(len(v))


def l1_forgetting():
    print('L1 忘れること。災害の頻度が時代で変わる: 300 年目までは毎年 1%、そこから 10%。450 年目から 1% に戻る。')
    print('   守りの量 = min(5%, 0.2 × 信じている頻度)。初めの信念は頻度 5%(重み 10 年)。乱数の種 12 個の平均 ± 標準誤差')
    cases = (('守りなし', dict(dis_rate=0.01)),
             ('守りを固定 1%', dict(dis_rate=0.01, protect=0.01)),
             ('学ぶ、忘れない(ω = 1)', dict(dis_rate=0.01, learn=True)),
             ('学ぶ、ω = 0.95(記録 20 年)', dict(dis_rate=0.01, learn=True, learn_omega=0.95)),
             ('学ぶ、ω = 0.8(記録 5 年)', dict(dis_rate=0.01, learn=True, learn_omega=0.8)),
             ('学ぶ、ω = 0.95、悲観', dict(dis_rate=0.01, learn=True, learn_omega=0.95, learn_attitude='pessimistic')),
             ('学ぶ、ω = 0.95、楽観', dict(dis_rate=0.01, learn=True, learn_omega=0.95, learn_attitude='optimistic')))
    ev = [(300, dict(dis_rate=0.10)), (450, dict(dis_rate=0.01))]
    out = batch(cases, 600, ev)
    print('  社会                         | 守り(静か 200-300 / 荒れ 300-450 / 静か 450-600) | 実質の生産(静か / 荒れ / 静か) | 一家離散/100年(荒れ / 静か)')
    for lab, rs in out.items():
        pq = [m(rs, lambda r, s=s: r['protect'][s].mean())[0] for s in (slice(200, 300), slice(300, 450), slice(450, 600))]
        ry = [m(rs, lambda r, s=s: (r['Y'][s] / r['bN'][s]).mean())[0] for s in (slice(200, 300), slice(300, 450), slice(450, 600))]
        bk = [m(rs, lambda r, s=s: r['breakups'][s].sum() / (s.stop - s.start) * 100) for s in (slice(300, 450), slice(450, 600))]
        print(f"  {lab:28s} | {pq[0]:.4f} / {pq[1]:.4f} / {pq[2]:.4f} | {ry[0]:.0f} / {ry[1]:.0f} / {ry[2]:.0f} | "
              f"{bk[0][0]:.0f}±{bk[0][1]:.0f} / {bk[1][0]:.0f}±{bk[1][1]:.0f}")


def l2_pretrain():
    print('L2 事前学習。若い社会(0 年目から)。災害の頻度は毎年 5%。100〜200 年目を比べる。乱数の種 12 個')
    cases = (('楽観的な初めの信念(頻度 0.5%、重み 20 年)', dict(dis_rate=0.05, learn=True, prior_a=0.1, prior_b=20.0)),)
    out_naive = batch(cases, 200)
    out_pre = batch((('縮めた模型の 3000 年の記録から', dict(dis_rate=0.05, learn=True, prior_a=0.1, prior_b=20.0)),), 200,
                    prior=lambda s: im1.pretrain(0.05, 3000, 1.0, s))
    out_pre95 = batch((('同じ、ただし記録は ω = 0.99 で割り引く', dict(dis_rate=0.05, learn=True, prior_a=0.1, prior_b=20.0, learn_omega=0.99)),), 200,
                      prior=lambda s: im1.pretrain(0.05, 3000, 0.99, s))
    for out in (out_naive, out_pre, out_pre95):
        for lab, rs in out.items():
            sl0, sl = slice(0, 50), slice(100, 200)
            p0 = m(rs, lambda r: r['protect'][sl0].mean())[0]
            ry = m(rs, lambda r: (r['Y'][sl] / r['bN'][sl]).mean())
            bk = m(rs, lambda r: r['breakups'][sl].sum())
            bk0 = m(rs, lambda r: r['breakups'][sl0].sum() * 2)
            print(f"  {lab:40s} 初めの 50 年の守り {p0:.4f}  一家離散(初めの 50 年、100 年あたり){bk0[0]:.0f}±{bk0[1]:.0f}  "
                  f"100〜200 年: 実質の生産 {ry[0]:.0f}±{ry[1]:.0f}  一家離散 {bk[0]:.0f}±{bk[1]:.0f}")


def v1_vote():
    print('V1 投票で資本所得への税(0.1 か 0.5。税収は一人あたり同じ額で配る)を 10 年ごとに決める。災害なし。400 年の後半 200 年')
    print('   群れの中の相関 ρ(ノート M 第 4 章: 投票の効き目は √(ρ/π))。投票の費用の平均は一人あたりの所得の 1%')
    rows = []
    for rk in (1e-4, 1e-2):
        for rw in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            rows.append((f'資産家 ρ_k = {rk:g}、労働者 ρ_w = {rw:g}', dict(disaster=False, vote=True, rho_w=rw, rho_k=rk)))
    out = batch(rows, 400)
    print('  群れの相関                          | 高い税の年の割合 | 投票率 労働者 / 資産家 | 資産家の人数 | 雇用 | 実質の生産 | 正味資産の上位10% | 所得の上位10%')
    for lab, rs in out.items():
        sl = slice(200, 400)
        hi = m(rs, lambda r: np.mean(r['tau_k'][sl] > 0.3))
        tw = m(rs, lambda r: np.nanmean(r['turnout_w'][sl]))[0]
        tk = m(rs, lambda r: np.nanmean(r['turnout_k'][sl]))[0]
        nr = m(rs, lambda r: np.nanmean(r['n_rentier'][sl]))[0]
        E = m(rs, lambda r: r['E'][sl].mean())[0]
        ry = m(rs, lambda r: (r['Y'][sl] / r['bN'][sl]).mean())[0]
        t10 = m(rs, lambda r: r['top10_nw'][sl].mean())[0]
        t10i = m(rs, lambda r: r['top10_inc'][sl].mean())[0]
        print(f"  {lab:34s} | {hi[0]:.2f}±{hi[1]:.2f} | {tw:.3f} / {tk:.3f} | {nr:.0f} | {E:.0f} | {ry:.0f} | {t10:.3f} | {t10i:.3f}")


def v2_regime():
    print('V2 投資の行動の仮定を変えると、選ばれた政策の効き目はどう変わるか(資産家 ρ_k = 0.01。労働者の相関が弱い / 強い)')
    rows = []
    for lab_b, kb in (('基準(φ_r = 0.5、φ_u = 0.3)', {}), ('需要主導(φ_r = 0.2、φ_u = 0.6)', dict(phi_r=0.2, phi_u=0.6))):
        for rw in (1e-6, 1e-2):
            rows.append((f'{lab_b}、ρ_w = {rw:g}', dict(disaster=False, vote=True, rho_w=rw, rho_k=1e-2, **kb)))
    out = batch(rows, 400)
    for lab, rs in out.items():
        sl = slice(200, 400)
        hi = m(rs, lambda r: np.mean(r['tau_k'][sl] > 0.3))[0]
        E = m(rs, lambda r: r['E'][sl].mean())
        ry = m(rs, lambda r: (r['Y'][sl] / r['bN'][sl]).mean())
        t10 = m(rs, lambda r: r['top10_nw'][sl].mean())[0]
        rw_ = m(rs, lambda r: r['real_wage'][sl].mean())[0]
        print(f"  {lab:44s} 高い税の年の割合 {hi:.2f}  雇用 {E[0]:.0f}±{E[1]:.0f}  実質の生産 {ry[0]:.0f}±{ry[1]:.0f}  実質賃金 {rw_:.3f}  正味資産の上位10% {t10:.3f}")


if __name__ == '__main__':
    l1_forgetting()
    l2_pretrain()
    v1_vote()
    v2_regime()
