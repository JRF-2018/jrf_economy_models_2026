# results_stage2.py: 段 2 の実験(土地と知財)。python -W ignore results_stage2.py > results_stage2.txt
import numpy as np
import im1

BASE = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, fiscal='deficit')   # 記録は「赤字を 0 に」のルールで出した
BURN = 300


def run(T, events=(), **kw):
    """events: (年, dict) の列。その年のはじめにパラメータを変える。"""
    e = im1.Im1(**dict(BASE, **kw))
    rec = []
    for t in range(T):
        for ts, ch in events:
            if ts == t:
                e.p.set(**ch)
        rec.append(e.step())
    a = {k: np.array([r[k] for r in rec], dtype=float) for k in rec[0] if k != 'inv_limited'}
    return a, e


def mean_last(a, k, n=100):
    return float(np.mean(a[k][-n:]))


def s1_base():
    print('S1 基準(400 年、最後の 100 年の平均)。地代・使用料・世代の入れ替わり・資産効果 0.005 を入れた土地あり、と土地なし')
    for lab, kw in (('土地なし(段 1)', dict(land=False)), ('土地あり', dict(c_h=0.005))):
        a, e = run(400, **kw)
        Y = a['Y'][-100:]
        line = f'  {lab:14s} 雇用 {mean_last(a, "E"):.0f}  実質賃金 {mean_last(a, "real_wage"):.2f}  所得の上位 10% {mean_last(a, "top10_inc"):.3f}'
        if 'pT' in a:
            line += (f'  土地の値打ち/所得 {np.mean(a["land_value"][-100:] / Y):.2f}  家計の借金/所得 {np.mean(a["hh_debt"][-100:] / Y):.3f}'
                     f'  借り手 {mean_last(a, "n_borrowers"):.0f} 人  正味資産のジニ {mean_last(a, "gini_nw"):.3f}  上位 10% {mean_last(a, "top10_nw"):.3f}'
                     f'  貸倒れ/所得 {np.mean(a["wo_h"][-100:] / Y):.4f}')
        print(line + f'  帳簿 {a["books"].max():.1e}  標本の差 {a["sample_gap"].max():.1e}' +
              (f'  土地の和のずれ {a["land_sum_err"].max():.1e}' if 'land_sum_err' in a else ''))


def euphoria_events(t0=BURN, boost=0.15, hike=0.06):
    return [(t0, dict(theta_shock=boost)), (t0 + 10, dict(theta_shock=0.0, r_cb=hike)), (t0 + 15, dict(r_cb=0.02))]


def shock_stats(a, t0=BURN):
    pre = slice(t0 - 20, t0)
    win = slice(t0, t0 + 40)
    real_pT = a['pT'] / a['PN']
    return dict(land_peak=real_pT[win].max() / real_pT[pre].mean(), land_bottom=real_pT[win].min() / real_pT[pre].mean(),
                E_bottom=a['E'][win].min() / a['E'][pre].mean(), wo=a['wo_h'][win].sum() / a['Y'][pre].mean(),
                bail=a['bailout'][win].sum() / a['Y'][pre].mean(), debt_peak=(a['hh_debt'] / a['Y'])[win].max())


def s2_euphoria():
    print(f'S2 ユーフォリア(300 年目から 10 年、土地に置きたい割合 +0.15)のあと、利上げ(政策金利 0.02 → 0.06 を 5 年)。担保の掛け目 λ_T を変える')
    print('   値は直前 20 年の平均に対する比(土地の値段は実質 = 土地の値段 / 必需品の価格)。貸倒れと救済は、直前の所得に対する 40 年の合計')
    for lt in (0.0, 0.6, 0.9):
        a, e = run(BURN + 60, events=euphoria_events(), lam_T=lt, c_h=0.005)
        s = shock_stats(a)
        print(f"  λ_T={lt:.1f}  土地の頂 ×{s['land_peak']:.2f}  底 ×{s['land_bottom']:.2f}  雇用の底 ×{s['E_bottom']:.3f}  "
              f"借金/所得の頂 {s['debt_peak']:.3f}  貸倒れ {s['wo']:.4f}  救済 {s['bail']:.4f}")


def s3_asset_effect():
    print('S3 資産効果 c_h(土地の正味の値打ちの割合を消費に足す)。同じユーフォリアと利上げ。λ_T = 0.6')
    for ch in (0.0, 0.005, 0.01, 0.02, 0.04):
        a, e = run(BURN + 60, events=euphoria_events(), lam_T=0.6, c_h=ch)
        s = shock_stats(a)
        print(f"  c_h={ch:.3f}  土地の頂 ×{s['land_peak']:.2f}  底 ×{s['land_bottom']:.2f}  雇用の底 ×{s['E_bottom']:.3f}  "
              f"借金/所得の頂 {s['debt_peak']:.3f}  貸倒れ {s['wo']:.4f}")


def s4_concentration():
    print('S4 貸し手の慎重さ(μ、θ = 0.15)と土地。土地があると、集中の回り道は強まるか')
    for land in (False, True):
        for mu in (0.0, 0.5, 1.0):
            a, e = run(400, land=land, mu_prem=mu, theta_prem=0.15, c_h=0.005)
            extra = f"  土地の値打ち/所得 {np.mean(a['land_value'][-100:] / a['Y'][-100:]):.2f}  正味資産の上位 10% {mean_last(a, 'top10_nw'):.3f}" if land else ''
            print(f"  土地{'あり' if land else 'なし'} μ={mu:.1f}  雇用 {mean_last(a, 'E'):.0f}  r̄ {mean_last(a, 'rbar'):.3f}  所得の上位 10% {mean_last(a, 'top10_inc'):.3f}{extra}")


def rank_corr(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def s5_inheritance():
    print('S5 世代の入れ替わりと相続(400 年。最後の 100 年の平均。順位の相関は、350 年目と 360 年目の一人あたりの正味資産)')
    for to, inh in ((0.0, 'lottery'), (0.025, 'lottery'), (0.025, 'equal'), (0.05, 'lottery')):
        e = im1.Im1(**dict(BASE, turnover=to, inherit=inh, c_h=0.005))
        recs, snap = [], {}
        for t in range(400):
            r = e.step()
            recs.append(r)
            if t in (350, 360):
                H = e.H
                snap[t] = (H.M + e.pT * H.land) / H.heads
        a = {k: np.array([r[k] for r in recs], dtype=float) for k in ('gini_nw', 'top10_nw', 'hh_debt', 'Y', 'n_borrowers', 'E')}
        print(f"  入れ替わり {to:.3f} 相続 {inh:8s}  正味資産のジニ {a['gini_nw'][-100:].mean():.3f}  上位 10% {a['top10_nw'][-100:].mean():.3f}  "
              f"借金/所得 {np.mean(a['hh_debt'][-100:] / a['Y'][-100:]):.3f}  借り手 {a['n_borrowers'][-100:].mean():.0f} 人  雇用 {a['E'][-100:].mean():.0f}  "
              f"10 年の順位の相関 {rank_corr(snap[350], snap[360]):.2f}")


if __name__ == '__main__':
    s1_base()
    s2_euphoria()
    s3_asset_effect()
    s4_concentration()
    s5_inheritance()
