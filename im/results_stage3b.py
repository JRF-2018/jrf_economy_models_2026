# results_stage3b.py: 段 3b の実験(別々の通貨と為替)。python -W ignore results_stage3b.py > results_stage3b.txt
# 通貨の制度を三つ比べる: 共通の通貨(段 3a)、別々の通貨で為替を固定、別々の通貨で為替が変動(毎年の貿易の収支が 0)。
import numpy as np
import im3

C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, fiscal='deficit')   # 記録は「赤字を 0 に」のルールで出した
REGIMES = ('common', 'fixed', 'float')
LABEL = {'common': '共通の通貨', 'fixed': '固定', 'float': '変動'}
CA = {'A': dict(gamma0=np.array([0.07, 0.3, 0.3])), 'B': dict(gamma0=np.array([0.1, 0.21, 0.3]))}


def last(a, k, n=100):
    return float(np.mean(a[k][-n:]))


def row(o, keys=('E', 'real_wage')):
    out = []
    for nm in 'AB':
        a = o[nm]
        out.append(f"{nm}: " + ' '.join(f'{k}={last(a, k):.3f}' if abs(last(a, k)) < 100 else f'{k}={last(a, k):.0f}' for k in keys))
    a = o['A']
    out.append(f"A の貿易の収支/所得 {np.mean(a['tb'][-100:] / a['Y'][-100:]):+.4f}  A の対外の持ち高/所得 {a['interbank'][-1] / a['Y'][-1]:+.2f}  "
               f"実質の為替 {np.mean(real_x(o)[-100:]):.3f}")
    return '  '.join(out)


def real_x(o):
    """実質の為替 q = e × (B の必需品の買う価格)/(A の必需品の買う価格)。B の財が A から見てどれだけ高いか。名目の漂流を除く。"""
    return o['A']['e'] * o['B']['bN'] / o['A']['bN']


def f1_imbalance():
    print('F1 貿易の不均衡: A の家計がよく貯める(消費性向 0.9 → 0.8)。輸送の費用 0.1。300 年、最後の 100 年')
    for cur in REGIMES:
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1, region_kw={'A': dict(c_y=0.8)}, currency=cur)
        print(f'  {LABEL[cur]:6s} {row(o)}  帳簿のずれ/所得 {max(np.max(o[n]["books"] / o[n]["Y"]) for n in "AB"):.1e}')


def f2_comparative():
    print('F2 比較優位(A は必需品、B は贅沢品が得意)。輸送の費用 0.1')
    for cur in REGIMES:
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1, region_kw=CA, currency=cur)
        print(f'  {LABEL[cur]:6s} {row(o)}')


def f3_crisis(t0=250):
    print(f'F3 危機時の生産: {t0} 年目に A の必需品の生産能力が半分。10 年の頂と底(直前 20 年の平均に対する比)')
    for cur in REGIMES:
        w = im3.World(common=C, home=0.5, tc=0.1, currency=cur)
        es = []
        for t in range(t0 + 40):
            if t == t0:
                w.R[0].K[0] *= 0.5
            rs = w.step()
            es.append(w.e[w.cur[1]] / w.e[w.cur[0]] * rs[1]['bN'] / rs[0]['bN'])
        A = {k: np.array([rs[0][k] for rs in w.log]) for k in ('bN', 'W', 'real_wage', 'E')}
        B = {k: np.array([rs[1][k] for rs in w.log]) for k in ('real_wage', 'E')}
        es = np.array(es)
        pre = slice(t0 - 20, t0)
        f = lambda d, k, g: g(d[k][t0:t0 + 10]) / d[k][pre].mean()
        realp = A['bN'] / A['W']
        print(f"  {LABEL[cur]:6s} A: 必需品の実質の価格の頂 ×{realp[t0:t0 + 10].max() / realp[pre].mean():.2f}  実質賃金の底 ×{f(A, 'real_wage', np.min):.3f}  "
              f"雇用の底 ×{f(A, 'E', np.min):.3f}   B: 実質賃金の底 ×{f(B, 'real_wage', np.min):.3f}  雇用の底 ×{f(B, 'E', np.min):.3f}   "
              f"実質の為替の動き ×{es[t0:t0 + 10].min() / es[pre].mean():.3f}〜×{es[t0:t0 + 10].max() / es[pre].mean():.3f}")


def f4_spillover():
    print('F4 集中の波及: A だけで貸し手の慎重さ μ = 1(θ = 0.15)。輸送の費用 0.1')
    for cur in REGIMES:
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1, region_kw={'A': dict(mu_prem=1.0, theta_prem=0.15)}, currency=cur)
        print(f"  {LABEL[cur]:6s} {row(o, ('E', 'real_wage', 'top10_inc'))}")


def f5_inflation():
    print('F5 物価上昇率の違い: A の財政が、赤字/所得 3% を目標にする(B は 0)。輸送の費用 0.1。為替は物価の比を追うか(購買力平価)')
    for cur in REGIMES:
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1, region_kw={'A': dict(deficit_target=0.03)}, currency=cur)
        a, b = o['A'], o['B']
        rel = np.log(a['PN'][-1] / b['PN'][-1]) - np.log(a['PN'][-101] / b['PN'][-101])
        de = np.log(a['e'][-1]) - np.log(a['e'][-101])
        print(f"  {LABEL[cur]:6s} {row(o)}  物価上昇率 A {last(a, 'infl'):+.4f}、B {last(b, 'infl'):+.4f}  "
              f"100 年の物価の比の変化(対数){rel:+.2f}、為替の変化(対数){de:+.2f}")


if __name__ == '__main__':
    f1_imbalance()
    f2_comparative()
    f3_crisis()
    f4_spillover()
    f5_inflation()
