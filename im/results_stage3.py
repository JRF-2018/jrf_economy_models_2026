# results_stage3.py: 段 3 の実験(二つの地域と輸出入、共通の通貨)。python -W ignore results_stage3.py > results_stage3.txt
import numpy as np
import im3

C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, fiscal='deficit')   # 記録は「赤字を 0 に」のルールで出した
AUTARKY = 30.0          # 輸送の費用がこれだけ大きいと、貿易は事実上ない(好みは同じまま)
CA = {'A': dict(gamma0=np.array([0.07, 0.3, 0.3])), 'B': dict(gamma0=np.array([0.1, 0.21, 0.3]))}   # A は必需品、B は贅沢品が得意


def last(a, k, n=100):
    return float(np.mean(a[k][-n:]))


def fmt(a, keys=('E', 'real_wage', 'top10_inc')):
    return '  '.join(f'{k}={last(a, k):.3f}' if abs(last(a, k)) < 100 else f'{k}={last(a, k):.0f}' for k in keys)


def trade_share(a):
    return np.mean((a['imports'][-100:]) / a['Y'][-100:])


def rel_books(a):
    return float(np.max(a['books'] / a['Y']))


def infl(a):
    return f"物価上昇率 {last(a, 'infl'):+.3f}±{np.std(a['infl'][-100:]):.3f}"


def t1_gains():
    print('T1 貿易の得: 同じ二つの地域(好みの重みは自地域 0.5)。輸送の費用 tc を振る(tc = 30 は事実上の閉鎖)。300 年、最後の 100 年の平均')
    for tc in (0.0, 0.1, 0.3, 1.0, AUTARKY):
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=tc)
        a = o['A']
        print(f"  tc={tc:5.1f}  {fmt(a)}  輸入/所得={trade_share(a):.3f}  {infl(a)}  帳簿のずれ/所得 {rel_books(a):.1e}")


def t2_comparative():
    print('T2 比較優位: A は必需品(一製品あたり労働 0.07)、B は贅沢品(0.21)が得意。輸送の費用 0.1 と、事実上の閉鎖')
    for tc in (0.1, AUTARKY):
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=tc, region_kw=CA)
        for nm in 'AB':
            a = o[nm]
            Y = a['Y'][-100:]
            print(f"  tc={tc:5.1f} {nm}  {fmt(a)}  必需品の輸出/所得 {np.mean(a['expN'][-100:] / Y):.3f}  贅沢品の輸出/所得 {np.mean(a['expL'][-100:] / Y):.3f}  "
                  f"貿易の収支/所得 {np.mean(a['tb'][-100:] / Y):+.4f}  銀行の間の貸し/所得 {a['interbank'][-1] / a['Y'][-1]:+.2f}  {infl(a)}")


def t3_crisis(t0=250):
    print(f'T3 危機時の生産: {t0} 年目に、A の必需品の生産能力が半分になる。A の必需品の実質の価格(買う価格/賃金)と実質賃金(10 年の頂と底を、直前 20 年の平均に対する比で)')
    for tc in (0.1, 0.3, AUTARKY):
        w = im3.World(common=C, home=0.5, tc=tc)
        for t in range(t0 + 40):
            if t == t0:
                w.R[0].K[0] *= 0.5
            w.step()
        A = {k: np.array([rs[0][k] for rs in w.log]) for k in ('bN', 'PN', 'W', 'real_wage', 'E', 'xN', 'imports')}
        A['bN'] = A['bN'] / A['W']                            # 必需品の実質の価格(賃金で割る。名目の漂流を除く)
        B = {k: np.array([rs[1][k] for rs in w.log]) for k in ('bN', 'real_wage', 'E', 'expN', 'Y')}
        B['expN'] = B['expN'] / B['Y']                        # 必需品の輸出の、所得に対する比
        pre = slice(t0 - 20, t0)
        pk = lambda d, k, f: f(d[k][t0:t0 + 10]) / d[k][pre].mean()
        print(f"  tc={tc:5.1f}  A: 必需品の実質の価格の頂 ×{pk(A, 'bN', np.max):.2f}  実質賃金の底 ×{pk(A, 'real_wage', np.min):.3f}  "
              f"雇用の底 ×{pk(A, 'E', np.min):.3f}   B: 実質賃金の底 ×{pk(B, 'real_wage', np.min):.3f}  必需品の輸出/所得の頂 ×{pk(B, 'expN', np.max):.2f}")


def t4_imbalance():
    print('T4 貿易の不均衡: A の家計がよく貯める(可処分所得からの消費性向 0.9 → 0.8)。輸送の費用 0.1')
    o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1, region_kw={'A': dict(c_y=0.8)})
    for nm in 'AB':
        a = o[nm]
        ib = a['interbank']
        print(f"  {nm}  {fmt(a)}  貿易の収支/所得 {np.mean(a['tb'][-100:] / a['Y'][-100:]):+.4f}  "
              f"銀行の間の貸し/所得: 100 年目 {ib[99] / a['Y'][99]:+.2f}、200 年目 {ib[199] / a['Y'][199]:+.2f}、300 年目 {ib[-1] / a['Y'][-1]:+.2f}")


def t5_spillover():
    print('T5 集中の波及: A だけで貸し手の慎重さ μ = 1(θ = 0.15)。B は μ = 0。輸送の費用 0.1 と、事実上の閉鎖')
    for tc in (0.1, AUTARKY):
        o, w = im3.simulate(T=300, common=C, home=0.5, tc=tc, region_kw={'A': dict(mu_prem=1.0, theta_prem=0.15)})
        print(f"  tc={tc:5.1f}  A: {fmt(o['A'], ('E', 'real_wage', 'top10_inc', 'rbar'))}   B: {fmt(o['B'], ('E', 'real_wage', 'top10_inc', 'rbar'))}")
    o, w = im3.simulate(T=300, common=C, home=0.5, tc=0.1)
    print(f"  比較(どちらも μ = 0、tc = 0.1)  B: {fmt(o['B'], ('E', 'real_wage', 'top10_inc', 'rbar'))}")


if __name__ == '__main__':
    t1_gains()
    t2_comparative()
    t3_crisis()
    t4_imbalance()
    t5_spillover()
