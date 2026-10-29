# results_stage3c.py: 段 3c の実験(三つの地域と、通貨の組み合わせ)。python -W ignore results_stage3c.py > results_stage3c.txt
import numpy as np
import im3

C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, fiscal='deficit')   # 記録は「赤字を 0 に」のルールで出した
R3 = ('A', 'B', 'C')
CA3 = {'A': dict(gamma0=np.array([0.07, 0.3, 0.3])), 'B': dict(gamma0=np.array([0.1, 0.21, 0.3])),
       'C': dict(gamma0=np.array([0.1, 0.3, 0.21]))}     # A は必需品、B は贅沢品、C は原料が得意
SYSTEMS = (('すべて共通', dict(currency='common')),
           ('A+B 同盟、C 変動', dict(currency={'A': 'x', 'B': 'x', 'C': 'y'}, regimes={'y': 'float'})),
           ('B+C 同盟、A 変動', dict(currency={'A': 'x', 'B': 'y', 'C': 'y'}, regimes={'y': 'float'})),
           ('すべて変動', dict(currency='float')))


def last(a, k, n=100):
    return float(np.mean(a[k][-n:]))


def sim(T=300, **kw):
    return im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1, **kw)


def c1_triangle():
    print('C1 三角の貿易: A は必需品、B は贅沢品、C は原料が得意(一製品あたり労働 0.07 / 0.21 / 0.21)。最後の 100 年の平均の、所得に対する比')
    for lab, kw in (SYSTEMS[0], SYSTEMS[3]):
        o, w = sim(region_kw=CA3, **kw)
        print(f'  {lab}')
        for nm in R3:
            a = o[nm]
            Y = a['Y'][-100:]
            bil = '  '.join(f"対 {q} {np.mean(a['bil_' + q][-100:] / Y):+.4f}" for q in R3 if q != nm)
            dpos = (a['interbank'][-1] - a['interbank'][-101]) / a['Y'][-100:].sum()
            print(f"    {nm}: 雇用 {last(a, 'E'):.0f}  実質賃金 {last(a, 'real_wage'):.3f}  収支 {np.mean(a['tb'][-100:] / Y):+.4f}  二国間の収支: {bil}  "
                  f"対外の持ち高の増え方(100 年の変化 / 100 年の所得の和) {dpos:+.4f}")


def c2_crisis(t0=250):
    print(f'C2 危機時の生産: {t0} 年目に A の必需品の生産能力が半分。10 年の底(直前 20 年の平均に対する比)。地域は同じ技術')
    for lab, kw in SYSTEMS:
        w = im3.World(regions=R3, common=C, home=0.5, tc=0.1, **kw)
        for t in range(t0 + 40):
            if t == t0:
                w.R[0].K[0] *= 0.5
            w.step()
        out = []
        for i, nm in enumerate(R3):
            E = np.array([rs[i]['E'] for rs in w.log]); rw = np.array([rs[i]['real_wage'] for rs in w.log])
            pre = slice(t0 - 20, t0)
            out.append(f"{nm}: 雇用 ×{E[t0:t0 + 10].min() / E[pre].mean():.3f} 実質賃金 ×{rw[t0:t0 + 10].min() / rw[pre].mean():.3f}")
        print(f'  {lab:14s} ' + '   '.join(out))


def c3_deficit():
    print('C3 財政の赤字: A が赤字/所得 3% を目標にする(B、C は 0)。最後の 100 年')
    for lab, kw in SYSTEMS:
        o, w = sim(region_kw={'A': dict(deficit_target=0.03)}, **kw)
        out = []
        for nm in R3:
            a = o[nm]
            out.append(f"{nm}: 雇用 {last(a, 'E'):.0f} 物価 {last(a, 'infl'):+.3f}/年 同盟の中の貸し {a['ib'][-1] / a['Y'][-1]:+.2f} 外の持ち高 {a['fxpos'][-1] / a['Y'][-1]:+.2f}")
        print(f'  {lab:14s} ' + '   '.join(out))


def c4_glut():
    print('C4 同盟の中の貯めすぎ: B の家計がよく貯める(消費性向 0.9 → 0.8)。A+B 同盟、C 変動 と、すべて共通')
    for lab, kw in (SYSTEMS[0], SYSTEMS[1]):
        o, w = sim(region_kw={'B': dict(c_y=0.8)}, **kw)
        out = []
        for nm in R3:
            a = o[nm]
            ib = a['ib']
            out.append(f"{nm}: 雇用 {last(a, 'E'):.0f} 同盟の中の貸し/所得 100 年 {ib[99] / a['Y'][99]:+.2f}・300 年 {ib[-1] / a['Y'][-1]:+.2f} "
                       f"外の持ち高 {a['fxpos'][-1] / a['Y'][-1]:+.2f}")
        print(f'  {lab:14s} ' + '   '.join(out))


if __name__ == '__main__':
    c1_triangle()
    c2_crisis()
    c3_deficit()
    c4_glut()
