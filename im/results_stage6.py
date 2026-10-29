# results_stage6.py: 段 6 の実験(資本の移動と財政同盟、富裕税と国際化)。python -W ignore results_stage6.py > results_stage6.txt
# 二つの地域 A、B(標本 2000 記入ずつ、家計の乱数の種は A = s、B = s + 100)。財政は既定(利払いの重さ)、g_gamma = 0。
# 出来事は 200 年目。同じ種で出来事のない歴史と比べる(差は 200〜260 年の平均)。乱数の種 6 組。± は平均の標準誤差。
import os
import numpy as np
import im3
from concurrent.futures import ProcessPoolExecutor

C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
INTL = dict(theta_F0=0.1, phi_K=1.0, chi_F=0.05)          # 国際化した家計(正味資産の 1 割を外に置きたい、ゆっくり動かす)
SEEDS = range(6)
T0, T1 = 200, 260
KEYS = ('E', 'Y', 'bN', 'real_wage', 'infl', 'tb_Y', 'e', 'b', 'levy', 'top10_nw', 'gini_nw', 'interbank', 'tau', 'union_in')


def world(seed, currency, capital, common, region_kw=None, union=None):
    return im3.World(common=common, region_kw=region_kw or {}, currency=currency, capital=capital, Ns=2000,
                     seeds={'A': seed, 'B': seed + 100}, union=union)


def history(args):
    """(設定, 出来事) の一つの歴史。出来事: 'none'、'hike'(A の政策金利 0.02 → 0.03)、'crisis'(A の必需品の生産能力が半分)、
    'wtax'(A が恒常の富裕税 2%)、'wtax2'(A と B の両方が 2%)。"""
    seed, currency, capital, common, region_kw, union, event = args
    w = world(seed, currency, capital, common, region_kw, union)
    rec = []
    try:
        for t in range(T1):
            if t == T0:
                if event == 'hike':
                    w.R[0].p.set(r_cb=0.03)
                elif event == 'crisis':
                    w.R[0].K[0] *= 0.5
                elif event == 'wtax':
                    w.R[0].p.set(wealth_tax=0.02)
                elif event == 'wtax2':
                    w.R[0].p.set(wealth_tax=0.02); w.R[1].p.set(wealth_tax=0.02)
            rec.append(w.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    out = {}
    for i, nm in enumerate('AB'):
        for k in KEYS + (('F_Y', 'KF_Y') if capital else ()):
            out[f'{nm}_{k}'] = np.array([r[i].get(k, 0.0) for r in rec], float)
    out['books'] = max(max(r[i]['books'] / r[i]['Y'] for i in range(2)) for r in rec)
    out['gap'] = max(max(r[i]['sample_gap'] / r[i]['Y'] for i in range(2)) for r in rec)
    return out


def pairs(cases, events):
    """cases: (ラベル, currency, capital, common, region_kw, union)。events: (処置, 比べ)。種ごとに、処置と比べの二つの歴史。"""
    jobs = []
    for case in cases:
        lab, cur, cap, com, rkw, un = case[:6]
        evs = (case[6], events[1]) if len(case) > 6 else events     # 7 番目があれば、その場合だけ処置を替える
        for s in SEEDS:
            for ev in evs:
                jobs.append((s, cur, cap, com, rkw, un, ev))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(history, jobs))
    out, k = {}, 0
    for lab, *_ in cases:
        ps = []
        for s in SEEDS:
            a, z = res[k], res[k + 1]
            k += 2
            if a is not None and z is not None:
                ps.append((a, z))
        out[lab] = ps
    return out


def stat(ps, f):
    v = np.array([f(a, z) for a, z in ps])
    return v.mean(), v.std() / np.sqrt(max(len(v), 1))


W = slice(T0, T1)
E10 = slice(T0, T0 + 10)


def rel(k, nm='A', sl=W):
    return lambda a, z: a[f'{nm}_{k}'][sl].mean() / z[f'{nm}_{k}'][sl].mean() - 1


def dif(k, nm='A', sl=W):
    return lambda a, z: a[f'{nm}_{k}'][sl].mean() - z[f'{nm}_{k}'][sl].mean()


def dlne(sl=W):
    """為替の、一年あたりの対数の変化(処置 − 比べ)。正なら A の通貨が弱くなる。"""
    def f(a, z):
        n = sl.stop - sl.start
        g = lambda h: np.log(h['A_e'][sl.stop - 1] / h['A_e'][sl.start - 1]) / n
        return g(a) - g(z)
    return f


def realY(nm='A', sl=W):
    return lambda a, z: (a[f'{nm}_Y'] / a[f'{nm}_bN'])[sl].mean() / (z[f'{nm}_Y'] / z[f'{nm}_bN'])[sl].mean() - 1


def fmt(ps, f, spec='+.3f'):
    m, s = stat(ps, f)
    return f'{m:{spec}}±{s:{spec[1:] if spec[0] == "+" else spec}}'


def k0_speed():
    print('K0 資本の動く速さと、変動の為替の揺れ(出来事なし、100〜300 年)。外に置きたい割合の基準 θ_F0、望む保有へ近づく割合 χ_F')
    cases = [('資本の移動なし', False, dict(C))]
    for th, chi in ((0.05, 0.2), (0.1, 0.2), (0.1, 0.1), (0.1, 0.05), (0.2, 0.05)):
        cases.append((f'θ_F0={th} χ_F={chi}', True, dict(C, theta_F0=th, phi_K=1.0, chi_F=chi)))
    jobs = [(s, 'float', cap, com) for lab, cap, com in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(k0_one, jobs))
    print('  場合                 | 為替の年の変化の標準偏差 | 実質の為替(対数)の標準偏差 | A の貿易の収支/所得の標準偏差 | 雇用 A | 雇用の標準偏差 | 物価 | 外の資産/所得')
    for i, (lab, cap, com) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"  {lab:20s} | {m['sde']:.3f} | {m['sdq']:.3f} | {m['sdtb']:.4f} | {m['E']:.0f} | {m['sdE']:.0f} | {m['infl']:+.3f} | {m['F_Y']:.2f}"
              + (f'  ({len(rs)}/{len(SEEDS)} の種だけ最後まで)' if len(rs) < len(SEEDS) else ''))


def k0_one(args):
    seed, cur, cap, com = args
    w = world(seed, cur, cap, com)
    try:
        R = [w.step() for _ in range(300)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda i, k: np.array([r[i].get(k, 0.0) for r in R[100:]], float)
    q = g(0, 'e') * g(1, 'bN') / g(0, 'bN')
    return dict(sde=np.std(np.diff(np.log(g(0, 'e')))), sdq=np.std(np.log(q)), sdtb=np.std(g(0, 'tb_Y')), E=g(0, 'E').mean(),
                sdE=g(0, 'E').std(), infl=g(0, 'infl').mean(), F_Y=g(0, 'F_Y').mean() if cap else 0.0)


def k1_hike():
    print('K1 A の政策金利を 0.02 → 0.03 に上げる(200 年目から、ずっと)。変動の為替。差は、同じ種で上げない歴史との 200〜260 年の平均')
    print('   為替の変化 = B の通貨一単位の A の通貨での値の、一年あたりの対数の変化(負なら A の通貨が強くなる)。B の外の資産 = B の家計が持つ A の預金/B の所得')
    cases = (('資本の移動なし', 'float', False, dict(C), None, None),
             ('資本の移動 φ_K=1', 'float', True, dict(C, **INTL), None, None),
             ('資本の移動 φ_K=3', 'float', True, dict(C, **dict(INTL, phi_K=3.0)), None, None))
    out = pairs(cases, ('hike', 'none'))
    print('  場合              | 為替の変化(最初の 10 年) | 為替の変化(60 年) | A の貿易の収支/所得 | B の外の資産/所得の差 | A の雇用 | B の雇用 | A の実質の生産 | A の物価 | B の物価 | 帳簿')
    for lab, ps in out.items():
        print(f"  {lab:16s} | {fmt(ps, dlne(E10), '+.4f')} | {fmt(ps, dlne(), '+.4f')} | {fmt(ps, dif('tb_Y'), '+.4f')} | "
              f"{fmt(ps, dif('F_Y', 'B')) if 'B_F_Y' in ps[0][0] else '  —  '} | {fmt(ps, rel('E'))} | {fmt(ps, rel('E', 'B'))} | {fmt(ps, realY())} | "
              f"{fmt(ps, dif('infl'), '+.4f')} | {fmt(ps, dif('infl', 'B'), '+.4f')} | {max(a['books'] for a, z in ps):.0e}")


def k2_crisis():
    print('K2 A の必需品の生産能力が半分(200 年目)。最初の 10 年の、雇用と実質賃金の比(同じ種で打撃のない歴史に対して)')
    cases = (('変動、資本の移動なし', 'float', False, dict(C), None, None),
             ('変動、資本の移動', 'float', True, dict(C, **INTL), None, None),
             ('共通の通貨', 'common', False, dict(C), None, None),
             ('共通の通貨、資本の移動', 'common', True, dict(C, **INTL), None, None),
             ('共通の通貨、財政同盟 ψ=0.3', 'common', False, dict(C), None, dict(members=('A', 'B'), psi=0.3)),
             ('共通の通貨、財政同盟 ψ=0.6', 'common', False, dict(C), None, dict(members=('A', 'B'), psi=0.6)))
    out = pairs(cases, ('crisis', 'none'))
    print('  場合                     | A の雇用 | A の実質賃金 | B の雇用 | B の実質賃金 | A の貿易の収支/所得 | A の国債/所得の差 | B の国債/所得の差 | A への同盟の移転/所得 | A の銀行の間の貸し/所得の差')
    for lab, ps in out.items():
        uni = lambda a, z: (a['A_union_in'][E10] / a['A_Y'][E10]).mean()
        ib = lambda a, z: (a['A_interbank'][E10] / a['A_Y'][E10]).mean() - (z['A_interbank'][E10] / z['A_Y'][E10]).mean()
        print(f"  {lab:24s} | {fmt(ps, rel('E', sl=E10))} | {fmt(ps, rel('real_wage', sl=E10))} | {fmt(ps, rel('E', 'B', E10))} | "
              f"{fmt(ps, rel('real_wage', 'B', E10))} | {fmt(ps, dif('tb_Y', sl=E10), '+.4f')} | {fmt(ps, dif('b', sl=E10))} | {fmt(ps, dif('b', 'B', E10))} | "
              f"{fmt(ps, uni, '+.4f')} | {fmt(ps, ib)}")


def k3_wealth_tax():
    print('K3 A が恒常の富裕税(一人あたりの正味資産が平均の 3 倍を超えた分に毎年 2%)を 200 年目から入れる。差は 200〜260 年の平均')
    print('   A の正味資産は外の資産を含む。「B も税」は A と B の両方が同じ富裕税を入れる(情報の交換はない)')
    print('   見える割合 enf_F = 税の当局が見える外の資産の割合(1 = 情報の交換が完全)。逃げ φ_L = 30: 逃れられる税の率 1% につき、外に置きたい割合 +0.3')
    print('   税収は、資本の移動のない閉じた場合に対する比。外の資産の差は、A の家計が持つ外の資産/所得の差(資本の逃げ)。')
    print('   銀行の間の貸し: 共通の通貨で、A の銀行の、ほかの地域の銀行に対する貸し(負なら借り。TARGET2 の残高にあたる)/所得の差')
    flee = lambda enf, phiL=30.0: dict(C, **INTL, enf_F=enf, phi_L=phiL)
    cases = []
    for cur, cl in (('float', '変動'), ('common', '共通')):
        cases += [(f'{cl}、資本の移動なし', cur, False, dict(C), None, None),
                  (f'{cl}、見える 1.0', cur, True, flee(1.0), None, None),
                  (f'{cl}、見える 0.5', cur, True, flee(0.5), None, None),
                  (f'{cl}、見える 0', cur, True, flee(0.0), None, None),
                  (f'{cl}、見える 0、B も税', cur, True, flee(0.0), None, None, 'wtax2')]
    out = pairs(cases, ('wtax', 'none'))
    base = {}
    for lab, ps in out.items():
        if '資本の移動なし' in lab:
            base[lab[:2]] = np.mean([np.mean(a['A_levy'][W] / a['A_Y'][W]) for a, z in ps])
    print('  場合                          | 税収/所得 | 閉じた場合に対する比 | A の外の資産/所得の差 | 為替の変化 | A の物価 | A の実質賃金 | A の雇用 | A の富の上位10%の差 | B の富の上位10%の差 | A の国債/所得の差 | 銀行の間の貸し/所得の差')
    for lab, ps in out.items():
        rev = np.mean([np.mean(a['A_levy'][W] / a['A_Y'][W]) for a, z in ps])
        fy = fmt(ps, dif('F_Y')) if 'A_F_Y' in ps[0][0] else '  —  '
        ib = lambda a, z: (a['A_interbank'][W] / a['A_Y'][W]).mean() - (z['A_interbank'][W] / z['A_Y'][W]).mean()
        print(f"  {lab:28s} | {rev:.4f} | {rev / base[lab[:2]]:.2f} | {fy} | {fmt(ps, dlne(), '+.4f')} | {fmt(ps, dif('infl'), '+.4f')} | "
              f"{fmt(ps, rel('real_wage'))} | {fmt(ps, rel('E'))} | {fmt(ps, dif('top10_nw'))} | {fmt(ps, dif('top10_nw', 'B'))} | {fmt(ps, dif('b'))} | {fmt(ps, ib)}")


def k4_union():
    print('K4 財政同盟の長い目の姿: A の生産性が恒常に低い(必需品の一製品あたり労働 0.1 → 0.13)。共通の通貨。100〜260 年の平均')
    print('   同盟なしと、同盟あり(ψ = 0.3、0.6)。資本の移動のありなし')
    rk = {'A': dict(gamma0=np.array([0.13, 0.3, 0.3]))}
    cases = [('同盟なし', False, None), ('ψ=0.3', False, 0.3), ('ψ=0.6', False, 0.6), ('同盟なし、資本の移動', True, None), ('ψ=0.3、資本の移動', True, 0.3)]
    jobs = [(s, 'common', cap, dict(C, **INTL) if cap else dict(C), rk, dict(members=('A', 'B'), psi=psi) if psi else None, 'none')
            for lab, cap, psi in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(history, jobs))
    L = slice(100, T1)
    print('  場合                  | A の雇用 | B の雇用 | A の実質賃金 | B の実質賃金 | A への移転/A の所得 | A の国債/所得 | B の国債/所得 | A の銀行の間の貸し/所得 | A の税率 | B の税率')
    for i, (lab, cap, psi) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        m = lambda k, nm='A': np.mean([r[f'{nm}_{k}'][L].mean() for r in rs])
        mr = lambda k, nm='A': np.mean([(r[f'{nm}_{k}'][L] / r[f'{nm}_Y'][L]).mean() for r in rs])
        print(f"  {lab:20s} | {m('E'):.0f} | {m('E', 'B'):.0f} | {m('real_wage'):.3f} | {m('real_wage', 'B'):.3f} | {mr('union_in'):+.4f} | {m('b'):.2f} | {m('b', 'B'):.2f} | "
              f"{mr('interbank'):+.3f} | {m('tau'):.3f} | {m('tau', 'B'):.3f}")


if __name__ == '__main__':
    k0_speed()
    k1_hike()
    k2_crisis()
    k3_wealth_tax()
    k4_union()
