# test_im.py: 段 0 の関門のテスト。python test_im.py で走る。
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'work_M'))
from im_core import Params, Ledger, Model, Component
import im_gj
import gj   # ノート M 第 2 章の模型(比べる相手)


def test_ledger_random(seed=0, n=2000):
    """でたらめな取引を大量に書き込んでも、帳簿の恒等式が保たれる。"""
    rng = np.random.default_rng(seed)
    S = ['a', 'b', 'c', 'bank', 'gov']
    led = Ledger(S, ['deposits', 'loans', 'bonds', 'equity'], real=['capital'])
    for t in range(5):
        for _ in range(n):
            k = rng.integers(4)
            x, y = rng.choice(S, 2, replace=False)
            amt = rng.random()
            if k == 0:
                led.pay(f'flow{rng.integers(6)}', x, y, amt)
            elif k == 1:
                led.trade(rng.choice(['loans', 'bonds', 'equity']), x, y, amt)
            elif k == 2:
                led.produce_real('capital', x, amt)
            else:
                led.revalue('equity', x, amt); led.revalue('equity', y, -amt)
        e = led.check()
        led.close()
    return e['max']


def gj_pairs():
    base = dict(a=1.5)
    cases = [dict(), dict(tau=0.0), dict(tau=0.6), dict(G=0.0), dict(G=0.08), dict(ic=0.1), dict(iB=0.04),
             dict(psi=0.2, n=4.0), dict(a=5.0), dict(bk=0.3), dict(gw=0.45)]
    for c in cases:
        yield c


def test_gj_equivalence():
    """労働者の貯蓄 0、投資 = ι × 資本家の貯蓄のとき、骨組みの上の gj は、ノート M の gj.period と同じ数を出す。"""
    worst = 0.0
    keys = ['pN', 'pL', 'EN', 'EL', 'LN', 'LL', 'I', 'L', 'rent', 'profit', 'Mw', 'Mk', 'T']
    for c in gj_pairs():
        p = im_gj.default_params().set(**c)
        gkw = {k: v for k, v in c.items() if k not in ('a', 'n')}
        ref = gj.period(gj.GJ(**gkw), a=c.get('a', 1.5), n=c.get('n', 1.0), B=1.0)
        r = im_gj.solve_year(p, 0.0, 1.0)
        worst = max(worst, max(abs(r[k] - ref[k]) for k in keys))
    # 短期の生産能力の上限があるときも
    ref0 = gj.period(gj.GJ(), 1.5)
    for tau in [0.4, 0.6]:
        ref = gj.period(gj.GJ(tau=tau), 1.5, Lcap=ref0['LN'])
        r = im_gj.solve_year(im_gj.default_params().set(tau=tau), 0.0, 1.0, Lcap=ref0['LN'])
        worst = max(worst, max(abs(r[k] - ref[k]) for k in keys))
    return worst


def test_gj_books(T=30, **kw):
    """骨組みの上で gj を T 年動かす。毎年、帳簿の恒等式が保たれ、各部門の預金が動かない(釣り合いが解けている)。
    国債の残高は、赤字の和だけ増える。"""
    m = im_gj.make_model(im_gj.default_params().set(**kw), Bw=0.0, Bk=1.0)
    h = m.run(T)
    books = max(r['books'] for r in h)
    drift = max(r['dep_drift'] for r in h)
    B_end = m.L.stock[('wk', 'bonds')] + m.L.stock[('kp', 'bonds')]
    bond_gap = abs(B_end - (1.0 + sum(r['deficit'] for r in h)))
    return books, drift, bond_gap


def test_layers():
    """時間の層: slow の部品は every 年ごとに動く。部品を切れば、何もしない(前の段と同じ数)。帳簿の時計は一つ。"""
    p = Params().add('x', 1.0, 'structure')
    led = Ledger(['a', 'b'], ['deposits'])
    calls = []
    comps = [Component('yearly', lambda m: m.L.pay('f', 'a', 'b', 1.0) or {'y': 1}, 'base'),
             Component('slow', lambda m: calls.append(m.L.t), 'slow', every=5),
             Component('off', lambda m: m.L.pay('g', 'a', 'b', 99.0), 'base', on=False)]
    m = Model(led, p, comps)
    m.run(12)
    assert calls == [0, 5, 10], calls
    assert abs(led.stock[('b', 'deposits')] - 12.0) < 1e-12
    return calls


if __name__ == '__main__':
    print('帳簿(でたらめな取引 1 万件):', f'{test_ledger_random():.1e}')
    print('gj と同じ数(13 通り、最大のずれ):', f'{test_gj_equivalence():.1e}')
    b, d, g = test_gj_books()
    print(f'gj を 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債の残高と赤字の和の差 {g:.1e}')
    b, d, g = test_gj_books(s_w=0.1)
    print(f'労働者の貯蓄 0.1 で 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債 {g:.1e}')
    b, d, g = test_gj_books(inv='bm', i0=0.02, i_u=0.03, i_h=0.1)
    print(f'投資の関数 bm で 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債 {g:.1e}')
    print('時間の層(slow, every=5 が動いた年):', test_layers())


# ---------------------------------------------------------------- 段 1
def test_stage1_me_c(T=400):
    """1a: 信用を切り、家計を二つの型にすれば、me_c v0.1 と同じ数(相対のずれ)。"""
    sys.path.insert(0, os.path.join(HERE, '..', 'work_mc'))
    import me_c, im1
    worst = books = 0.0
    keys = ['W', 'PN', 'PL', 'PI', 'E', 'KN', 'KL', 'KI', 'tau', 'M_w', 'M_c', 'G', 'deficit']
    for kw in ({}, dict(g_gamma=0.0), dict(b_N=0.6), dict(c_m=0.05)):
        a, _ = me_c.simulate(T=T, **kw)
        b, _ = im1.simulate(T=T, fiscal='deficit', **kw)          # me_c と同じ財政のルールで比べる
        worst = max(worst, max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys))
        books = max(books, b['books'].max(), b['sample_gap'].max())
    return worst, books


def test_stage1_weights(T=100):
    """1b: 連続の標本で、どの世帯も 3 つに分けて重みを 1/3 にしても、同じ数(重みの付け方によらない)。"""
    import im1
    a, _ = im1.simulate(T=T, households='continuous', Ns=400)
    b, _ = im1.simulate(T=T, households='continuous', Ns=400, split=3)
    keys = ['W', 'PN', 'E', 'tau', 'KN', 'gini_inc', 'top10_inc']
    return max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys), max(a['books'].max(), a['sample_gap'].max())


if __name__ == '__main__':
    w, b = test_stage1_me_c()
    print(f'段 1a: me_c v0.1 と同じ数(4 通り × 400 年、相対のずれの最大) {w:.1e}、帳簿と標本の差 {b:.1e}')
    w, b = test_stage1_weights()
    print(f'段 1b: 重みの付け方によらない(分けた標本との相対のずれ) {w:.1e}、帳簿と標本の差 {b:.1e}')


def test_stage1_credit_off_equiv(T=200):
    """1c: 信用の部品を入れても、金利 0、借入 0(内部留保 100%、担保の上限 0)、r_eq = r_bar なら、信用なしと同じ数。"""
    import im1
    keys = ['W', 'PN', 'E', 'tau', 'KN', 'KL', 'KI', 'gini_inc', 'top10_inc', 'G']
    a, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0)
    b, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True,
                        r_cb=0.0, spread=0.0, r_eq=0.06, f_R=1.0, lam=0.0)
    return max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys), max(b['books'].max(), b['sample_gap'].max())


if __name__ == '__main__':
    w, b = test_stage1_credit_off_equiv()
    print(f'段 1c: 信用の部品を入れても、金利 0・借入 0 なら信用なしと同じ数(相対のずれ) {w:.1e}、帳簿と標本の差 {b:.1e}')


# ---------------------------------------------------------------- 段 2
def test_stage2(T=150):
    """段 2: 土地と知財を入れても、帳簿が閉じ、標本の預金の和がマクロと一致し、土地の和が総量に等しい。
    世代の入れ替わり(乱数)を切れば、どの世帯も 3 つに分けて重みを 1/3 にしても同じ数(土地の市場も重みによらない)。
    土地を切れば(land = False)、段 1 と同じ数(部品はすべて land で守ってある)。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    books = max(a['books'].max(), a['sample_gap'].max(), a['land_sum_err'].max())
    b, _ = im1.simulate(T=T, **dict(kw, turnover=0.0))
    c, _ = im1.simulate(T=T, split=3, **dict(kw, turnover=0.0))
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw']
    w = max(np.max(np.abs(b[k] - c[k]) / (np.abs(b[k]) + 1)) for k in keys)
    d, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True)
    e, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True, land=False, c_h=0.04, rho_T=0.3)
    off = max(np.max(np.abs(d[k] - e[k]) / (np.abs(d[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'top10_inc'])
    return books, w, off


if __name__ == '__main__':
    b, w, off = test_stage2()
    print(f'段 2: 帳簿・標本の差・土地の和のずれ {b:.1e}、重みの付け方によらない {w:.1e}、土地を切れば段 1 と同じ {off:.1e}')


# ---------------------------------------------------------------- 段 3
def test_stage3(T=120):
    """段 3: (1) 貿易を切れば(自地域の財だけを買う)、二つの地域は、それぞれ一国の模型と同じ数。
    (2) 貿易があっても、帳簿(銀行ごとの預金の負債と、銀行の間の貸し借りを含む)が閉じ、標本の預金の和が地域ごとに一致する。
    (3) 同じ二つの地域なら、二つは同じ数になる(対称)。"""
    import im1, im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    o, _ = im3.simulate(T=T, common=C, home=1.0)
    a, _ = im1.simulate(T=T, households='continuous', Ns=400, **C)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw']
    nt = max(np.max(np.abs(o[nm][k] - a[k]) / (np.abs(a[k]) + 1)) for nm in 'AB' for k in keys)
    o, w = im3.simulate(T=T, common=C, home=0.5, tc=0.1)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for nm in 'AB')
    gap = max(o[nm]['sample_gap'].max() for nm in 'AB')
    sym = max(np.max(np.abs(o['A'][k] - o['B'][k]) / (np.abs(o['A'][k]) + 1)) for k in keys)
    return nt, books, gap, sym


if __name__ == '__main__':
    nt, books, gap, sym = test_stage3()
    print(f'段 3: 貿易を切れば一国と同じ {nt:.1e}、貿易ありの帳簿のずれ/所得 {books:.1e}、標本の差 {gap:.1e}、同じ二つの地域の対称 {sym:.1e}')


def test_stage3b(T=120):
    """段 3b: (1) 別々の通貨で為替を 1 に固定すれば、共通の通貨(段 3a)と同じ数。(2) 同じ二つの地域で為替を変動させると、
    為替は 1 のままで、共通の通貨と同じ数。(3) 通貨ごとの帳簿が閉じる。"""
    import im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw', 'tb_Y']
    oc, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1)
    of, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1, currency='fixed')
    ofl, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1, currency='float')
    d = lambda o: max(np.max(np.abs(o[nm][k] - oc[nm][k]) / (np.abs(oc[nm][k]) + 1)) for nm in 'AB' for k in keys)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for o in (of, ofl) for nm in 'AB')
    return d(of), d(ofl), float(np.max(np.abs(ofl['A']['e'] - 1))), books


if __name__ == '__main__':
    a, b, e, bk = test_stage3b()
    print(f'段 3b: 固定(e = 1)と共通の通貨の差 {a:.1e}、変動(同じ二地域)と共通の差 {b:.1e}、為替の 1 からのずれ {e:.1e}、通貨ごとの帳簿のずれ/所得 {bk:.1e}')


def test_stage3c(T=120):
    """段 3c: 同じ三つの地域なら、すべて変動でも、A+B の同盟と C の変動でも、為替は 1 のままで、すべて共通の通貨と同じ数。
    通貨ごとの帳簿が閉じる。"""
    import im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    R3 = ('A', 'B', 'C')
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw', 'tb_Y']
    oc, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1)
    of, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1, currency='float')
    ou, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1, currency={'A': 'x', 'B': 'x', 'C': 'y'}, regimes={'y': 'float'})
    d = lambda o: max(np.max(np.abs(o[nm][k] - oc[nm][k]) / (np.abs(oc[nm][k]) + 1)) for nm in R3 for k in keys)
    e = max(np.max(np.abs(o[nm]['e_cur'] - 1)) for o in (of, ou) for nm in R3)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for o in (oc, of, ou) for nm in R3)
    return d(of), d(ou), e, books


if __name__ == '__main__':
    a, b, e, bk = test_stage3c()
    print(f'段 3c: 同じ三地域で、すべて変動と共通の差 {a:.1e}、A+B 同盟・C 変動と共通の差 {b:.1e}、為替の 1 からのずれ {e:.1e}、帳簿のずれ/所得 {bk:.1e}')


def test_stage2b(T=150):
    """段 2b: 知財の市場を入れても、帳簿・標本の預金の和が閉じ、知財の和が売り買いで変わらない。
    知財の市場を切れば(ip_market = False)、段 2 と同じ数。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, ip_market=True, **kw)
    books = max(a['books'].max(), a['sample_gap'].max(), a['land_sum_err'].max())
    b, _ = im1.simulate(T=T, **kw)
    c, _ = im1.simulate(T=T, ip_market=False, theta0_IP=0.3, xcap_IP=2.0, **kw)
    off = max(np.max(np.abs(b[k] - c[k]) / (np.abs(b[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    return books, off


if __name__ == '__main__':
    b, off = test_stage2b()
    print(f'段 2b: 知財の市場ありの帳簿・標本の差 {b:.1e}、知財の市場を切れば段 2 と同じ {off:.1e}')


# ---------------------------------------------------------------- 段 4
def test_stage4(T=150):
    """段 4: 災害を切れば(disaster = False)、段 2 と同じ数。災害の部品を入れても、災害が起きず守りも救済もなければ同じ数。
    災害が起きても(国の救済、守りあり)、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, disaster=True, **kw)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'hh_debt']
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys)
    e = im1.Im1(disaster=True, dis_rate=0.1, protect=0.01, relief='state', **kw)
    e.shock_queue = [(50, 0.4, 'common'), (80, 0.3, 'idio')]
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] for r in rec), max(r['sample_gap'] for r in rec))
    e = im1.Im1(disaster=True, relief='mutual', **kw)
    e.shock_queue = [(50, 0.3, 'idio')]
    rec = [e.step() for _ in range(T)]
    books = max(books, max(r['books'] for r in rec), max(r['sample_gap'] for r in rec))
    return same, books


if __name__ == '__main__':
    same, books = test_stage4()
    print(f'段 4: 災害の部品を入れても、起きなければ段 2 と同じ {same:.1e}、災害が起きても帳簿と標本の差 {books:.1e}')


def test_stage4_is(T=200):
    """段 4 の重点標本化(分割と併合): 使っても、帳簿と標本の預金の和が閉じ、土地の和と人数の和が変わらない。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    e = im1.Im1(is_split=5, **kw)
    heads0 = e.H.heads.sum()
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] for r in rec), max(r['sample_gap'] for r in rec), max(r['land_sum_err'] for r in rec))
    return books, abs(e.H.heads.sum() - heads0), getattr(e, 'n_splits', 0)


if __name__ == '__main__':
    b, h, n = test_stage4_is()
    print(f'段 4(重点標本化): 帳簿・標本の差・土地の和のずれ {b:.1e}、人数の和のずれ {h:.1e}(分割と併合 {n} 回)')


def test_stage4b(T=200):
    """段 4b: 上乗せ金利、利払いの重さのルール、行き詰まりの道(富裕税・緊縮・踏み倒し・金融抑圧)を入れても、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.1,
              relief='state', mu_B=0.3, theta_B=0.6, fiscal='burden', burden_target=0.016, phi_pi=0.5, rbar_link=0.3, theta_resolve=0.55)
    worst = 0.0
    for res in ('none', 'levy', 'austerity', 'default', 'repression'):
        e = im1.Im1(resolve=res, **kw)
        rec = [e.step() for _ in range(T)]
        worst = max(worst, max(r['books'] / r['Y'] for r in rec), max(r['sample_gap'] / r['Y'] for r in rec))
    return worst


if __name__ == '__main__':
    print(f'段 4b: 行き詰まりの道を五つ入れても、帳簿と標本の差/所得 {test_stage4b():.1e}')


# ---------------------------------------------------------------- 段 5
def test_stage5(T=200):
    """段 5: 学習と投票を入れても、帳簿と標本の差が閉じる。学ばず投票しなければ(learn = vote = False)、段 4 と同じ数。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.05, relief='state')
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, learn=False, vote=False, k_learn=9.0, rho_w=0.5, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    e = im1.Im1(learn=True, learn_omega=0.95, vote=True, rho_w=1e-2, **kw)
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] / r['Y'] for r in rec), max(r['sample_gap'] / r['Y'] for r in rec))
    return same, books
# test_im.py: 段 0 の関門のテスト。python test_im.py で走る。
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'work_M'))
from im_core import Params, Ledger, Model, Component
import im_gj
import gj   # ノート M 第 2 章の模型(比べる相手)


def test_ledger_random(seed=0, n=2000):
    """でたらめな取引を大量に書き込んでも、帳簿の恒等式が保たれる。"""
    rng = np.random.default_rng(seed)
    S = ['a', 'b', 'c', 'bank', 'gov']
    led = Ledger(S, ['deposits', 'loans', 'bonds', 'equity'], real=['capital'])
    for t in range(5):
        for _ in range(n):
            k = rng.integers(4)
            x, y = rng.choice(S, 2, replace=False)
            amt = rng.random()
            if k == 0:
                led.pay(f'flow{rng.integers(6)}', x, y, amt)
            elif k == 1:
                led.trade(rng.choice(['loans', 'bonds', 'equity']), x, y, amt)
            elif k == 2:
                led.produce_real('capital', x, amt)
            else:
                led.revalue('equity', x, amt); led.revalue('equity', y, -amt)
        e = led.check()
        led.close()
    return e['max']


def gj_pairs():
    base = dict(a=1.5)
    cases = [dict(), dict(tau=0.0), dict(tau=0.6), dict(G=0.0), dict(G=0.08), dict(ic=0.1), dict(iB=0.04),
             dict(psi=0.2, n=4.0), dict(a=5.0), dict(bk=0.3), dict(gw=0.45)]
    for c in cases:
        yield c


def test_gj_equivalence():
    """労働者の貯蓄 0、投資 = ι × 資本家の貯蓄のとき、骨組みの上の gj は、ノート M の gj.period と同じ数を出す。"""
    worst = 0.0
    keys = ['pN', 'pL', 'EN', 'EL', 'LN', 'LL', 'I', 'L', 'rent', 'profit', 'Mw', 'Mk', 'T']
    for c in gj_pairs():
        p = im_gj.default_params().set(**c)
        gkw = {k: v for k, v in c.items() if k not in ('a', 'n')}
        ref = gj.period(gj.GJ(**gkw), a=c.get('a', 1.5), n=c.get('n', 1.0), B=1.0)
        r = im_gj.solve_year(p, 0.0, 1.0)
        worst = max(worst, max(abs(r[k] - ref[k]) for k in keys))
    # 短期の生産能力の上限があるときも
    ref0 = gj.period(gj.GJ(), 1.5)
    for tau in [0.4, 0.6]:
        ref = gj.period(gj.GJ(tau=tau), 1.5, Lcap=ref0['LN'])
        r = im_gj.solve_year(im_gj.default_params().set(tau=tau), 0.0, 1.0, Lcap=ref0['LN'])
        worst = max(worst, max(abs(r[k] - ref[k]) for k in keys))
    return worst


def test_gj_books(T=30, **kw):
    """骨組みの上で gj を T 年動かす。毎年、帳簿の恒等式が保たれ、各部門の預金が動かない(釣り合いが解けている)。
    国債の残高は、赤字の和だけ増える。"""
    m = im_gj.make_model(im_gj.default_params().set(**kw), Bw=0.0, Bk=1.0)
    h = m.run(T)
    books = max(r['books'] for r in h)
    drift = max(r['dep_drift'] for r in h)
    B_end = m.L.stock[('wk', 'bonds')] + m.L.stock[('kp', 'bonds')]
    bond_gap = abs(B_end - (1.0 + sum(r['deficit'] for r in h)))
    return books, drift, bond_gap


def test_layers():
    """時間の層: slow の部品は every 年ごとに動く。部品を切れば、何もしない(前の段と同じ数)。帳簿の時計は一つ。"""
    p = Params().add('x', 1.0, 'structure')
    led = Ledger(['a', 'b'], ['deposits'])
    calls = []
    comps = [Component('yearly', lambda m: m.L.pay('f', 'a', 'b', 1.0) or {'y': 1}, 'base'),
             Component('slow', lambda m: calls.append(m.L.t), 'slow', every=5),
             Component('off', lambda m: m.L.pay('g', 'a', 'b', 99.0), 'base', on=False)]
    m = Model(led, p, comps)
    m.run(12)
    assert calls == [0, 5, 10], calls
    assert abs(led.stock[('b', 'deposits')] - 12.0) < 1e-12
    return calls


if __name__ == '__main__':
    print('帳簿(でたらめな取引 1 万件):', f'{test_ledger_random():.1e}')
    print('gj と同じ数(13 通り、最大のずれ):', f'{test_gj_equivalence():.1e}')
    b, d, g = test_gj_books()
    print(f'gj を 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債の残高と赤字の和の差 {g:.1e}')
    b, d, g = test_gj_books(s_w=0.1)
    print(f'労働者の貯蓄 0.1 で 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債 {g:.1e}')
    b, d, g = test_gj_books(inv='bm', i0=0.02, i_u=0.03, i_h=0.1)
    print(f'投資の関数 bm で 30 年: 帳簿 {b:.1e}、預金の動き {d:.1e}、国債 {g:.1e}')
    print('時間の層(slow, every=5 が動いた年):', test_layers())


# ---------------------------------------------------------------- 段 1
def test_stage1_me_c(T=400):
    """1a: 信用を切り、家計を二つの型にすれば、me_c v0.1 と同じ数(相対のずれ)。"""
    sys.path.insert(0, os.path.join(HERE, '..', 'work_mc'))
    import me_c, im1
    worst = books = 0.0
    keys = ['W', 'PN', 'PL', 'PI', 'E', 'KN', 'KL', 'KI', 'tau', 'M_w', 'M_c', 'G', 'deficit']
    for kw in ({}, dict(g_gamma=0.0), dict(b_N=0.6), dict(c_m=0.05)):
        a, _ = me_c.simulate(T=T, **kw)
        b, _ = im1.simulate(T=T, fiscal='deficit', **kw)          # me_c と同じ財政のルールで比べる
        worst = max(worst, max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys))
        books = max(books, b['books'].max(), b['sample_gap'].max())
    return worst, books


def test_stage1_weights(T=100):
    """1b: 連続の標本で、どの世帯も 3 つに分けて重みを 1/3 にしても、同じ数(重みの付け方によらない)。"""
    import im1
    a, _ = im1.simulate(T=T, households='continuous', Ns=400)
    b, _ = im1.simulate(T=T, households='continuous', Ns=400, split=3)
    keys = ['W', 'PN', 'E', 'tau', 'KN', 'gini_inc', 'top10_inc']
    return max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys), max(a['books'].max(), a['sample_gap'].max())


if __name__ == '__main__':
    w, b = test_stage1_me_c()
    print(f'段 1a: me_c v0.1 と同じ数(4 通り × 400 年、相対のずれの最大) {w:.1e}、帳簿と標本の差 {b:.1e}')
    w, b = test_stage1_weights()
    print(f'段 1b: 重みの付け方によらない(分けた標本との相対のずれ) {w:.1e}、帳簿と標本の差 {b:.1e}')


def test_stage1_credit_off_equiv(T=200):
    """1c: 信用の部品を入れても、金利 0、借入 0(内部留保 100%、担保の上限 0)、r_eq = r_bar なら、信用なしと同じ数。"""
    import im1
    keys = ['W', 'PN', 'E', 'tau', 'KN', 'KL', 'KI', 'gini_inc', 'top10_inc', 'G']
    a, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0)
    b, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True,
                        r_cb=0.0, spread=0.0, r_eq=0.06, f_R=1.0, lam=0.0)
    return max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys), max(b['books'].max(), b['sample_gap'].max())


if __name__ == '__main__':
    w, b = test_stage1_credit_off_equiv()
    print(f'段 1c: 信用の部品を入れても、金利 0・借入 0 なら信用なしと同じ数(相対のずれ) {w:.1e}、帳簿と標本の差 {b:.1e}')


# ---------------------------------------------------------------- 段 2
def test_stage2(T=150):
    """段 2: 土地と知財を入れても、帳簿が閉じ、標本の預金の和がマクロと一致し、土地の和が総量に等しい。
    世代の入れ替わり(乱数)を切れば、どの世帯も 3 つに分けて重みを 1/3 にしても同じ数(土地の市場も重みによらない)。
    土地を切れば(land = False)、段 1 と同じ数(部品はすべて land で守ってある)。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    books = max(a['books'].max(), a['sample_gap'].max(), a['land_sum_err'].max())
    b, _ = im1.simulate(T=T, **dict(kw, turnover=0.0))
    c, _ = im1.simulate(T=T, split=3, **dict(kw, turnover=0.0))
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw']
    w = max(np.max(np.abs(b[k] - c[k]) / (np.abs(b[k]) + 1)) for k in keys)
    d, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True)
    e, _ = im1.simulate(T=T, households='continuous', Ns=400, g_gamma=0.0, credit=True, land=False, c_h=0.04, rho_T=0.3)
    off = max(np.max(np.abs(d[k] - e[k]) / (np.abs(d[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'top10_inc'])
    return books, w, off


if __name__ == '__main__':
    b, w, off = test_stage2()
    print(f'段 2: 帳簿・標本の差・土地の和のずれ {b:.1e}、重みの付け方によらない {w:.1e}、土地を切れば段 1 と同じ {off:.1e}')


# ---------------------------------------------------------------- 段 3
def test_stage3(T=120):
    """段 3: (1) 貿易を切れば(自地域の財だけを買う)、二つの地域は、それぞれ一国の模型と同じ数。
    (2) 貿易があっても、帳簿(銀行ごとの預金の負債と、銀行の間の貸し借りを含む)が閉じ、標本の預金の和が地域ごとに一致する。
    (3) 同じ二つの地域なら、二つは同じ数になる(対称)。"""
    import im1, im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    o, _ = im3.simulate(T=T, common=C, home=1.0)
    a, _ = im1.simulate(T=T, households='continuous', Ns=400, **C)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw']
    nt = max(np.max(np.abs(o[nm][k] - a[k]) / (np.abs(a[k]) + 1)) for nm in 'AB' for k in keys)
    o, w = im3.simulate(T=T, common=C, home=0.5, tc=0.1)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for nm in 'AB')
    gap = max(o[nm]['sample_gap'].max() for nm in 'AB')
    sym = max(np.max(np.abs(o['A'][k] - o['B'][k]) / (np.abs(o['A'][k]) + 1)) for k in keys)
    return nt, books, gap, sym


if __name__ == '__main__':
    nt, books, gap, sym = test_stage3()
    print(f'段 3: 貿易を切れば一国と同じ {nt:.1e}、貿易ありの帳簿のずれ/所得 {books:.1e}、標本の差 {gap:.1e}、同じ二つの地域の対称 {sym:.1e}')


def test_stage3b(T=120):
    """段 3b: (1) 別々の通貨で為替を 1 に固定すれば、共通の通貨(段 3a)と同じ数。(2) 同じ二つの地域で為替を変動させると、
    為替は 1 のままで、共通の通貨と同じ数。(3) 通貨ごとの帳簿が閉じる。"""
    import im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw', 'tb_Y']
    oc, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1)
    of, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1, currency='fixed')
    ofl, _ = im3.simulate(T=T, common=C, home=0.5, tc=0.1, currency='float')
    d = lambda o: max(np.max(np.abs(o[nm][k] - oc[nm][k]) / (np.abs(oc[nm][k]) + 1)) for nm in 'AB' for k in keys)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for o in (of, ofl) for nm in 'AB')
    return d(of), d(ofl), float(np.max(np.abs(ofl['A']['e'] - 1))), books


if __name__ == '__main__':
    a, b, e, bk = test_stage3b()
    print(f'段 3b: 固定(e = 1)と共通の通貨の差 {a:.1e}、変動(同じ二地域)と共通の差 {b:.1e}、為替の 1 からのずれ {e:.1e}、通貨ごとの帳簿のずれ/所得 {bk:.1e}')


def test_stage3c(T=120):
    """段 3c: 同じ三つの地域なら、すべて変動でも、A+B の同盟と C の変動でも、為替は 1 のままで、すべて共通の通貨と同じ数。
    通貨ごとの帳簿が閉じる。"""
    import im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    R3 = ('A', 'B', 'C')
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'hh_debt', 'top10_inc', 'gini_nw', 'tb_Y']
    oc, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1)
    of, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1, currency='float')
    ou, _ = im3.simulate(T=T, regions=R3, common=C, home=0.5, tc=0.1, currency={'A': 'x', 'B': 'x', 'C': 'y'}, regimes={'y': 'float'})
    d = lambda o: max(np.max(np.abs(o[nm][k] - oc[nm][k]) / (np.abs(oc[nm][k]) + 1)) for nm in R3 for k in keys)
    e = max(np.max(np.abs(o[nm]['e_cur'] - 1)) for o in (of, ou) for nm in R3)
    books = max(np.max(o[nm]['books'] / o[nm]['Y']) for o in (oc, of, ou) for nm in R3)
    return d(of), d(ou), e, books


if __name__ == '__main__':
    a, b, e, bk = test_stage3c()
    print(f'段 3c: 同じ三地域で、すべて変動と共通の差 {a:.1e}、A+B 同盟・C 変動と共通の差 {b:.1e}、為替の 1 からのずれ {e:.1e}、帳簿のずれ/所得 {bk:.1e}')


def test_stage2b(T=150):
    """段 2b: 知財の市場を入れても、帳簿・標本の預金の和が閉じ、知財の和が売り買いで変わらない。
    知財の市場を切れば(ip_market = False)、段 2 と同じ数。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, ip_market=True, **kw)
    books = max(a['books'].max(), a['sample_gap'].max(), a['land_sum_err'].max())
    b, _ = im1.simulate(T=T, **kw)
    c, _ = im1.simulate(T=T, ip_market=False, theta0_IP=0.3, xcap_IP=2.0, **kw)
    off = max(np.max(np.abs(b[k] - c[k]) / (np.abs(b[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    return books, off


if __name__ == '__main__':
    b, off = test_stage2b()
    print(f'段 2b: 知財の市場ありの帳簿・標本の差 {b:.1e}、知財の市場を切れば段 2 と同じ {off:.1e}')


# ---------------------------------------------------------------- 段 4
def test_stage4(T=150):
    """段 4: 災害を切れば(disaster = False)、段 2 と同じ数。災害の部品を入れても、災害が起きず守りも救済もなければ同じ数。
    災害が起きても(国の救済、守りあり)、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, disaster=True, **kw)
    keys = ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'hh_debt']
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in keys)
    e = im1.Im1(disaster=True, dis_rate=0.1, protect=0.01, relief='state', **kw)
    e.shock_queue = [(50, 0.4, 'common'), (80, 0.3, 'idio')]
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] for r in rec), max(r['sample_gap'] for r in rec))
    e = im1.Im1(disaster=True, relief='mutual', **kw)
    e.shock_queue = [(50, 0.3, 'idio')]
    rec = [e.step() for _ in range(T)]
    books = max(books, max(r['books'] for r in rec), max(r['sample_gap'] for r in rec))
    return same, books


if __name__ == '__main__':
    same, books = test_stage4()
    print(f'段 4: 災害の部品を入れても、起きなければ段 2 と同じ {same:.1e}、災害が起きても帳簿と標本の差 {books:.1e}')


def test_stage4_is(T=200):
    """段 4 の重点標本化(分割と併合): 使っても、帳簿と標本の預金の和が閉じ、土地の和と人数の和が変わらない。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    e = im1.Im1(is_split=5, **kw)
    heads0 = e.H.heads.sum()
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] for r in rec), max(r['sample_gap'] for r in rec), max(r['land_sum_err'] for r in rec))
    return books, abs(e.H.heads.sum() - heads0), getattr(e, 'n_splits', 0)


if __name__ == '__main__':
    b, h, n = test_stage4_is()
    print(f'段 4(重点標本化): 帳簿・標本の差・土地の和のずれ {b:.1e}、人数の和のずれ {h:.1e}(分割と併合 {n} 回)')


def test_stage4b(T=200):
    """段 4b: 上乗せ金利、利払いの重さのルール、行き詰まりの道(富裕税・緊縮・踏み倒し・金融抑圧)を入れても、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.1,
              relief='state', mu_B=0.3, theta_B=0.6, fiscal='burden', burden_target=0.016, phi_pi=0.5, rbar_link=0.3, theta_resolve=0.55)
    worst = 0.0
    for res in ('none', 'levy', 'austerity', 'default', 'repression'):
        e = im1.Im1(resolve=res, **kw)
        rec = [e.step() for _ in range(T)]
        worst = max(worst, max(r['books'] / r['Y'] for r in rec), max(r['sample_gap'] / r['Y'] for r in rec))
    return worst


if __name__ == '__main__':
    print(f'段 4b: 行き詰まりの道を五つ入れても、帳簿と標本の差/所得 {test_stage4b():.1e}')


# ---------------------------------------------------------------- 段 5
def test_stage5(T=200):
    """段 5: 学習と投票を入れても、帳簿と標本の差が閉じる。学ばず投票しなければ(learn = vote = False)、段 4 と同じ数。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.05, relief='state')
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, learn=False, vote=False, k_learn=9.0, rho_w=0.5, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    e = im1.Im1(learn=True, learn_omega=0.95, vote=True, rho_w=1e-2, **kw)
    rec = [e.step() for _ in range(T)]
    books = max(max(r['books'] / r['Y'] for r in rec), max(r['sample_gap'] / r['Y'] for r in rec))
    return same, books


if __name__ == '__main__':
    same, books = test_stage5()
    print(f'段 5: 学ばず投票しなければ段 4 と同じ {same:.1e}、学習と投票を入れた帳簿と標本の差/所得 {books:.1e}')


# ---------------------------------------------------------------- 段 5 の探索
def test_stage5b(T=200):
    """段 5 の探索: 研究(学ぶ)・株式の市場・知財の市場を入れても、帳簿と標本の差が閉じ、持分の和は 1。
    入れなければ(research = eq_market = False)、段 5 と同じ数。株式の値段の決め方は二つとも確かめる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, research=False, eq_market=False, s_R=0.5, kappa_E=3.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    books, eq_err = 0.0, 0.0
    for pricing in ('fundamental', 'demand'):
        e = im1.Im1(research=True, adaptive_R=True, eq_market=True, eq_pricing=pricing, kappa_E=0.0, ip_market=True, **kw)
        for _ in range(T):
            r = e.step()
            books = max(books, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
            eq_err = max(eq_err, abs(e.H.eq.sum() - 1))
    return same, books, eq_err


if __name__ == '__main__':
    same, books, eq_err = test_stage5b()
    print(f'段 5 の探索: 研究と株式の市場を入れなければ段 5 と同じ {same:.1e}、入れた帳簿と標本の差/所得 {books:.1e}、持分の和のずれ {eq_err:.1e}')


# ---------------------------------------------------------------- 段 6
def test_stage6(T=120):
    """段 6: 資本の移動(共通・固定・変動の通貨)と財政同盟と富裕税を入れても、通貨ごとの帳簿が閉じ、家計の標本の預金と外の資産が
    帳簿と合う。資本の移動を入れなければ、段 3 と同じ数(外に置きたい割合を与えても、capital = False なら効かない)。"""
    import im3
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im3.simulate(T=T, common=C, currency='float')
    b, _ = im3.simulate(T=T, common=dict(C, theta_F0=0.3, phi_K=5.0), currency='float', capital=False)
    same = max(np.max(np.abs(a[n][k] - b[n][k]) / (np.abs(a[n][k]) + 1)) for n in 'AB' for k in ('W', 'PN', 'E', 'e', 'tau'))
    worst = 0.0
    for cur in ('common', 'fixed', 'float'):
        kw = dict(common=dict(C, theta_F0=0.1, phi_K=1.0, enf_F=0.5, phi_L=30.0), currency=cur, capital=True,
                  seeds={'A': 0, 'B': 1}, events=((40, 'A', dict(wealth_tax=0.02)), (60, 'A', dict(r_cb=0.03))))
        if cur == 'common':
            kw['union'] = dict(members=('A', 'B'), psi=0.5)
        o, w = im3.simulate(T=T, **kw)
        for n in 'AB':
            worst = max(worst, np.max(o[n]['books'] / o[n]['Y']), np.max(o[n]['sample_gap'] / o[n]['Y']), np.max(o[n]['F_gap'] / o[n]['Y']))
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage6()
    print(f'段 6: 資本の移動を入れなければ段 3 と同じ {same:.1e}、資本の移動・財政同盟・富裕税を入れた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 5 の世界樹の案
def test_stage5c(T=150):
    """部門の中に企業を置いても(規則的・真似る・公の研究所・知財の保護・集中の値上げ)、帳簿と標本の差が閉じ、占有率の和は 1。
    企業を置かなければ(firms = 0)、段 5 の探索と同じ数。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, firms=0, ip_lag=7, mu_hhi=0.5, iota_L=0.9, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'gamma_front'])
    e = im1.Im1(firms=12, ip_lag=5, s_pub=0.01, mu_hhi=0.2, **kw)
    worst, s_err = 0.0, 0.0
    for _ in range(T):
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
        s_err = max(s_err, float(np.max(np.abs(e.fp.s.sum(1) - 1))))
    return same, worst, s_err


if __name__ == '__main__':
    same, worst, s_err = test_stage5c()
    print(f'段 5 の世界樹の案: 企業を置かなければ段 5 の探索と同じ {same:.1e}、企業を置いた帳簿と標本の差/所得 {worst:.1e}、占有率の和のずれ {s_err:.1e}')


# ---------------------------------------------------------------- 段 7
def test_stage7(T=150):
    """段 7: 危機時の生産の部品を切った値(在庫なし、上限 0、手当てなし、備蓄 0、公共事業 0、固定費 0)なら、段 6 までと同じ数。
    在庫・価格の上限と品不足・最後の生産者・損失の分担・備蓄・公共事業・給付・固定費と廃棄を入れても、帳簿(実物の在庫と備蓄の列を含む)と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, inventory=False, cap_N=0.0, crisis_rule='none', g_stock=0.0, g_pw=0.0, g_tr=0.0, f_K=0.0, delta_scrap=0.0,
                        inv_target=0.3, zeta_inv=0.9, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    for rule in ('none', 'last', 'share'):
        e = im1.Im1(inventory=True, g_stock=0.3, g_pw=0.01, g_tr=0.01, f_K=0.03, delta_scrap=0.1, crisis_rule=rule, **kw)
        for t in range(T):
            if t == 60:
                e.K[0] *= 0.6; e.gamma[2] *= 1.5
                e.p.set(cap_N=float(e.prices[0]))
            if t == 70:
                e.gamma[2] /= 1.5; e.p.set(cap_N=0.0)
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage7()
    print(f'段 7: 危機時の生産の部品を切れば前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 8
def test_stage8(T=150):
    """段 8: 文化の効果 ψ = 0 なら(ε と σ を変えても)前と同じ数。ψ > 0 でも帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, psi_v=0.0, eps_v=3.0, sigma_v=5.0, ip_ref=100.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    for extra in (dict(), dict(research=True, inventory=True)):
        e = im1.Im1(psi_v=1.0, eps_v=3.0, ip_market=True, **extra, **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage8()
    print(f'段 8: ψ = 0 なら前と同じ {same:.1e}、ψ > 0 の帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 9
def test_stage9(T=150):
    """段 9: 運 0・腕 0・養育の費用 0 なら前と同じ数。運・腕・裾の重い運・養育の費用・重点標本化を入れても、帳簿と標本の差が閉じ、持分の和は 1。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, luck_sigma=0.0, luck_skill=0.0, luck_nu=3.0, luck_rho=0.9, child_cost=0.0, child_years=5, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst, eq_err = 0.0, 0.0
    for extra in (dict(eq_init='equal', luck_sigma=0.3, luck_nu=3.0, luck_skill=0.03, luck_rho=0.9, child_cost=1.0, crisis_track=True),
                  dict(luck_sigma=0.2, child_cost=0.5, is_split=5)):
        e = im1.Im1(**extra, **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
            eq_err = max(eq_err, abs(e.H.eq.sum() - 1))
    return same, worst, eq_err


if __name__ == '__main__':
    same, worst, eq_err = test_stage9()
    print(f'段 9: 運と養育の費用を切れば前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}、持分の和のずれ {eq_err:.1e}')


# ---------------------------------------------------------------- 段 10
def test_stage10(T=150):
    """段 10: 引退なし・年金なし・児童手当 0・消費税なしなら前と同じ数。引退と三つの年金・児童手当・消費税で賄う BI・重点標本化を入れても、
    帳簿と標本の差(預金 = 引き出せる預金 + 積立の残高)が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, ret_age=0, pension='payg', child_allow=0.0, vat_bi=True, pen_rate=0.9, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    for pen in ('payg', 'points', 'funded'):
        e = im1.Im1(ret_age=45, pension=pen, child_cost=1.0, child_allow=0.5, g_tr=0.02, vat_bi=True, crisis_track=True,
                    is_split=(5 if pen == 'funded' else 0), **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage10()
    print(f'段 10: 年齢と年金と BI を切れば前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 11
def test_stage11(T=150):
    """段 11: 自動化の割合 0 なら(資本の係数の上がり方を変えても)前と同じ数。自動化を入れても(研究、投票、世界の模型の中でも)帳簿と標本の差が閉じる。"""
    import im1, im3
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, auto_share=0.0, auto_kappa=3.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    e = im1.Im1(auto_share=1.0, phi_u=0.1, phi_r=0.25, vote=True, **kw)
    for _ in range(T):
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    o, w = im3.simulate(T=80, common=dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, auto_share=0.5, phi_u=0.1, phi_r=0.25))
    worst = max(worst, max(np.max(o[n]['books'] / o[n]['Y']) for n in 'AB'))
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage11()
    print(f'段 11: 自動化を切れば前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 12
def test_stage12(T=150):
    """段 12: 決め直す割合 1 なら前と同じ数。割合を下げても(土地・株・知財・外の資産の市場と合わせて)帳簿と標本の差が閉じる。"""
    import im1, im3
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, attention=1.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    e = im1.Im1(attention=0.3, eq_market=True, ip_market=True, is_split=5, **kw)
    for _ in range(T):
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    o, w = im3.simulate(T=80, common=dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, attention=0.3, theta_F0=0.1, phi_K=1.0),
                        currency='float', capital=True, seeds={'A': 0, 'B': 1})
    worst = max(worst, max(np.max(np.maximum(o[n]['books'], o[n]['sample_gap']) / o[n]['Y']) for n in 'AB'))
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage12()
    print(f'段 12: 決め直す割合 1 なら前と同じ {same:.1e}、割合を下げた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 13
def test_stage13(T=150):
    """段 13: 信念の区間は、選ばなければ(learn_attitude が 'interval' 以外)何も変えない(三つの勘定は記録だけ)。
    区間の幅を最小(q = 0.5、床 0)にし α = 1 なら、上の端は max(中央値, 平均) = 平均で、中立の信念と同じになる。帳簿が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.05, relief='state', learn=True)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, amb_q=0.2, amb_floor=0.7, amb_alpha=0.3, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'protect'])
    e = im1.Im1(learn_attitude='interval', amb_q=0.5, amb_floor=0.0, **kw)
    worst, gap = 0.0, 0.0
    from scipy.stats import gamma
    for _ in range(T):
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
        a_, b_ = e.p.prior_a + e.n_eff, e.p.prior_b + e.t_eff
        gap = max(gap, abs(e.belief - a_ / b_))
    return same, worst, gap


if __name__ == '__main__':
    same, worst, gap = test_stage13()
    print(f'段 13: 区間を選ばなければ前と同じ {same:.1e}、区間の帳簿と標本の差/所得 {worst:.1e}、最小の区間と平均の信念の差 {gap:.1e}')


# ---------------------------------------------------------------- 段 14
def test_stage14(T=150):
    """段 14: 流出 0・国庫 0 なら、企業を置いた経済は前と同じ数。流出と国庫を入れても、帳簿と標本の差が閉じ、占有率の和は 1。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, firms=12)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, p_spin=0.0, t_prize=0.0, prize_reinvest=False, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'gamma_front'])
    e = im1.Im1(p_spin=0.05, t_prize=0.2, ip_lag=3, **kw)
    worst, s_err = 0.0, 0.0
    for _ in range(T):
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
        s_err = max(s_err, float(np.max(np.abs(e.fp.s.sum(1) - 1))))
    return same, worst, s_err


if __name__ == '__main__':
    same, worst, s_err = test_stage14()
    print(f'段 14: 流出と国庫を切れば前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}、占有率の和のずれ {s_err:.1e}')


# ---------------------------------------------------------------- 段 15
def test_stage15(T=150):
    """段 15: 貸し手の型 'none' なら、企業を置いた経済は前と同じ数(悪い企業の割合などを変えても)。関係融資・市場型・半々で、
    企業の倒産の損を銀行が書いても、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, firms=12)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, lender='none', p_bad=0.9, d_bad=0.5, zombie=0.1, crunch=5.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw', 'gamma_front'])
    worst, loss = 0.0, 0.0
    for lender in ('relation', 'market', 'mix', 'blind'):
        e = im1.Im1(lender=lender, d_bad=0.3, **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
            loss += r.get('loss', 0.0)
    return same, worst, loss


if __name__ == '__main__':
    same, worst, loss = test_stage15()
    print(f'段 15: 貸し手の型を入れなければ前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}(貸倒れの和 {loss:.0f})')


# ---------------------------------------------------------------- 段 16
def test_stage16(T=150):
    """段 16: 死蔵なしなら前と同じ数。死蔵(準備は内生・固定)と運を入れても、帳簿と標本の差が閉じる。準備が固定で足りなければ、貸せる割合は 1 未満。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, hoard=False, hoard_cap=2.0, reserves='fixed', res_supply=0.01, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst, cfmin = 0.0, 1.0
    for res in ('endog', 'fixed'):
        e = im1.Im1(hoard=True, reserves=res, res_supply=0.2, luck_sigma=0.2, is_split=5, **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
            if res == 'fixed':
                cfmin = min(cfmin, r['cf'])
    return same, worst, cfmin


if __name__ == '__main__':
    same, worst, cfmin = test_stage16()
    print(f'段 16: 死蔵なしなら前と同じ {same:.1e}、死蔵の帳簿と標本の差/所得 {worst:.1e}、固定の準備での貸せる割合の最小 {cfmin:.2f}')


# ---------------------------------------------------------------- 段 17
def test_stage17(T=150):
    """段 17: 歴史の重点標本化を使わなければ(dis_q = 0)前と同じ数。使っても帳簿が閉じ、重みの対数は、引いた災害の回数から計算した値と一致する。
    粗い時間の進め方(im_time.projective)は模型の外の包みで、模型の数を変えない。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.03, relief='state')
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, dis_q=0.0, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    e = im1.Im1(dis_q=0.1, **kw)
    n_ev, worst = 0, 0.0
    for _ in range(T):
        r = e.step()
        n_ev += int(r.get('dis_size', 0.0) > 0)
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    lw = n_ev * np.log(0.03 / 0.1) + (T - n_ev) * np.log(0.97 / 0.9)
    return same, worst, abs(e.logw - lw)


if __name__ == '__main__':
    same, worst, dw = test_stage17()
    print(f'段 17: 重点標本化を使わなければ前と同じ {same:.1e}、使った帳簿と標本の差/所得 {worst:.1e}、重みの対数のずれ {dw:.1e}')


# ---------------------------------------------------------------- 段 18(別枠の模型 upgrade.py)
def test_stage18(T=200):
    """段 18: 格上げの別枠の模型でも、毎年の帳簿(七つの部門、預金・貸出・二つの持分、二つの能力)が閉じる。体制の交代と持分の移転を含めて。"""
    import upgrade
    worst = 0.0
    for kw in (dict(), dict(pricing='fundamental'), dict(politics='revolution', tauX_max=2.0, rev_thr=0.3, expropriate=True)):
        a, m = upgrade.simulate(T=T, **kw)
        worst = max(worst, float(np.max(a['books'] / a['Y'])))
    return worst


if __name__ == '__main__':
    print(f'段 18(別枠): 格上げの模型の帳簿のずれ/所得 {test_stage18():.1e}')


# ---------------------------------------------------------------- 段 15b
def test_stage15b(T=150):
    """段 15b: 貸し手を入れなければ(lending = 'none')前と同じ数。市場型・関係融資で、失敗した事業の能力と貸倒れを書いても、帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005)
    a, _ = im1.simulate(T=T, **kw)
    b, _ = im1.simulate(T=T, lending='none', kappa_x=0.9, p0_s=0.5, h_caut=0.3, **kw)
    same = max(np.max(np.abs(a[k] - b[k]) / (np.abs(a[k]) + 1)) for k in ['W', 'PN', 'E', 'tau', 'pT', 'gini_nw'])
    worst = 0.0
    for lend in ('market', 'relation'):
        e = im1.Im1(lending=lend, kappa_x=0.6, **kw)
        for _ in range(T):
            r = e.step()
            worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    return same, worst


if __name__ == '__main__':
    same, worst = test_stage15b()
    print(f'段 15b: 貸し手を入れなければ前と同じ {same:.1e}、入れた帳簿と標本の差/所得 {worst:.1e}')


# ---------------------------------------------------------------- 段 17b と 段 8b
def test_stage17b_8b():
    """段 17b: 平行世界(多段の分割)は、複製した世界の帳簿も閉じ、乱数を替えた複製は別の未来をたどる。段 8b: 文化の波及 0 なら前と同じ数。"""
    import copy, im1, im_time
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, dis_rate=0.05, relief='state')
    e = im1.Im1(**kw)
    for _ in range(60):
        e.step()
    a, b = copy.deepcopy(e), copy.deepcopy(e)
    rng = np.random.default_rng(1)
    im_time.reseed(a, rng); im_time.reseed(b, rng)
    worst, diff = 0.0, 0.0
    for _ in range(60):
        ra, rb = a.step(), b.step()
        worst = max(worst, ra['books'] / ra['Y'], rb['books'] / rb['Y'])
        diff = max(diff, abs(ra['E'] - rb['E']))
    kw2 = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, psi_v=1.0)
    x, _ = im1.simulate(T=100, **kw2)
    y, _ = im1.simulate(T=100, culture_spill=0.0, **kw2)
    same = max(np.max(np.abs(x[k] - y[k]) / (np.abs(x[k]) + 1)) for k in ['W', 'PN', 'E', 'gamma_front'])
    return worst, diff, same


if __name__ == '__main__':
    worst, diff, same = test_stage17b_8b()
    print(f'段 17b: 複製した世界の帳簿/所得 {worst:.1e}、乱数を替えた二つの世界の雇用の差の最大 {diff:.0f}(別の未来)。段 8b: 波及 0 なら前と同じ {same:.1e}')


def test_stage19():
    """段 19: プレイヤーの記入。切らなければ前と同じ数。切り出しは総量を変えず、選び方(支出・働く度合い・子・投票・呼びかけ)と
    代替わり・重点標本化を入れても、帳簿と標本の差が閉じる。プレイヤーの記入は分けも併せもされない。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True)
    x, _ = im1.simulate(T=60, **kw)
    y, _ = im1.simulate(T=60, player=False, k_sway=0.5, **kw)
    same = max(np.max(np.abs(x[k] - y[k]) / (np.abs(x[k]) + 1)) for k in ['W', 'PN', 'E'])
    a = im1.Im1(**kw)
    b = im1.Im1(player=True, **kw)
    carve = max(abs(getattr(a.H, n).sum() - getattr(b.H, n).sum()) / (abs(getattr(a.H, n).sum()) + 1) for n in a.EXT)
    e = im1.Im1(player=True, vote=True, rho_w=5e-4, rho_k=1e-2, child_cost=0.5, ret_age=45, pension='funded', turnover=0.05,
                is_split=2, disaster=True, dis_rate=0.05, relief='state', **kw)
    worst, h0 = 0.0, e.H.heads[e.pl]
    for t in range(120):
        e.act = dict(spend=0.6 + 0.8 * (t % 2), work=0.5, child=(t % 30 == 0), vote='hi' if t % 20 < 10 else 'lo', campaign=1.0)
        r = e.step()
        worst = max(worst, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    kept = abs(e.H.heads[e.pl] - h0)
    return same, carve, worst, kept, r['pl_gen']


if __name__ == '__main__':
    same, carve, worst, kept, gen = test_stage19()
    print(f'段 19: プレイヤーなしなら前と同じ {same:.1e}、切り出しの総量のずれ {carve:.1e}、選び方と代替わりを入れた帳簿と標本の差/所得 {worst:.1e}、'
          f'プレイヤーの人数の変わり {kept:.1e}、代の数 {gen}')


def test_stage10b_15c():
    """段 10b: 基金が株を持たなければ前と同じ数。持てば、株の市場で売り買いしても帳簿と標本の差が閉じる(預金 = 家計の預金 + 基金の預金)。
    段 15c: 探索を借りず、ペテンと記憶が 0 なら前と同じ数。入れても帳簿と標本の差が閉じる。"""
    import im1
    kw = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True,
              ret_age=45, pension='funded', eq_market=True, eq_pricing='demand')
    x, _ = im1.simulate(T=80, **kw)
    y, _ = im1.simulate(T=80, fund_eq=0.0, phi_q=0.0, **kw)
    same10 = max(np.max(np.abs(x[k] - y[k]) / (np.abs(x[k]) + 1)) for k in ['W', 'PN', 'E'])
    worst10 = 0.0
    for extra in (dict(fund_eq=0.6, phi_q=0.05), dict(fund_eq=0.6, eq_pricing='fundamental')):
        e = im1.Im1(**dict(kw, **extra))
        for _ in range(150):
            r = e.step()
            worst10 = max(worst10, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    kw2 = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, lending='market', p0_s=0.9)
    x, _ = im1.simulate(T=80, **kw2)
    y, _ = im1.simulate(T=80, res_loan=0.0, phi_sw=0.0, k_mem=0.0, g_rel_x=0.9, **kw2)
    same15 = max(np.max(np.abs(x[k] - y[k]) / (np.abs(x[k]) + 1)) for k in ['W', 'PN', 'E'])
    worst15 = 0.0
    for lend in ('market', 'relation'):
        e = im1.Im1(**dict(kw2, lending=lend, res_loan=0.5, eq_market=True, eq_pricing='demand', phi_sw=0.3, k_mem=20.0))
        for _ in range(150):
            r = e.step()
            worst15 = max(worst15, r['books'] / r['Y'], r['sample_gap'] / r['Y'])
    return same10, worst10, same15, worst15


if __name__ == '__main__':
    s10, w10, s15, w15 = test_stage10b_15c()
    print(f'段 10b: 基金が株を持たなければ前と同じ {s10:.1e}、持ったときの帳簿と標本の差/所得 {w10:.1e}。'
          f'段 15c: 探索・ペテン・記憶が 0 なら前と同じ {s15:.1e}、入れたときの帳簿と標本の差/所得 {w15:.1e}')
