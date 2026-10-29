# results_stage4b.py: 段 4b の実験(財政の行き詰まりと、誰が負担するか)。python -W ignore results_stage4b.py > results_stage4b.txt
# 災害が毎年 5% の確率で起きる歴史(500 年の後半 300 年)。標本 2000 記入、乱数の種 24 個。± は平均の標準誤差。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005,
            disaster=True, dis_rate=0.05, relief='state', mu_B=0.3, theta_B=0.6, fiscal='deficit')   # 'burden' の場合は BURDEN で上書きする
BURDEN = dict(fiscal='burden', burden_target=0.016, phi_pi=1.0, rbar_link=0.3, theta_resolve=0.55)
SEEDS = range(24)
T, BURN = 500, 200


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError):
        return None
    a = {k: np.array([r[k] for r in rec], float) for k in ('b', 'premB', 'E', 'Y', 'bN', 'tau', 'gini_nw', 'top10_nw', 'top10_inc', 'breakups',
                                                           'levy', 'crisis', 'defaulted', 'infl', 'r_policy', 'real_wage', 'prem_income', 'hh_debt')}
    n = slice(BURN, T)
    Y = a['Y'][n]
    return dict(crisis=float(np.any((a['premB'][n] > 0.05) | (a['b'][n] > 3.0))), maxb=a['b'][n].max(), maxprem=a['premB'][n].max(),
                realY=np.mean(Y / a['bN'][n]), E=a['E'][n].mean(), rw=a['real_wage'][n].mean(), tau=a['tau'][n].mean(),
                gini=a['gini_nw'][n].mean(), top10=a['top10_nw'][n].mean(), top10i=a['top10_inc'][n].mean(),
                brk=a['breakups'][n].sum() / (T - BURN) * 100, zone=a['crisis'][n].mean(), levy=np.mean(a['levy'][n] / Y),
                prem=np.mean(a['prem_income'][n] / Y), defaults=float((a['defaulted'][n] > 0).sum()), infl=a['infl'][n].mean(),
                r=a['r_policy'][n].mean(), zlb=np.mean(a['r_policy'][n] <= 1e-9))


def table(cases):
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    out = {}
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        out[lab] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} if rs else None
        if out[lab] is not None:
            out[lab]['n_ok'] = len(rs)
    return out


def f1_who_pays():
    print('F1 災害が毎年 5% の確率で起きる歴史での、財政の行き詰まりの道。上乗せ金利: 国債/所得 0.6 から(強さ 0.3)。行き詰まりの道は国債/所得 0.55 から取る')
    print('   財政のルールは「利払いの重さ」(目標 0.016。テイラー・ルール 1.0、r̄ の金利への連動 0.3)。比べとして「赤字を 0 に」(同じ中央銀行)')
    cases = (('赤字を 0 に、道なし', dict(phi_pi=1.0, rbar_link=0.3)),
             ('利払いの重さ、道なし', BURDEN),
             ('利払いの重さ、富裕税', dict(BURDEN, resolve='levy')),
             ('利払いの重さ、緊縮', dict(BURDEN, resolve='austerity')),
             ('利払いの重さ、踏み倒し', dict(BURDEN, resolve='default')),
             ('利払いの重さ、金融抑圧', dict(BURDEN, resolve='repression')))
    out = table(cases)
    print('  道                     | 破綻の割合 | 国債/所得の頂 | 上乗せの頂 | 実質の生産 | 雇用 | 実質賃金 | 正味資産のジニ | 上位10%の富 | 一家離散/100年 | 富裕税/所得 | 上乗せの利子/所得 | 踏み倒しの回数')
    for lab, r in out.items():
        print(f"  {lab:20s} | {r['crisis']:.2f} | {r['maxb']:.2f} | {r['maxprem']:.3f} | {r['realY']:.0f} | {r['E']:.0f} | {r['rw']:.3f} | "
              f"{r['gini']:.3f} | {r['top10']:.3f} | {r['brk']:.0f} | {r['levy']:.4f} | {r['prem']:.4f} | {r['defaults']:.1f}")
    return out


def f2_rates():
    print('F2 利下げで借りる余地が広がるか: 利払いの重さのルール、道なし。テイラー・ルールの強さと、r̄ の金利への連動')
    cases = (('金利は固定(φ_π = 0)', dict(BURDEN, phi_pi=0.0)),
             ('φ_π = 0.5、連動 0.3', dict(BURDEN, phi_pi=0.5)),
             ('φ_π = 1.0、連動 0.3', BURDEN),
             ('φ_π = 1.0、連動 1.0(全部)', dict(BURDEN, rbar_link=1.0)),
             ('φ_π = 1.5、連動 0.3', dict(BURDEN, phi_pi=1.5)))
    out = table(cases)
    for lab, r in out.items():
        if r is None or r['n_ok'] < len(SEEDS):
            print(f"  {lab:24s} 発散する種がある(物価が無限に上がり、価格が解けなくなる。{0 if r is None else int(r['n_ok'])}/{len(SEEDS)} の種だけ最後まで。以下はその種の平均)")
            if r is None:
                continue
        head = f"  {'':24s}" if r['n_ok'] < len(SEEDS) else f"  {lab:24s}"
        print(head + f" 破綻の割合 {r['crisis']:.2f}  国債/所得の頂 {r['maxb']:.2f}  実質の生産 {r['realY']:.0f}  雇用 {r['E']:.0f}  "
              f"政策金利の平均 {r['r']:.3f}(0 に張り付く年の割合 {r['zlb']:.2f})  物価 {r['infl']:+.3f}  一家離散 {r['brk']:.0f}")


SERIES = [(250, 0.25, 'common'), (258, 0.25, 'common'), (266, 0.25, 'common')]


def series(args):
    """大きな災害が三度続く歴史(250、258、266 年目)と、同じ種で災害のない歴史。250〜330 年を比べる。"""
    kw, seed = args
    out = []
    for q in (SERIES, []):
        e = im1.Im1(seed=seed, **dict(BASE, dis_rate=0.0, **kw))
        e.shock_queue = list(q)
        try:
            rec = [e.step() for _ in range(330)]
        except (ValueError, FloatingPointError, AssertionError):
            return None
        out.append({k: np.array([r[k] for r in rec], float) for k in ('b', 'premB', 'E', 'Y', 'bN', 'W', 'gini_nw', 'top10_nw',
                                                                      'breakups', 'levy', 'prem_income', 'defaulted', 'real_wage', 'tau')})
    a, z = out
    w = slice(250, 330)
    pre = slice(230, 250)
    Yz = z['Y'][w].sum()
    return dict(maxb=a['b'][w].max(), maxprem=a['premB'][w].max(), crisis=float(np.any((a['premB'][w] > 0.05) | (a['b'][w] > 3.0))),
                loss=1 - (a['Y'][w] / a['bN'][w]).sum() / (z['Y'][w] / z['bN'][w]).sum(),
                lab_loss=1 - (a['E'][w] * a['real_wage'][w]).sum() / (z['E'][w] * z['real_wage'][w]).sum(),
                dtop10=a['top10_nw'][329] - z['top10_nw'][329], dgini=a['gini_nw'][329] - z['gini_nw'][329],
                brk=a['breakups'][w].sum() - z['breakups'][w].sum(), levy=a['levy'][w].sum() / Yz * len(range(250, 330)),
                prem=a['prem_income'][w].sum() / Yz * len(range(250, 330)), defaults=float((a['defaulted'][w] > 0).sum()),
                b_end=a['b'][329], tau_max=a['tau'][w].max())


def f3_series():
    print('F3 大きな災害が三度続く(250、258、266 年目に大きさ 0.25 の共通の打撃。救済は国)。250〜330 年を、同じ種で災害のない歴史と比べる。乱数の種 24 個の平均 ± 標準誤差')
    print('   生産の損 = 実質の生産の 80 年の和の減り方。労働の損 = 雇用 × 実質賃金の和の減り方。富裕税と上乗せの利子は、80 年の和を一年の所得で割った値')
    cases = (('赤字を 0 に、道なし', dict(phi_pi=1.0, rbar_link=0.3)),
             ('利払いの重さ、道なし', BURDEN),
             ('利払いの重さ、富裕税', dict(BURDEN, resolve='levy')),
             ('利払いの重さ、緊縮', dict(BURDEN, resolve='austerity')),
             ('利払いの重さ、踏み倒し', dict(BURDEN, resolve='default')),
             ('利払いの重さ、金融抑圧', dict(BURDEN, resolve='repression')),
             ('利払いの重さ、道なし、金利は固定', dict(BURDEN, phi_pi=0.0)))
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(series, jobs))
    print('  道                     | 破綻 | 国債/所得の頂 | 上乗せの頂 | 税率の頂 | 生産の損 | 労働の損 | 上位10%の富の差 | ジニの差 | 一家離散の差 | 富裕税 | 上乗せの利子 | 踏み倒し | 80年後の国債/所得')
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:20s} | 発散'); continue
        r = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
        se = {k: float(np.std([x[k] for x in rs]) / np.sqrt(len(rs))) for k in rs[0]}
        print(f"  {lab:20s} | {r['crisis']:.2f}±{se['crisis']:.2f} | {r['maxb']:.2f} | {r['maxprem']:.3f} | {r['tau_max']:.2f} | {r['loss']:+.3f}±{se['loss']:.3f} | {r['lab_loss']:+.3f} | "
              f"{r['dtop10']:+.3f}±{se['dtop10']:.3f} | {r['dgini']:+.3f} | {r['brk']:+.0f}±{se['brk']:.0f} | {r['levy']:.3f} | {r['prem']:.3f} | {r['defaults']:.1f} | {r['b_end']:.2f}"
              + (f'  ({len(rs)}/{len(SEEDS)} の種だけ最後まで)' if len(rs) < len(SEEDS) else ''))


if __name__ == '__main__':
    f1_who_pays()
    f3_series()
    f2_rates()
