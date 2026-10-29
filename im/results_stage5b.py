# results_stage5b.py: 段 5 の探索(企業の研究、株式の市場、研究の割合の学習)。python -W ignore results_stage5b.py > results_stage5b.txt
# 標本 2000 記入、400 年の後半 300 年。乱数の種 16 個。± は平均の標準誤差。財政のルールは既定(利払いの重さ)、技術の自然な進み g_gamma = 0。
import os
import numpy as np
from scipy.optimize import brentq
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005)
SEEDS = range(16)
T, BURN = 400, 100
MEAN_GAIN = 0.006        # 当たり一回の平均の大きさ(対数)。裾を変えても、これを揃える


def g0_for(alpha, cap=0.3, mean=MEAN_GAIN):
    """上限 cap で切ったパレート型(下限 g0、指数 alpha)の平均が mean になる g0。"""
    def m(g0):
        if abs(alpha - 1) < 1e-9:
            return g0 + g0 * np.log(cap / g0)
        return g0 + g0 ** alpha * (cap ** (1 - alpha) - g0 ** (1 - alpha)) / (1 - alpha)
    return brentq(lambda g: m(g) - mean, 1e-9, mean)


def tail(alpha):
    return dict(alpha_R=alpha, g0=g0_for(alpha))


KEYS = ('Y', 'bN', 'PN', 'E', 'real_wage', 'div', 'top10_inc', 'top10_nw', 'gini_nw', 'gamma_front', 'sR', 'gain', 'infl',
        'r_loan', 'R_eq', 'pE', 'eq_fund', 'R_ip', 'qIP', 'ip_fund', 'res_w', 'W', 'unemp', 'breakups')


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    rec = []
    try:
        for _ in range(T):
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError, OverflowError):
        return dict(collapse=1.0, years=len(rec))
    a = {k: np.array([r.get(k, np.nan) for r in rec], float) for k in KEYS}
    n = slice(BURN, T)
    realY = a['Y'] / a['bN']
    gr = np.diff(np.log(realY[n]))
    prod = np.log(a['gamma_front'][BURN] / a['gamma_front'][T - 1]) / (T - 1 - BURN)
    sR = a['sR'][n]
    # 崩れ: 名目の賃金が 300 年で 1/1000 を下回る(デフレの渦)
    collapse = float(a['W'][T - 1] < a['W'][BURN] * 1e-3)
    real_div = a['div'][n] / a['PN'][n]
    out = dict(collapse=collapse, years=T, prod=prod, gY=np.mean(gr), sdY=np.std(gr), E=a['E'][n].mean(),
               rw_g=np.log(a['real_wage'][T - 1] / a['real_wage'][BURN]) / (T - 1 - BURN),
               div_g=np.log(max(real_div[-20:].mean(), 1e-12) / max(real_div[:20].mean(), 1e-12)) / (T - 1 - BURN),
               top10i=a['top10_inc'][n].mean(), top10=a['top10_nw'][n].mean(), sR=sR.mean(), sR_sd=sR.std(),
               giveup=np.mean(sR < 0.005), infl=a['infl'][n].mean(), unemp=a['unemp'][n].mean(),
               bursty=float(np.mean(a['gain'][n] > 0) and np.max(a['gain'][n]) / max(np.sum(a['gain'][n]), 1e-12)))
    if np.isfinite(a['R_eq'][n]).any():
        # 実質の利回りは、対数で平均する(幾何平均。揺れの大きい利回りの算術平均は高めに出る)
        lr = np.log1p(a['R_eq'][n]) - np.log1p(a['infl'][n])
        out.update(R_eq=np.nanmean(lr), xs_eq=np.nanmean(np.log1p(a['R_eq'][n]) - np.log1p(a['r_loan'][n])),
                   pe_fund=np.nansum(a['pE'][n]) / np.sum(a['eq_fund'][n]), sd_Req=np.nanstd(lr))
    if np.isfinite(a['R_ip'][n]).any():
        lr = np.log1p(a['R_ip'][n].clip(-0.99)) - np.log1p(a['infl'][n])
        out.update(R_ip=np.nanmean(lr), xs_ip=np.nanmean(np.log1p(a['R_ip'][n].clip(-0.99)) - np.log1p(a['r_loan'][n])))
    H = e.H
    nw = H.M + e.pT * H.land + (e.pE or 0.0) * H.eq + (e.qIP or 0.0) * H.ip
    o = np.argsort(-nw / H.heads)
    top = o[np.cumsum(H.heads[o]) / H.heads.sum() <= 0.1]
    out['eq_top10'] = float(H.eq[top].sum() / H.eq.sum())        # 最後の年に、富の上位 10% が持つ持分の割合
    out['r_real'] = np.mean(np.log1p(a['r_loan'][n]) - np.log1p(a['infl'][n]))
    return out


def table(cases):
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    out = {}
    for i, (lab, kw) in enumerate(cases):
        rs = res[i * len(SEEDS):(i + 1) * len(SEEDS)]
        ok = [r for r in rs if r['collapse'] == 0 and 'prod' in r]
        r = {k: float(np.mean([x[k] for x in ok])) for k in ok[0]} if ok else {}
        se = {k: float(np.std([x[k] for x in ok]) / np.sqrt(len(ok))) for k in ok[0]} if ok else {}
        r['n_collapse'] = sum(1 for x in rs if x['collapse'] > 0)
        out[lab] = (r, se)
    return out


def x1_tail():
    print('X1 研究の割合(売上に対して固定)と、当たりの大きさの裾(指数 α。小さいほど巨大な当たりが出る)。当たり一回の平均の大きさは 0.006 に揃える')
    print('   技術の進み = 最先端の労働の係数が一年に下がる割合。崩れ = 名目の賃金が 1/1000 を下回ったデフレの渦(16 種のうち)。他の数は崩れなかった種の平均')
    print('   当たりなし = 研究に人を雇うが当たらない(需要としての研究だけ)。研究の額は去年の利益(利子のあと)の半分までに縛る')
    print('   一番の当たりの割合 = 300 年の向上のうち、一番大きな一年の当たりが占める割合(裾が重いと、少数の大当たりが技術の進みを決める)')
    cases = [('研究なし', dict()), ('当たりなし s_R=0.02', dict(research=True, s_R=0.02, p_hit=0.0))]
    for a in (3.0, 1.5, 1.1):
        for s in (0.01, 0.02, 0.03, 0.05, 0.1):
            cases.append((f'α={a} s_R={s}', dict(research=True, s_R=s, **tail(a))))
    out = table(cases)
    print('  場合            | 崩れ | 技術の進み | 実質の生産の伸び | その揺れ | 雇用 | 失業者 | 実質賃金の伸び | 実質配当の伸び | 所得の上位10% | 富の上位10% | 物価 | 一番の当たりの割合')
    for lab, (r, se) in out.items():
        if 'prod' not in r:
            print(f'  {lab:16s} | {r["n_collapse"]:2d} | すべて崩れた'); continue
        print(f"  {lab:16s} | {r['n_collapse']:2d} | {r['prod']:.4f}±{se['prod']:.4f} | {r['gY']:+.4f} | {r['sdY']:.3f} | {r['E']:.0f} | {r['unemp']:.0f} | "
              f"{r['rw_g']:+.4f} | {r['div_g']:+.4f} | {r['top10i']:.3f} | {r['top10']:.3f} | {r['infl']:+.3f} | {r['bursty']:.2f}")
    return out


def x2_adaptive():
    print('X2 研究の割合を、過去の当たりから学ぶ(見込みの比 b = 当たりの見込み × 生産の労働 × 5 年、割合 = 0.1 × b/(1+b))。')
    print('   あきらめ = 研究の割合が 0.005 を下回る年の割合。探索の床 s_R_min を置くと、あきらめの罠を防げるか')
    cases = []
    for a in (3.0, 1.5, 1.1):
        cases.append((f'α={a} 固定 0.02', dict(research=True, s_R=0.02, **tail(a))))
        for om in (0.9, 0.98):
            for fl in (0.0, 0.01):
                cases.append((f'α={a} 学ぶ ω={om} 床 {fl}', dict(research=True, adaptive_R=True, omega_R=om, s_R_min=fl, **tail(a))))
    out = table(cases)
    print('  場合                     | 崩れ | 技術の進み | 研究の割合 | その揺れ | あきらめ | 実質の生産の伸び | 雇用 | 実質賃金の伸び')
    for lab, (r, se) in out.items():
        if 'prod' not in r:
            print(f'  {lab:24s} | {r["n_collapse"]:2d} | すべて崩れた'); continue
        print(f"  {lab:24s} | {r['n_collapse']:2d} | {r['prod']:.4f}±{se['prod']:.4f} | {r['sR']:.3f} | {r['sR_sd']:.3f} | {r['giveup']:.2f} | "
              f"{r['gY']:+.4f} | {r['E']:.0f} | {r['rw_g']:+.4f}")
    return out


def x3_markets():
    print('X3 株式の市場と知財の市場: 買い手の実際の利回り(実質、対数の平均)と、名目の利回りの貸出の金利との差。値段/基本の値打ち = 市場の値段の和 / (配当 ÷ r̄)の和')
    print('   株の値段の既定は基本の値打ち(均した配当 ÷ r̄)で、市場は持ち主を入れ替えるだけ。置きたい割合: 基準 0.12、富の弾力性 κ_E = 0.5(富むほど株に置く)、')
    print('   稼ぐ力は数えない、借りて買えない(λ_E = 0)。「需要で値段」は標本の望む保有の和 = 1 となる値段。崩れ = 解けなくなった種(物価の発散)')
    print('   知財の市場は段 2b と同じ(需要で値を決める)。研究ありは α=1.5、s_R=0.02')
    R = dict(research=True, s_R=0.02, **tail(1.5))
    cases = (('株式の市場(既定)', dict(eq_market=True)),
             ('株式 κ_E=0(皆が同じ割合)', dict(eq_market=True, kappa_E=0.0)),
             ('株式 κ_E=0、稼ぐ力も数える', dict(eq_market=True, kappa_E=0.0, human_E=1.0)),
             ('株式 κ_E=1', dict(eq_market=True, kappa_E=1.0)),
             ('株式 信用取引 λ_E=0.5', dict(eq_market=True, lam_E=0.5)),
             ('株式 需要で値段 κ_E=0', dict(eq_market=True, eq_pricing='demand', kappa_E=0.0)),
             ('株式 需要で値段 κ_E=0.5', dict(eq_market=True, eq_pricing='demand')),
             ('株式 需要で値段 κ_E=0.5 λ_E=1', dict(eq_market=True, eq_pricing='demand', lam_E=1.0)),
             ('株式の市場 + 研究', dict(eq_market=True, **R)),
             ('知財の市場', dict(ip_market=True)),
             ('知財の市場 基本の値打ち', dict(ip_market=True, ip_pricing='fundamental')),
             ('知財と株式の市場', dict(ip_market=True, eq_market=True)),
             ('知財と株式の市場 + 研究', dict(ip_market=True, eq_market=True, **R)),
             ('市場なし(比べ)', dict()),
             ('市場なし + 研究(比べ)', R))
    out = table(cases)
    print('  場合                        | 崩れ | 株の実質の利回り | 株−貸出の金利 | その揺れ | 値段/基本 | 知財の実質の利回り | 知財−貸出の金利 | 貸出の実質の金利 | 物価 | 実質の生産の揺れ | 雇用 | 富の上位10% | 所得の上位10% | 上位10%の持分')
    f = lambda r, k, fmt: (format(r[k], fmt) if k in r else '  —  ')
    for lab, (r, se) in out.items():
        if 'prod' not in r:
            print(f'  {lab:26s} | {r["n_collapse"]:2d} | すべて崩れた'); continue
        print(f"  {lab:26s} | {r['n_collapse']:2d} | {f(r, 'R_eq', '+.3f')} | {f(r, 'xs_eq', '+.3f')} | {f(r, 'sd_Req', '.3f')} | {f(r, 'pe_fund', '.2f')} | "
              f"{f(r, 'R_ip', '+.3f')} | {f(r, 'xs_ip', '+.3f')} | {r['r_real']:+.3f} | {r['infl']:+.3f} | {r['sdY']:.3f} | {r['E']:.0f} | {r['top10']:.3f} | {r['top10i']:.3f} | {r['eq_top10']:.2f}")
    return out


if __name__ == '__main__':
    print('g0(当たりの下限)を、平均を揃えるよう決めた値:', {a: round(g0_for(a), 5) for a in (3.0, 1.5, 1.1)})
    x1_tail()
    x2_adaptive()
    x3_markets()
