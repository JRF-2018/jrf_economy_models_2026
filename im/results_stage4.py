# results_stage4.py: 段 4 の実験(災害)。python -W ignore results_stage4.py > results_stage4.txt
# ノート 3 第 7 章の問い: どの安全網が、どの打撃に効くか。価格と信用と土地のある経済でも、結論は残るか。
import numpy as np
import im1, im3

BASE = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, fiscal='deficit')
T0 = 250                 # 災害の年
SEEDS = range(6)


def one(seed, size=0.3, scope='common', T=T0 + 30, **kw):
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    e.shock_queue = [(T0, size, scope)] if size > 0 else []
    hitB = None
    rec = []
    for t in range(T):
        rec.append(e.step())
        if t == T0 - 1:
            B_pre = e._B.copy()
        if t == T0:
            hitB = (e.last_hit.copy() if size > 0 else None)
    a = {k: np.array([r[k] for r in rec], float) for k in ('E', 'S', 'real_wage', 'breakups', 'wo_h', 'Y', 'bN', 'gini_nw', 'relief', 'G', 'hh_debt')}
    a['realY'] = a['Y'] / a['bN']
    return a, e, hitB


def avg(scope, size=0.3, **kw):
    """乱数の種ごとに、災害のあと 10 年の値を、直前 20 年の平均に対する比で。一家離散は 10 年の合計(人数)。"""
    out = {k: [] for k in ('Ymin', 'Y10', 'rwmin', 'brk', 'brk0', 'dgini', 'relief', 'debt')}
    pre, post = slice(T0 - 20, T0), slice(T0, T0 + 10)
    for sd in SEEDS:
        a, e, hit = one(sd, size, scope, **kw)
        b, _, _ = one(sd, 0.0, scope, **kw)                       # 同じ種で災害なし(比べる相手)
        out['Ymin'].append(a['realY'][post].min() / a['realY'][pre].mean())
        out['Y10'].append(a['realY'][T0 + 9] / a['realY'][pre].mean())
        out['rwmin'].append(a['real_wage'][post].min() / a['real_wage'][pre].mean())
        out['brk'].append(a['breakups'][post].sum())
        out['brk0'].append(b['breakups'][post].sum())
        out['dgini'].append(a['gini_nw'][T0 + 9] - b['gini_nw'][T0 + 9])
        out['relief'].append(a['relief'][T0] / a['Y'][pre].mean())
        out['debt'].append(a['G'][T0 + 9] / a['Y'][T0 + 9] - b['G'][T0 + 9] / b['Y'][T0 + 9])
    return {k: float(np.mean(v)) for k, v in out.items()}


def fmt(r):
    return (f"実質の生産の底 ×{r['Ymin']:.3f}  10 年後 ×{r['Y10']:.3f}  実質賃金の底 ×{r['rwmin']:.3f}  "
            f"一家離散(10 年、人数){r['brk']:.1f}(災害なし {r['brk0']:.1f})  正味資産のジニの差 {r['dgini']:+.3f}  "
            f"救済/所得 {r['relief']:.3f}  国債/所得の差 {r['debt']:+.3f}")


def d1_common():
    print(f'D1 大きな共通の打撃(大きさ 0.3: 資本の 3 割が壊れ、全員が働く力の 15% を失う)。乱数の種 {len(SEEDS)} つの平均')
    for rel in ('none', 'state', 'mutual'):
        print(f'  救済 {rel:7s} {fmt(avg("common", relief=rel))}')


def d2_idio():
    print(f'D2 ばらばらの打撃(大きさ 0.3: 人数で 1 割の世帯が働く力をすべて失い、資本は 3% 壊れる)。乱数の種 {len(SEEDS)} つの平均')
    for rel in ('none', 'state', 'mutual'):
        print(f'  救済 {rel:7s} {fmt(avg("idio", relief=rel))}')


def d3_protection(T=500, burn=200):
    print('D3 災害が毎年 5% の確率で起きる(大きさはパレート型、下限 0.05、指数 1.5、上限 0.8。共通の打撃)。守りの資本の量を変える。')
    print('   300 年(200 年目から)の平均。乱数の種 4 つ。守りは、政府が前の年の所得の割合で人を雇って積む')
    for pr in (0.0, 0.005, 0.01, 0.02, 0.04):
        res = []
        for sd in range(4):
            e = im1.Im1(seed=sd, **dict(BASE, dis_rate=0.05, protect=pr, relief='state'))
            rec = [e.step() for _ in range(T)]
            a = {k: np.array([r[k] for r in rec], float) for k in ('Y', 'bN', 'real_wage', 'breakups', 'G', 'dis_size', 'E')}
            ry = (a['Y'] / a['bN'])[burn:]
            drops = ry[1:] / ry[:-1]
            res.append((ry.mean(), np.percentile(drops, 1), a['real_wage'][burn:].mean(), a['breakups'][burn:].sum() / (T - burn) * 100,
                        np.mean(a['G'][burn:] / a['Y'][burn:]), (a['dis_size'][burn:] > 0).sum()))
        r = np.mean(res, axis=0)
        print(f"  守り {pr:.3f}  実質の生産 {r[0]:.1f}  最悪 1% の年の落ち込み ×{r[1]:.3f}  実質賃金 {r[2]:.3f}  一家離散 {r[3]:.1f} 人/100 年  "
              f"国債/所得 {r[4]:.2f}  災害の回数 {r[5]:.1f}")


def d4_regions():
    print('D4 二つの地域。250 年目に大きさ 0.3 の共通の打撃が、A だけ(局地)か、両方(共通)に起きる。救済は国。共通の通貨と変動の為替')
    C = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True, relief='state', fiscal='deficit')
    for cur in ('common', 'float'):
        for where in ('A', 'both'):
            w = im3.World(common=C, home=0.5, tc=0.1, currency=cur)
            for i, e in enumerate(w.R):
                if where == 'both' or i == 0:
                    e.shock_queue = [(T0, 0.3, 'common')]
            for t in range(T0 + 15):
                w.step()
            out = []
            for i, nm in enumerate(w.names):
                ry = np.array([rs[i]['Y'] / rs[i]['bN'] for rs in w.log]); rw = np.array([rs[i]['real_wage'] for rs in w.log])
                pre = slice(T0 - 20, T0)
                out.append(f"{nm}: 実質の生産の底 ×{ry[T0:T0 + 10].min() / ry[pre].mean():.3f} 実質賃金の底 ×{rw[T0:T0 + 10].min() / rw[pre].mean():.3f}")
            print(f"  {'共通の通貨' if cur == 'common' else '変動の為替':6s} {'A だけ' if where == 'A' else '両方':4s}  " + '   '.join(out))


def d5_decompose(T=340):
    print('D5 災害のあとの好況の分解(大きさ 0.3 の共通の打撃、救済なし、乱数の種 0)。災害なしの同じ種に対する、実質の生産の比')
    cases = (('基準(赤字 0 のルール、建て直しの反応 φ_u = 0.3、資産効果 c_m = 0.2)', {}),
             ('建て直しの投資の反応を弱める(φ_u = 0.1)', dict(phi_u=0.1)),
             ('財政のルールを「国債/所得 = 0.28」に(強さ 0.2)', dict(fiscal='debt', b_target=0.28, k_debt=0.2)),
             ('資産効果を弱める(c_m = 0.1)', dict(c_m=0.1)),
             ('資産効果を弱める(c_m = 0.05)', dict(c_m=0.05)))
    for lab, kw in cases:
        a, _, _ = one(0, 0.3, 'common', T=T, **kw)
        b, _, _ = one(0, 0.0, 'common', T=T, **kw)
        r = a['realY'] / b['realY']
        g = lambda d, t: d['G'][t] / d['Y'][t]
        print(f"  {lab}")
        print(f"     5 年 ×{r[T0 + 5]:.2f}  10 年 ×{r[T0 + 10]:.2f}  20 年 ×{r[T0 + 20]:.2f}  40 年 ×{r[T0 + 40]:.2f}  89 年 ×{r[T0 + 89]:.2f}   "
              f"国債/所得: 災害の前 {g(b, T0 - 1):.2f}、89 年後 {g(a, T0 + 89):.2f}(災害なし {g(b, T0 + 89):.2f})  "
              f"89 年後の失業(1 − 雇用/働きたい人){1 - a['E'][T0 + 89] / one_S(a, T0 + 89):.3f}")


def one_S(a, t):
    return a['S'][t]


def d6_importance(seeds=12):
    print('D6 一家離散の数の推定と重点標本化(災害なし、400 年の後半 200 年、100 年あたりの人数)。乱数の種ごとの平均と標準偏差')
    import importlib
    rows = (('普通の標本 400', 400, {}, seeds), ('重点標本化 400(平均で併せる、毎年 5 組)', 400, dict(is_split=5), seeds),
            ('重点標本化 400(ルーレット、毎年 5 組)', 400, dict(is_split=5, is_mode='roulette'), seeds),
            ('普通の標本 2000', 2000, {}, max(seeds // 2, 2)))
    for lab, Ns, kw, ns in rows:
        r = []
        for sd in range(ns):
            e = im1.Im1(seed=sd, **dict(BASE, Ns=Ns, disaster=False, **kw))
            rec = [e.step() for _ in range(400)]
            x = {k: np.array([q[k] for q in rec], float) for k in ('breakups', 'E', 'gini_nw', 'hh_debt', 'Y')}
            n = slice(200, 400)
            r.append((x['breakups'][n].sum() / 2, x['E'][n].mean(), x['gini_nw'][n].mean(), np.mean(x['hh_debt'][n] / x['Y'][n])))
        r = np.array(r)
        print(f"  {lab:34s} 一家離散 {r[:, 0].mean():5.1f} ± {r[:, 0].std():4.1f}  雇用 {r[:, 1].mean():.1f} ± {r[:, 1].std():.1f}  "
              f"正味資産のジニ {r[:, 2].mean():.3f} ± {r[:, 2].std():.3f}  借金/所得 {r[:, 3].mean():.4f} ± {r[:, 3].std():.4f}")


if __name__ == '__main__':
    d1_common()
    d2_idio()
    d3_protection()
    d4_regions()
    d5_decompose()
    d6_importance()
