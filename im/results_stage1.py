# results_stage1.py: 段 1 の実験。python results_stage1.py > results_stage1.txt
import numpy as np
import im1

BASE = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, fiscal='deficit')   # 記録は「赤字を 0 に」のルールで出した
SHOW = ('E', 'real_wage', 'top10_inc', 'rbar', 'tau', 'infl')


def steady(T=400, n=100, **kw):
    a, e = im1.simulate(T=T, **dict(BASE, **kw))
    r = {k: float(np.mean(a[k][-n:])) for k in a if k not in ('inv_limited',)}
    r['infl_sd'] = float(np.std(a['infl'][-n:]))
    return r, a, e


def fmt(r, keys=SHOW):
    out = '  '.join(f'{k}={r[k]:.3f}' if abs(r[k]) < 100 else f'{k}={r[k]:.0f}' for k in keys)
    return out + (f"±{r['infl_sd']:.3f}" if 'infl_sd' in r and 'infl' in keys else '')


def e1_base():
    print('E1 基準の定常状態(400 年、最後の 100 年の平均。技術の進歩なし、家計は 400 世帯の標本)')
    for lab, kw in (('信用なし', dict(credit=False)), ('信用あり(慎重さなし)', dict())):
        r, a, e = steady(**kw)
        print(f'  {lab:22s} {fmt(r)}  帳簿 {a["books"].max():.1e}  標本の差 {a["sample_gap"].max():.1e}')


def e2_caution(theta=0.15):
    print(f'E2 貸し手の慎重さ μ を上げ、下げる(閾値 θ = {theta}。各 150 年の最後の値。Φ 固定は、集中を基準の値に止めた比較。G/Y は国債の対所得比)')
    e = im1.Im1(**dict(BASE, theta_prem=theta))
    for _ in range(200):
        r0 = e.step()
    phi0 = r0['top10_inc']
    mus = [round(x, 2) for x in np.arange(0, 1.61, 0.2)]
    for mu in mus + mus[::-1][1:]:
        e.p.mu_prem = mu
        for _ in range(150):
            r = e.step()
        fixed, _, _ = steady(theta_prem=theta, mu_prem=mu, phi_fixed=phi0) if mu in mus and e.t < 200 + 150 * len(mus) + 1 else (None, 0, 0)
        extra = f'   | Φ 固定: E={fixed["E"]:.0f} rbar={fixed["rbar"]:.3f}' if fixed else ''
        print(f"  μ={mu:.1f}  {fmt(r, SHOW[:-1])}  G/Y={r['G'] / r['Y']:.2f}{extra}")


def e3_redistribution(mu=0.5, theta=0.15):
    print(f'E3 再分配(資本所得の税 τ_k、税収を一人あたり同額で配る)と、投資の行動の仮定(μ = {mu}、θ = {theta})')
    regimes = (('需要主導(稼働率に強く反応)', dict(phi_r=0.2, phi_u=0.6)),
               ('基準', dict()),
               ('利潤主導(利潤率に強く反応)', dict(phi_r=1.5, phi_u=0.1)))
    for lab, kw in regimes:
        print(f'  {lab}')
        for tk in (0.2, 0.4, 0.6):
            r, a, e = steady(mu_prem=mu, theta_prem=theta, tau_k=tk, redistrib=1.0, **kw)
            print(f'    τ_k={tk:.1f}  {fmt(r)}')


def e4_debt_deflation():
    print('E4 借入の決まりと担保の上限: 平時の借入 (1 − f_R)δ/amort が上限 λ を超えると(amort < 0.117)、借入が上限に張り付く')
    for am in (0.3, 0.2, 0.15, 0.12, 0.1, 0.08):
        r, a, e = steady(amort=am)
        print(f"  amort={am:.2f}  {fmt(r)}  借入/所得={r['loans'] / r['Y']:.2f}")


def e5_taylor():
    print('E5 テイラー・ルール(政策金利 = max(0, 0.02 + φ_π × 物価上昇率))。r̄ が貸出の金利に連動するので、利上げは費用を通じて物価を上げる')
    cases = (('基準', dict()), ('慎重さ μ = 1', dict(mu_prem=1.0, theta_prem=0.15)),
             ('需要主導、τ_k = 0.2、μ = 0.5', dict(mu_prem=0.5, theta_prem=0.15, tau_k=0.2, redistrib=1.0, phi_r=0.2, phi_u=0.6)))
    for lab, kw in cases:
        for pp in (0.0, 0.6, 1.0, 1.5):
            try:
                r, a, e = steady(phi_pi=pp, **kw)
                print(f"  {lab:28s} φ_π={pp:.1f}  {fmt(r)}  政策金利={r['r_policy']:.3f}")
            except (ValueError, FloatingPointError, AssertionError):
                print(f"  {lab:28s} φ_π={pp:.1f}  発散(物価が無限に上がり、価格が解けなくなる)")


if __name__ == '__main__':
    e1_base()
    e2_caution()
    e3_redistribution()
    e4_debt_deflation()
    e5_taylor()
