# results_stage18.py: 段 18 格上げ(別枠の模型 upgrade.py)の実験。python -W ignore results_stage18.py > results_stage18.txt
# 決定論の模型(乱数なし)。300 年。
import numpy as np
import upgrade


def summary(kw, T=300):
    try:
        a, m = upgrade.simulate(T=T, **kw)
    except (AssertionError, ValueError, FloatingPointError):
        return None
    take = lambda th: int(np.argmax(a['sX'] >= th)) if (a['sX'] >= th).any() else None
    pf = a['pX'] / a['fund']
    return dict(t10=take(0.1), t30=take(0.3), t50=take(0.5), sX_end=a['sX'][-1], KX100=a['KX'][100], q_mean=a['q'][:150].mean(),
                pf_max=pf[5:].max(), pf_mean=pf[5:150].mean(), u_mean=a['u'].mean(), u_max=a['u'][5:].max(), wo=a['wo'].sum(),
                rw_end=a['real_w'][-1] / a['real_w'][0], books=a['books'].max(), rev=m.revolution, tau_mean=a['tauX'].mean())


def table(title, cases):
    print(title)
    print('  場合                              | 新しい財の割合が 10% / 30% / 50% に届く年 | 300 年目の割合 | 100 年目の新しい能力 | 初め 150 年の q | 値段/基本(平均、頂) | 失業(平均、頂) | 棒引き | 実質賃金(300年/初め) | 新しい財への税の平均 | 革命の年')
    for lab, kw in cases:
        r = summary(kw)
        if r is None:
            print(f'  {lab:32s} | 崩れた'); continue
        f = lambda v: '—' if v is None else str(v)
        print(f"  {lab:32s} | {f(r['t10'])} / {f(r['t30'])} / {f(r['t50'])} | {r['sX_end']:.2f} | {r['KX100']:.0f} | {r['q_mean']:.2f} | {r['pf_mean']:.2f}, {r['pf_max']:.1f} | "
              f"{r['u_mean']:.3f}, {r['u_max']:.2f} | {r['wo']:.0f} | {r['rw_end']:.2f} | {r['tau_mean']:.3f} | {f(r['rev'])}")
    print('  (帳簿の最大のずれは、すべての場合で 1e-6 × 消費 より小さい: 各年の L.check で確かめている)')


if __name__ == '__main__':
    table('U18-1 新しい企業の持分の値段の決め方(泡があるか)と、新しい産業の育ち方', (
        ('基本の値打ち(泡なし)', dict(pricing='fundamental')),
        ('需要で値段、外挿なし(ν=0)', dict(nu=0.0)),
        ('需要で値段、外挿 0.5', dict()),
        ('需要で値段、外挿 1.0', dict(nu=1.0)),
        ('需要で値段、借りて買えない(lev=0)', dict(lev=0.0)),
        ('需要で値段、置きたい割合の反応 10', dict(theta1=10.0))))
    table('U18-2 古い体制の抵抗(古い資本家の富の割合で新しい財に税)と、革命(新しい財で働く人と失業者の割合が閾値を超えると体制が替わる)', (
        ('体制なし', dict()),
        ('抵抗(税の上限 0.5)', dict(politics='protect')),
        ('抵抗(税の上限 2.0)', dict(politics='protect', tauX_max=2.0)),
        ('抵抗 2.0 + 革命(閾値 0.5)', dict(politics='revolution', tauX_max=2.0)),
        ('抵抗 2.0 + 革命(閾値 0.3)', dict(politics='revolution', tauX_max=2.0, rev_thr=0.3)),
        ('抵抗 2.0 + 革命(閾値 0.3)+ 古い持分を労働者へ', dict(politics='revolution', tauX_max=2.0, rev_thr=0.3, expropriate=True)),
        ('基本の値打ち + 抵抗 2.0', dict(pricing='fundamental', politics='protect', tauX_max=2.0))))
