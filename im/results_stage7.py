# results_stage7.py: 段 7 の実験(危機時の生産: 在庫、価格の上限と品不足、最後の生産者、損失の分担、備蓄、公共事業、固定費と廃棄)。
# python -W ignore results_stage7.py > results_stage7.txt
# 一国、標本 2000 記入、乱数の種 12 個。打撃は 200 年目。同じ種で打撃のない歴史と比べる。財政は既定(利払いの重さ)、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True)   # 記録の道(動きは同じ)
SEEDS = range(12)
T0, T1 = 200, 225
KEYS = ('Y', 'bN', 'E', 'real_wage', 'infl', 'b', 'short', 'hungry', 'sub_gap', 'subsidy', 'share_pay', 'gov_buy', 'KN', 'KL', 'KI',
        'S_G', 'InvN', 'capped', 'scrap', 'tau', 'xN_prod', 'hhN')


def history(args):
    """(設定, 打撃, 種)。打撃: 'cost'(原料の労働の係数 ×1.5 と必需品の能力 ×0.6、dur 年。上限は打撃の前の必需品の価格)、
    'slump'(労働者の消費の割合 0.9 → 0.75 を dur 年)、'none'。cap: 上限を置く年数(0 なら置かない)。"""
    kw, shock, dur, cap, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **{k: v for k, v in kw.items() if k != 'pw_on'}))
    rec = []
    try:
        for t in range(T1):
            if t == T0 and shock == 'cost':
                e.K[0] *= 0.6
                e.gamma[2] *= 1.5; e.gamma_new[2] *= 1.5
                if cap:
                    e.p.set(cap_N=float(e.prices[0]))
            if t == T0 + dur and shock == 'cost':
                e.gamma[2] /= 1.5; e.gamma_new[2] /= 1.5
            if shock == 'cost' and cap and t == T0 + cap:
                e.p.set(cap_N=0.0)
            if t == T0 and shock == 'slump':
                e.H.cw = (0.75, 0.2)
                if 'pw_on' in kw:
                    e.p.set(**kw['pw_on'])
            if t == T0 + dur and shock == 'slump':
                e.H.cw = (0.9, 0.2)
                if 'pw_on' in kw:
                    e.p.set(g_pw=0.0, g_tr=0.0)
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    return {k: np.array([r.get(k, 0.0) for r in rec], float) for k in KEYS} | {'books': max(r['books'] / r['Y'] for r in rec),
                                                                               'gap': max(r['sample_gap'] / r['Y'] for r in rec)}


def run(cases, shock, dur):
    """cases: (ラベル, kw, cap)。"""
    jobs = []
    for lab, kw, cap in cases:
        kw2 = {k: v for k, v in kw.items() if k != 'pw_on'}
        for s in SEEDS:
            jobs.append((kw, shock, dur, cap, s))
            jobs.append((kw2, 'none', dur, 0, s))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(history, jobs))
    out, k = {}, 0
    for lab, kw, cap in cases:
        ps = []
        for s in SEEDS:
            a, z = res[k], res[k + 1]
            k += 2
            if a is not None and z is not None:
                ps.append((a, z))
        out[lab] = ps
    return out


W = slice(T0, T0 + 15)


def m(ps, f):
    v = np.array([f(a, z) for a, z in ps])
    return v.mean(), v.std() / np.sqrt(len(v))


def fm(ps, f, spec='.3f'):
    a, s = m(ps, f)
    return f'{a:{spec}}±{s:{spec.lstrip("+")}}'


realY = lambda a, z: (a['Y'] / a['bN'])[W].sum() / (z['Y'] / z['bN'])[W].sum() - 1
relE = lambda a, z: a['E'][W].sum() / z['E'][W].sum() - 1
relw = lambda a, z: a['real_wage'][W].mean() / z['real_wage'][W].mean() - 1
dB = lambda a, z: a['b'][T0 + 14] - z['b'][T0 + 14]
KN = lambda a, z: a['KN'][T0 + 14] / z['KN'][T0 + 14] - 1
KLI = lambda a, z: (a['KL'][T0 + 14] + a['KI'][T0 + 14]) / (z['KL'][T0 + 14] + z['KI'][T0 + 14]) - 1
pk_infl = lambda a, z: a['infl'][W].max()


def table(out, extra=()):
    print('  場合                        | 崩れ | 品不足(平均) | 品不足(頂) | 生存の量に届かない人の割合の平均 | 同(打撃のない歴史) | 生存の量の不足(平均) | 物価の頂 | 実質の生産 | 雇用 | 実質賃金 | 国債/所得の差 | 必需品の能力 | 他の能力 | 補填/所得 | 分担/所得' + ''.join(' | ' + h for h, _ in extra))
    for lab, ps in out.items():
        if not ps:
            print(f'  {lab:26s} | すべて崩れた'); continue
        n_bad = len(SEEDS) - len(ps)
        cols = [fm(ps, lambda a, z: a['short'][W].mean()), fm(ps, lambda a, z: a['short'][W].max()), fm(ps, lambda a, z: a['hungry'][W].mean()), fm(ps, lambda a, z: z['hungry'][W].mean()),
                fm(ps, lambda a, z: a['sub_gap'][W].mean(), '.4f'), fm(ps, pk_infl, '+.3f'), fm(ps, realY, '+.3f'), fm(ps, relE, '+.3f'),
                fm(ps, relw, '+.3f'), fm(ps, dB, '+.3f'), fm(ps, KN, '+.3f'), fm(ps, KLI, '+.3f'),
                fm(ps, lambda a, z: (a['subsidy'][W] / a['Y'][W]).mean(), '.4f'), fm(ps, lambda a, z: (a['share_pay'][W] / a['Y'][W]).mean(), '.4f')]
        cols += [fm(ps, f, sp) for _, (f, sp) in extra]
        print(f'  {lab:26s} | {n_bad:2d} | ' + ' | '.join(cols))
    bk = max(max(a['books'], a['gap']) for ps in out.values() for a, z in ps)
    print(f'  (帳簿と標本の差/所得の最大 {bk:.0e})')


INV = dict(inventory=True)


def c1_cost():
    print('C1 原料の費用の急騰(原料の一製品あたり労働 ×1.5 を 10 年)と、必需品の能力の 4 割の損失(200 年目)。必需品の価格の上限は、打撃の前の価格で 10 年')
    print('   15 年(200〜214 年)。品不足 = 家計の必需品の需要のうち買えなかった割合。生存の量に届かない人 = 割り当てで一人 q0 に届かない人の割合')
    print('   差と比は、同じ種で打撃のない歴史に対して。補填 = 最後の生産者の政府の補填、分担 = 損失の分担で他の部門が払った額')
    cases = (('上限なし(価格は自由)', dict(), 0),
             ('上限なし + 在庫', INV, 0),
             ('上限、手当てなし', dict(), 10),
             ('上限 + 在庫', INV, 10),
             ('上限 + 最後の生産者', dict(crisis_rule='last'), 10),
             ('上限 + 損失の分担', dict(crisis_rule='share'), 10),
             ('上限 + 備蓄 0.3', dict(g_stock=0.3), 10),
             ('上限 + 在庫 + 備蓄 0.3', dict(INV, g_stock=0.3), 10),
             ('上限なし + 備蓄 0.3', dict(g_stock=0.3), 0))
    table(run(cases, 'cost', 10))


def c2_duration():
    print('C2 上限の長さ: 同じ打撃で、上限を 2 年だけ置く場合と 10 年置く場合(在庫あり)')
    cases = (('上限 2 年 + 在庫', INV, 2), ('上限 10 年 + 在庫', INV, 10),
             ('上限 2 年 + 在庫 + 最後の生産者', dict(INV, crisis_rule='last'), 2), ('上限 10 年 + 在庫 + 最後の生産者', dict(INV, crisis_rule='last'), 10),
             ('上限 2 年 + 在庫 + 備蓄 0.3', dict(INV, g_stock=0.3), 2))
    table(run(cases, 'cost', 10))


def c3_fixed():
    print('C3 固定費と能力の廃棄: 労働者の消費の割合 0.9 → 0.75 を 10 年(需要の不足)。能力一単位あたり維持の労働 0.01(生産の労働の係数 0.1 の 1 割)、赤字の部門は能力を捨てる(率 0.2)')
    print('   最後の生産者は、必需品の能力を捨てさせず、赤字を埋める。上限はなし')
    FK = dict(f_K=0.01, delta_scrap=0.2)
    cases = (('固定費なし', dict(), 0), ('固定費と廃棄', FK, 0), ('固定費と廃棄 + 最後の生産者', dict(FK, crisis_rule='last'), 0),
             ('固定費と廃棄 + 在庫', dict(FK, **INV), 0))
    extra = (('比べの歴史の雇用の揺れ(100〜200 年の標準偏差)', (lambda a, z: z['E'][100:200].std(), '.0f')),
             ('比べの歴史の原料の能力の揺れ(変動係数)', (lambda a, z: z['KI'][100:200].std() / z['KI'][100:200].mean(), '.2f')),
             ('15 年の廃棄/能力', (lambda a, z: a['scrap'][W].sum() / (a['KN'][T0] + a['KL'][T0] + a['KI'][T0]), '.3f')))
    print('   注: 固定費と廃棄があると、打撃のない経済そのものが大きな景気の波を持つ(下の「比べの歴史」の列)。15 年後の能力の比は波の位相に左右される')
    table(run(cases, 'slump', 10), extra)


def c4_public_works():
    print('C4 需要の不足(同じ 10 年の不況)で、公共事業(投資財を去年の所得の 2% 買う)と、同じ額の一人あたり同額の給付を比べる(不況の 10 年だけ)')
    cases = (('何もしない', dict(), 0), ('公共事業 2%', dict(pw_on=dict(g_pw=0.02)), 0), ('給付 2%', dict(pw_on=dict(g_tr=0.02)), 0),
             ('公共事業 4%', dict(pw_on=dict(g_pw=0.04)), 0), ('給付 4%', dict(pw_on=dict(g_tr=0.04)), 0))
    W10 = slice(T0, T0 + 10)
    extra = (('不況の 10 年の雇用', (lambda a, z: a['E'][W10].sum() / z['E'][W10].sum() - 1, '+.3f')),
             ('不況の 10 年の実質の生産', (lambda a, z: (a['Y'] / a['bN'])[W10].sum() / (z['Y'] / z['bN'])[W10].sum() - 1, '+.3f')))
    table(run(cases, 'slump', 10), extra)


if __name__ == '__main__':
    c1_cost()
    c2_duration()
    c3_fixed()
    c4_public_works()
