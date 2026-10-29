# results_stage0.py: 段 0 の実験。労働者の貯蓄と、投資の利潤への反応で、gj の「資本家寄り」の結論がどう変わるか。
# python results_stage0.py > results_stage0.txt
import im_gj

P = im_gj.default_params
KEYS = ('L', 'unemp', 'Mw', 'Mk', 'I', 'pL')


def row(label, r):
    note = '  ← 釣り合いに収束しない(賃金を固定したままの超過需要)' if r['it'] >= 19999 else ''
    print(f'{label:24s} ' + ' '.join(f'{k}={r[k]:.3f}' for k in KEYS) + f" 資本家の実質消費={r['Ck'] / r['pL']:.3f}" + note)


def e1_worker_saving():
    print('E1 労働者の貯蓄 s_w と再分配 τ(国債は初め資本家が 1 持つ)')
    for sw in [0.0, 0.05, 0.1]:
        for tau in [0.0, 0.2, 0.4]:
            row(f's_w={sw} τ={tau}', im_gj.solve_year(P().set(s_w=sw, tau=tau), 0.0, 1.0))
    print('一般のカレツキの式: (1 − ι)s·Mk + s_w·Mw = 赤字')
    for sw in [0.0, 0.1]:
        p = P().set(s_w=sw)
        r = im_gj.solve_year(p, 0.0, 1.0)
        lhs = (1 - p.iota) * r['s'] * r['Mk'] + sw * r['Mw']
        print(f'  s_w={sw}: 左辺 {lhs:.6f}、赤字 {p.G + p.iB * 1.0:.6f}')


def e2_bond_holders(T=50):
    print(f'E2 {T} 年後、国債を誰が持つか')
    for sw in [0.0, 0.05, 0.1]:
        m = im_gj.make_model(P().set(s_w=sw))
        h = m.run(T)
        r = h[-1]
        print(f"  s_w={sw}: 労働者の国債 {r['Bw']:.3f}、資本家の国債 {r['Bk']:.3f}、"
              f"資本家の所得 {r['Mk']:.3f}、労働者の所得 {r['Mw']:.3f}、失業 {r['unemp']:.3f}、帳簿 {max(x['books'] for x in h):.1e}")


def e3_bhaduri_marglin():
    print('E3 投資の関数(バドゥリ–マーグリン): I = i0 + i_h·h(h は税と再分配のあとの利潤の取り分)。基準 τ = 0.2 で I が gj と同じになるよう i0 を合わせる')
    base = im_gj.solve_year(P(), 0.0, 1.0)
    h0 = base['Mk'] / (base['Mw'] + base['Mk'])
    print(f'  基準: 利潤の取り分 h = {h0:.3f}、投資 I = {base["I"]:.3f}')
    for ih in [0.0, 0.5, 1.0, 2.0]:
        i0 = base['I'] - ih * h0
        for tau in [0.1, 0.2, 0.3]:
            row(f'i_h={ih} τ={tau}', im_gj.solve_year(P().set(inv='bm', i0=i0, i_h=ih, tau=tau), 0.0, 1.0))


if __name__ == '__main__':
    e1_worker_saving()
    e2_bond_holders()
    e3_bhaduri_marglin()
