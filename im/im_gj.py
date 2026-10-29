# im_gj.py: ノート M 第 2 章の二財の模型(gj)を、im の骨組みの上に書き直したもの(段 0 のテスト)。
# 足した部品(既定では切ってあり、切ればノート M の gj.period と同じ数になる):
#   - 労働者の貯蓄 s_w: 労働者も所得の一部を貯め、国債を持つ(年金基金にあたる。Pasinetti 1962)。国債の利子の一部が労働者に入る。
#   - 投資の関数 inv = 'bm': 投資が雇用(稼働)と利潤の取り分に反応する(Bhaduri and Marglin 1990)。既定の 'saving' は gj と同じ(資本家の貯蓄の一定割合)。
import numpy as np
from im_core import Params, Ledger, Model, Component

SECTORS = ('wk', 'kp', 'fN', 'fL', 'bank', 'gov')     # 労働者、資本家、必需品の企業、贅沢品の企業、銀行、政府
INSTRUMENTS = ('deposits', 'loans', 'bonds')


def default_params():
    p = Params()
    # 構造
    p.add('Hw', 0.9, 'structure', '労働者の人口')
    p.add('Hk', 0.1, 'structure', '資本家の人口')
    p.add('Lmax', 1.0, 'structure', '労働力')
    p.add('aL', 1.0, 'structure', '贅沢品の生産性')
    p.add('a', 1.5, 'structure', '必需品の生産性(技術)')
    p.add('n', 1.0, 'structure', '贅沢品の種類の数(文化)')
    # 行動の仮定
    p.add('gw', 0.3, 'behavior', '労働者一人あたりの必需品の必要量')
    p.add('gk', 0.5, 'behavior', '資本家一人あたりの必需品の必要量')
    p.add('bw', 0.3, 'behavior', '労働者の余裕のうち必需品に回す割合')
    p.add('bk', 0.1, 'behavior', '資本家の同じ割合')
    p.add('s', 0.5, 'behavior', '資本家の貯蓄の割合')
    p.add('s_w', 0.0, 'behavior', '労働者の貯蓄の割合(国債で持つ)')
    p.add('iota', 0.5, 'behavior', '投資 = ι × 資本家の貯蓄(inv = saving のとき)')
    p.add('psi', 0.0, 'behavior', '文化が資本家の貯蓄の割合を下げる強さ')
    p.add('rho', 0.75, 'behavior', '贅沢品の種類の間の代替のしやすさ(値上げ幅 1/ρ)')
    p.add('inv', 'saving', 'behavior', "投資の関数: 'saving'(gj と同じ)か 'bm'(バドゥリ–マーグリン)")
    p.add('i0', 0.0, 'behavior', "'bm' の投資の定数")
    p.add('i_u', 0.0, 'behavior', "'bm' の投資の、雇用(稼働)への反応")
    p.add('i_h', 0.0, 'behavior', "'bm' の投資の、利潤の取り分への反応")
    # 政策
    p.add('tau', 0.2, 'policy', '資本家の所得への税率(全額を労働者に配る)')
    p.add('G', 0.02, 'policy', '公共事業(人を雇う。国債で賄う)')
    p.add('ic', 0.03, 'policy', '運転資金の金利')
    p.add('iB', 0.02, 'policy', '国債の金利')
    return p


def stone_geary(M, need, b, pN):
    if M <= pN * need:
        return M, 0.0
    extra = M - pN * need
    return pN * need + b * extra, (1 - b) * extra


def solve_year(p, Bw, Bk, Lcap=np.inf, tol=1e-12):
    """年の中の釣り合い(速い層)。所得・価格・雇用を、支出 = 所得になるまで反復で解く。
    労働者の貯蓄 s_w = 0、inv = 'saving' のとき、ノート M の gj.period と同じ計算になる。"""
    a, n = p.a, p.n
    cN = (1 + p.ic) / a
    cL = (1 + p.ic) / (p.rho * p.aL)
    s = p.s / (1 + p.psi * n)
    B = Bw + Bk
    pN, pL, Mw, Mk, L = cN, cL, 0.5, 0.2, 0.9
    for it in range(20000):
        ENw, ELw = stone_geary((1 - p.s_w) * Mw, p.Hw * p.gw, p.bw, pN)
        Ck = (1 - s) * Mk
        ENk, ELk = stone_geary(Ck, p.Hk * p.gk, p.bk, pN)
        EN, EL = ENw + ENk, ELw + ELk
        LN = min(EN / pN / a, Lcap)
        if p.inv == 'saving':
            I = p.iota * s * Mk
        else:
            # 利潤の取り分は、税と再分配のあとの資本家の所得で測る(再分配が投資の誘因に効く)
            h = Mk / max(Mw + Mk, 1e-12)
            I = max(p.i0 + p.i_u * L + p.i_h * h, 0.0)
            I = min(I, max(p.Lmax - LN - p.G - 1e-6, 0.0))   # 人手が足りなければ、投資は人手の分まで
        pN_new = max(cN, EN / (a * Lcap)) if np.isfinite(Lcap) else cN
        free = max(p.Lmax - LN - I - p.G, 1e-9)
        LL = min(EL / pL / p.aL, free)
        pL_new = max(cL, EL / (p.aL * free))
        L = LN + LL + I + p.G
        rent = EN - (1 + p.ic) * LN
        profit = EL - (1 + p.ic) * LL
        Mk_pre = rent + profit + p.ic * (LN + LL) + p.iB * Bk
        T = p.tau * Mk_pre
        Mw_new, Mk_new = L + T + p.iB * Bw, Mk_pre - T
        err = abs(Mw_new - Mw) + abs(Mk_new - Mk) + abs(pN_new - pN) + abs(pL_new - pL)
        d = 0.3
        Mw += d * (Mw_new - Mw); Mk += d * (Mk_new - Mk)
        pN += d * (pN_new - pN); pL += d * (pL_new - pL)
        if err < tol:
            break
    return dict(pN=pN, pL=pL, cN=cN, cL=cL, ENw=ENw, ELw=ELw, ENk=ENk, ELk=ELk, EN=EN, EL=EL,
                LN=LN, LL=LL, I=I, L=L, unemp=p.Lmax - L, rent=rent, profit=profit, Mw=Mw, Mk=Mk, T=T,
                Ck=Ck, s=s, it=it)


def post_year(led, p, r, Bw, Bk):
    """解いた釣り合いを、帳簿の取引として書き込む(基本の層)。"""
    LN, LL = r['LN'], r['LL']
    # 運転資金: 銀行が賃金の分を貸し、期末に返してもらう
    led.trade('loans', 'bank', 'fN', LN)
    led.trade('loans', 'bank', 'fL', LL)
    # 賃金(企業、資本家の投資、政府の公共事業)
    led.pay('wages', 'fN', 'wk', LN)
    led.pay('wages', 'fL', 'wk', LL)
    led.pay('investment', 'kp', 'wk', r['I'])
    led.pay('public works', 'gov', 'wk', p.G)
    # 消費
    led.pay('consumption N', 'wk', 'fN', r['ENw'])
    led.pay('consumption L', 'wk', 'fL', r['ELw'])
    led.pay('consumption N', 'kp', 'fN', r['ENk'])
    led.pay('consumption L', 'kp', 'fL', r['ELk'])
    # 運転資金の利子と返済、銀行の利益の配当
    led.pay('loan interest', 'fN', 'bank', p.ic * LN)
    led.pay('loan interest', 'fL', 'bank', p.ic * LL)
    led.trade('loans', 'fN', 'bank', LN)
    led.trade('loans', 'fL', 'bank', LL)
    led.pay('bank dividends', 'bank', 'kp', p.ic * (LN + LL))
    # 企業の利潤(レントと、贅沢品の値上げ幅の利潤)
    led.pay('profits', 'fN', 'kp', r['rent'])
    led.pay('profits', 'fL', 'kp', r['profit'])
    # 国債の利子、税と再分配
    led.pay('bond interest', 'gov', 'wk', p.iB * Bw)
    led.pay('bond interest', 'gov', 'kp', p.iB * Bk)
    led.pay('tax', 'kp', 'gov', r['T'])
    led.pay('transfer', 'gov', 'wk', r['T'])
    # 赤字を国債で賄う。労働者は貯蓄の分を買い、残りを資本家が買う
    deficit = p.G + p.iB * (Bw + Bk)
    bw_buy = p.s_w * r['Mw']
    led.trade('bonds', 'wk', 'gov', bw_buy)
    led.trade('bonds', 'kp', 'gov', deficit - bw_buy)
    return deficit, bw_buy


def make_model(p=None, Bw=0.0, Bk=1.0, Lcap=np.inf):
    """gj の経済を、骨組みの Model として組む。国債の初めの残高は資本家が Bk、労働者が Bw を持つ。"""
    p = p or default_params()
    led = Ledger(SECTORS, INSTRUMENTS)
    # 初めの国債を置く(過去の赤字の積み重ね。取引ではなく、期首の残高として)
    led.revalue('bonds', 'kp', +Bk); led.revalue('bonds', 'wk', +Bw); led.revalue('bonds', 'gov', -(Bk + Bw))
    led.close()
    led.t = 0
    state = {'Lcap': Lcap}

    def year(m):
        Bw_ = m.L.stock[('wk', 'bonds')]
        Bk_ = m.L.stock[('kp', 'bonds')]
        r = solve_year(m.p, Bw_, Bk_, m.state['Lcap'])
        deficit, bw_buy = post_year(m.L, m.p, r, Bw_, Bk_)
        # 釣り合いが解けていれば、各部門の預金は動かない(支出 = 所得)
        dep = max(abs(m.L.fin['deposits'][s]) for s in m.L.sectors)
        return dict(r, Bw=Bw_, Bk=Bk_, deficit=deficit, dep_drift=dep)

    return Model(led, p, [Component('goods market and income', year, 'base')], state)
