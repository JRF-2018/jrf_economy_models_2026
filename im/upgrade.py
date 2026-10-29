# upgrade.py: 段 18 格上げ(別枠の模型)。経済の中で新しい経済が育つ(ノート 1.5、Perez 2002 の技術経済パラダイムの転換)。
# im の枠組み(im_core の帳簿、残差ではなく年ごとの流れ、パラメータの登録簿)を使った、小さな別の模型。
#
# 財は二つ: 古い財 O(消費にも資本にも使う。労働の係数は一定)と、新しい財 X(消費だけ。作った量の累計とともに労働の係数が下がる。ライトの法則)。
# 家計は三つの群れ: 労働者 w、古い資本家 ko(古い企業の持分を持つ)、新しい投資家 kn(新しい企業の持分を買う。銀行から借りて買える)。
# 新しい企業の投資は、持分を売って賄う(トービンの q が 1 を超える分だけ投資する)。持分の値段の決め方は二つ:
#   'demand'(新しい投資家の望む保有で決まる。値上がりの外挿があると泡になる)、'fundamental'(均した配当 ÷ r̄)。
# 政治(体制): 古い資本家の富の割合が大きいほど、古い体制は新しい財に税(規制)をかける。
#   革命: 新しい財の企業で働く人と失業者の割合が閾値を超えると、体制が替わり、税がなくなる(古い企業の持分を労働者に移す道も選べる)。
import numpy as np
from im_core import Params, Ledger


def params():
    p = Params()
    p.add('N', 1000.0, 'structure', '働ける人')
    p.add('gO', 0.1, 'structure', '古い財の一製品あたり労働')
    p.add('gX0', 0.3, 'structure', '新しい財の、初めの一製品あたり労働')
    p.add('gX_min', 0.03, 'structure', '新しい財の一製品あたり労働の下限')
    p.add('learn_b', 0.3, 'structure', 'ライトの法則: 一製品あたり労働 = 初め × (累計 / 初めの累計)^(−b)')
    p.add('cum0', 500.0, 'structure', '新しい財の初めの累計(試作の経験)')
    p.add('omega', 1.0, 'structure', '能力を超えて作るときの残業の割増(im と同じ形)')
    p.add('kn0', 300.0, 'structure', '新しい投資家の初めの預金')
    p.add('kappa', 2.0, 'structure', '能力一単位に要る古い財(資本の係数)')
    p.add('delta', 0.05, 'structure', '能力の減耗')
    p.add('m', 0.3, 'behavior', '値上げ幅(変動費の上)')
    p.add('sigma', 3.0, 'behavior', '古い財と新しい財の間の代わりの利きやすさ')
    p.add('aX', 0.1, 'behavior', '新しい財への好みの重み(古い財は 1 − aX)')
    p.add('c_w', 0.9, 'behavior', '労働者の所得からの消費の割合')
    p.add('c_k', 0.4, 'behavior', '資本家の所得からの消費の割合')
    p.add('c_m', 0.05, 'behavior', '資本家の預金からの消費の割合')
    p.add('phi_u', 0.3, 'behavior', '古い企業の投資の、稼働率への反応')
    p.add('phi_q', 0.3, 'behavior', '新しい企業の投資の、トービンの q への反応')
    p.add('pricing', 'demand', 'structure', "新しい企業の持分の値段: 'demand' か 'fundamental'")
    p.add('theta0', 0.3, 'behavior', '新しい投資家が、正味の財産のうち新しい持分に置きたい割合の基準')
    p.add('theta1', 5.0, 'behavior', '置きたい割合の、期待の利回りの差への反応')
    p.add('nu', 0.5, 'behavior', '値上がりの期待の、実際の値上がりへの寄り方(外挿)')
    p.add('lev', 1.0, 'policy', '新しい投資家が借りて買える割合(持分の値打ちに対する借金の上限)')
    p.add('r', 0.04, 'policy', '貸出と預金の金利')
    p.add('r_eq', 0.03, 'behavior', 'r̄ = r + r_eq(基本の値打ちの割引)')
    p.add('phi_w', 0.3, 'behavior', '賃金の、雇用の割合への反応(フィリップス)')
    p.add('u_star', 0.85, 'behavior', '目標の稼働率')
    p.add('politics', 'none', 'policy', "体制: 'none'(税なし)、'protect'(古い資本家の富の割合で新しい財に税)、'revolution'(protect + 革命)")
    p.add('tauX_max', 0.5, 'policy', '古い体制が新しい財にかける税の上限')
    p.add('rev_thr', 0.5, 'policy', '革命の閾値: 新しい財で働く人と失業者の割合(働ける人に対する)')
    p.add('expropriate', False, 'policy', '革命のとき、古い企業の持分を労働者に移すか')
    return p


SECT = ('w', 'ko', 'kn', 'fO', 'fX', 'bank', 'gov')
INST = ('deposits', 'loans', 'eqO', 'eqX')


class Upgrade:
    def __init__(self, **kw):
        P = self.p = params()
        P.set(**kw)
        self.L = Ledger(list(SECT), INST, real=('capO', 'capX'))
        L = self.L
        self.KO, self.KX = 900.0, 5.0
        self.cumX0 = self.cumX = P.cum0
        self.W, self.PO, self.PX = 1.0, (1 + P.m) * 1.0 * P.gO, (1 + P.m) * 1.0 * P.gX0
        self.S = 1.0                                 # 新しい企業の持分の数
        self.pX = P.kappa * self.PO * self.KX        # 初めは q = 1
        self.VO = 0.0
        self.g_e = 0.0
        self.div_s = 0.0
        self.tauX = 0.0
        self.revolution = None
        self.Yd = {'w': 700.0, 'ko': 200.0, 'kn': 5.0}
        L.revalue('deposits', 'w', 300.0); L.revalue('deposits', 'ko', 1000.0); L.revalue('deposits', 'kn', P.kn0)
        L.revalue('deposits', 'bank', -1300.0 - P.kn0)
        L.revalue('capO', 'fO', self.KO); L.revalue('capX', 'fX', self.KX)
        L.revalue('eqX', 'kn', self.pX * self.S); L.revalue('eqX', 'fX', -self.pX * self.S)
        L.close(); L.t = 0
        self.t = 0
        self.log = []

    def D(self, s):
        return self.L.stock[(s, 'deposits')]

    def step(self):
        P, L = self.p, self.L
        W = self.W
        gX = max(P.gX_min, P.gX0 * (self.cumX / self.cumX0) ** (-P.learn_b))
        sp = lambda z: np.logaddexp(0.0, 20.0 * z) / 20.0
        uO_, uX_ = getattr(self, 'uO_last', 0.85), getattr(self, 'uX_last', 0.85)   # 去年の稼働率で残業の割増を決める
        gOe, gXe = P.gO * (1 + P.omega * sp(uO_ - 1.0)), gX * (1 + P.omega * sp(uX_ - 1.0))
        PO = (1 + P.m) * W * gOe
        PX = (1 + P.m) * W * gXe
        PXc = PX * (1 + self.tauX)                   # 家計が払う新しい財の値段(税込み)
        # ---- 予算(去年の所得と預金から)
        B = {'w': P.c_w * max(self.Yd['w'], 0.0),
             'ko': P.c_k * max(self.Yd['ko'], 0.0) + P.c_m * max(self.D('ko'), 0.0),
             'kn': P.c_k * max(self.Yd['kn'], 0.0) + P.c_m * max(self.D('kn') + self.pX * self.S, 0.0)}
        C = sum(B.values())
        sX = P.aX * PXc ** (1 - P.sigma) / (P.aX * PXc ** (1 - P.sigma) + (1 - P.aX) * PO ** (1 - P.sigma))
        CX, CO = C * sX / PXc, C * (1 - sX) / PO     # 量
        # ---- 投資: 古い企業は稼働率で(自分の利益で賄う)、新しい企業は q で(持分を売って賄う)
        uO = (CO + 0.0) / max(self.KO, 1e-9)
        invO = self.KO * max(P.delta + P.phi_u * (uO - P.u_star), 0.0)
        q = self.pX * self.S / max(P.kappa * PO * self.KX, 1e-9)
        invX = self.KX * min(max(P.delta + P.phi_q * (q - 1.0), 0.0), P.delta + 0.5)   # 能力の伸びは年 50% まで
        IO, IX = PO * P.kappa * invO, PO * P.kappa * invX
        xO = CO + P.kappa * (invO + invX)
        xX = CX
        Ld = gOe * xO + gXe * xX
        self.uO_last, self.uX_last = xO / max(self.KO, 1e-9), xX / max(self.KX, 1e-9)
        E = min(Ld, P.N)
        # ---- 持分の値段(新しい企業)と、新しい投資家の買い
        profX = (PX - W * gXe) * xX
        self.div_s = 0.7 * self.div_s + 0.3 * profX
        p0 = self.pX
        if P.pricing == 'fundamental':
            V = max(self.div_s, 1e-6) / (P.r + P.r_eq)
            pX = max((V - IX) / self.S, 1e-9) if V > IX else V / (self.S + 1e-9)
        else:
            y = profX / max(p0 * self.S, 1e-9)
            x = 0.3 * np.tanh((y + self.g_e - P.r) / 0.3)
            th = min(P.theta0 * np.exp(P.theta1 * x), 0.9)   # 置きたい割合は 0.9 まで(1 を超えると値段が決まらない)
            Wkn0 = self.D('kn') + p0 * self.S
            if IX > 0.5 * th * max(Wkn0, 0.0):            # 売れる新しい持分は、望む保有の半分まで
                IX = 0.5 * th * max(Wkn0, 0.0)
                invX = IX / (PO * P.kappa)
                xO = CO + P.kappa * (invO + invX)
                Ld = gOe * xO + gXe * xX
                E = min(Ld, P.N)
                self.uO_last = xO / max(self.KO, 1e-9)
            # 望む値打ち θ × (預金 + 持分の値打ち) = 持っている持分の値打ち + 新しく買う分、を値段について解く: p S (1 − θ) = θ D − I
            pX = max((th * self.D('kn') - IX) / ((1 - th) * self.S), 1e-9)
        newS = IX / max(pX, 1e-12)
        # ---- 帳簿: 消費、投資、賃金、配当、税、持分
        rev_tax = self.tauX * PX * xX
        for s in ('w', 'ko', 'kn'):
            L.pay('consumption', s, 'fO', B[s] * (1 - sX))
            L.pay('consumption', s, 'fX', B[s] * sX * PX / PXc)
            L.pay('consumption tax', s, 'gov', B[s] * sX * (PXc - PX) / PXc)
        L.pay('investment goods', 'fX', 'fO', IX)       # 新しい企業は持分の売り上げで、古い財(資本)を買う
        L.trade('eqX', 'kn', 'fX', IX)                  # 新しい投資家が新しい持分を買う
        L.produce_real('capO', 'fO', -P.delta * self.KO + invO)
        L.produce_real('capX', 'fX', -P.delta * self.KX + invX)
        wagesO, wagesX = W * gOe * xO * E / Ld, W * gXe * xX * E / Ld
        L.pay('wages', 'fO', 'w', wagesO)
        L.pay('wages', 'fX', 'w', wagesX)
        profO = PO * xO - wagesO - IO                   # 古い企業の配当(自分の投資を引いたあと)
        divX = PX * xX - wagesX
        L.pay('dividends', 'fO', 'ko', profO)
        L.pay('dividends', 'fX', 'kn', divX)
        L.pay('transfers', 'gov', 'w', rev_tax)         # 税は労働者に配る(政府の収支は 0)
        intr = {s: P.r * self.D(s) for s in ('w', 'ko', 'kn')}    # 預金の利子(負なら借金の利子)
        for s, a in intr.items():
            L.pay('interest', 'bank', s, a)
        bank_profit = -sum(intr.values())
        L.pay('dividends', 'bank', 'ko', bank_profit)
        # 持分の値打ちを付け直す
        S1 = self.S + newS
        dVX = pX * S1 - (self.L.stock[('kn', 'eqX')])
        L.revalue('eqX', 'kn', dVX); L.revalue('eqX', 'fX', -dVX)
        VO = max(profO, 0.0) / (P.r + P.r_eq)
        dVO = VO - self.L.stock[('ko', 'eqO')]
        L.revalue('eqO', 'ko', dVO); L.revalue('eqO', 'fO', -dVO)
        # 借りて買う上限: 新しい投資家の借金が lev × 持分の値打ちを超えたら、超えた分を棒引き(銀行の損は古い資本家が負う)
        debt = max(-self.D('kn'), 0.0)
        over = max(debt - P.lev * pX * S1, 0.0)
        wo = 0.0
        if over > 0:
            L.pay('write-off', 'bank', 'kn', over)
            L.pay('bank loss', 'ko', 'bank', over)
            wo = over
        books = L.check(tol=1e-6 * max(1.0, C))['max']
        L.close()
        # ---- 所得(来年の予算)、賃金、学習、体制
        self.Yd = {'w': wagesO + wagesX + rev_tax + intr['w'], 'ko': profO + bank_profit + intr['ko'] - wo, 'kn': divX + intr['kn'] + wo}
        self.W = W * np.exp(P.phi_w * (Ld / P.N - 0.9))
        self.KO = (1 - P.delta) * self.KO + invO
        self.KX = (1 - P.delta) * self.KX + invX
        self.cumX += xX
        self.g_e += P.nu * (np.log(max(pX, 1e-12) / max(p0, 1e-12)) - self.g_e)
        self.pX, self.S = pX, S1
        wealth_o = max(VO + self.D('ko'), 0.0)
        wealth_all = wealth_o + max(pX * S1 + self.D('kn'), 0.0) + max(self.D('w'), 0.0) + 1e-9
        power = wealth_o / wealth_all
        newwork = (gXe * xX * E / Ld + (P.N - E)) / P.N         # 新しい財で働く人と失業者の割合
        if P.politics in ('protect', 'revolution') and self.revolution is None:
            self.tauX = P.tauX_max / (1 + np.exp(-(power - 0.5) / 0.05))
            if P.politics == 'revolution' and newwork > P.rev_thr:
                self.revolution = self.t
                self.tauX = 0.0
                if P.expropriate:
                    v = self.L.stock[('ko', 'eqO')]
                    L.revalue('eqO', 'ko', -v); L.revalue('eqO', 'w', v)
                    L.close()
        rec = dict(t=self.t, sX=sX, gX=gX, q=q, pX=pX, fund=max(self.div_s, 1e-6) / (P.r + P.r_eq) / max(S1, 1e-9), KO=self.KO, KX=self.KX,
                   E=E, u=1 - E / P.N, W=W, PO=PO, real_w=W / (PO ** (1 - sX) * PXc ** sX), tauX=self.tauX, power=power, newwork=newwork,
                   wo=wo, books=books, Y=C + IO + IX, invX=IX, debt_kn=debt)
        self.log.append(rec)
        self.t += 1
        return rec


def simulate(T=200, **kw):
    m = Upgrade(**kw)
    rec = [m.step() for _ in range(T)]
    return {k: np.array([r[k] for r in rec], float) for k in rec[0]}, m
