# im3.py: 統合模型 im の段 3(地域と輸出入、通貨と為替)
# 地域は、それぞれ段 2 までの一国の模型(im1.Im1)を部品として持つ。帳簿は通貨ごとに一つ。
# 段 3a: 共通の通貨(銀行は地域ごと、地域をまたぐ支払いは銀行の間の貸し = TARGET2 にあたる)。
# 段 3b: 別々の通貨と為替(固定と変動)。段 3c: 三つ以上の地域と、通貨の組み合わせ(通貨同盟と、その外の変動など)。
# 段 6: 資本の移動(家計が外の資産を持つ。地域ごとの窓口 ext が請求権を出し、行き先の地域の銀行の預金で持つ)と、
#       財政同盟(同じ通貨の地域の間で、失業に応じた給付をまとめて賄う)。
# 財の貿易はアーミントンの型: 地域ごとの財は互いに少し違うもので、買い手は二つの産地の財を、
# 価格と輸送の費用(氷山の型: 運ぶと 1/(1 + tc) しか届かない)と好みの重み w で混ぜて買う。
# 代わりの利きやすさ σ を財ごとに変える(必需品は大きく: 比較優位で貿易される。贅沢品は小さく: 種類の数で互いに買い合う)。
import numpy as np
from scipy.optimize import least_squares
from im_core import Ledger
from portable import lm_solve, LMResult
import im1
from im1 import softplus, softmin, INSTRUMENTS


class World:
    def __init__(self, regions=('A', 'B'), region_kw=None, common=None, sigma=(8.0, 2.0, 4.0), tc=0.1, home=0.5,
                 households='continuous', Ns=400, seeds=None, currency='common', regimes=None, e0=None, fx_tactic=1e-4,
                 capital=False, dest=None, union=None, kappa_q=0.5):
        """region_kw: 地域ごとのパラメータの上書き(dict の dict)。common: すべてに共通の上書き。
        sigma: (必需品、贅沢品、原料)の代わりの利きやすさ。tc: 輸送の費用(氷山)。home: 自地域の財への好みの重み(0〜1)。
        残りの重みは、ほかの地域に等しく分ける。home = 1 なら貿易なし。
        currency: 通貨の割り当て。'common'(すべて同じ通貨)、'fixed'・'float'(地域ごとに別々の通貨で、基準の通貨以外を
        すべて固定・変動)、または dict {地域: 通貨の名前}(例 {'A': 'x', 'B': 'x', 'C': 'y'})。
        regimes: dict {通貨の名前: 'fixed' か 'float'}。最初の地域の通貨が基準(その値は 1)。変動の通貨は、その通貨の圏全体の、
        圏の外との収支が毎年 0 になるように、価格と一緒に解く(資本の移動はない)。固定の通貨は値が変わらず、
        圏の外との不均衡は為替の窓口の持ち高として積み上がる。
        帳簿は通貨ごとに一つ。同じ通貨の地域どうしは銀行の間の貸し(TARGET2 にあたる)で、違う通貨の地域とは
        各地域の為替の窓口 fx@r を通して決済する。
        capital: 資本の移動(段 6)。家計は外の資産を持てる(どれだけ持ちたいかは地域の模型のパラメータ theta_F0、phi_K、phi_L)。
        地域 o の家計の外の資産は、窓口 ext@o の請求権で、その裏は行き先の地域 d = dest[o] の銀行の預金(同じ通貨なら ext@o 自身が
        d の銀行に預け、違う通貨なら d の帳簿の非居住者の部門 nr@d が預ける)。利子は d の政策金利。変動の通貨の圏は、
        貿易の収支 + 資本の流れ + 利子の流れ = 0 となるよう、為替を解く(資本が流れ込めば、貿易の赤字を続けられる)。
        外の資産の見込みの利回り = 行き先の金利 − kappa_q × ln(実質の為替)。実質の為替 = 行き先の必需品の価格(自分の通貨で)/
        自分の必需品の価格。外が割高なら、外の通貨が下がると見込む(購買力平価への戻りの見込み。Dornbusch 1976)。
        dest の既定は、二つの地域なら相手、三つ以上なら最初の地域(最初の地域は二番目)。
        union: 財政同盟(段 6)。dict(members=地域の名前の列、psi=移転の率)。同じ通貨の地域だけ。去年の失業者の数が同盟の平均より
        多い地域へ、psi × 賃金の平均 × (失業者の数 − 平均) を、少ない地域の政府から移す(和は 0。地域に偏った打撃への保険)。
        受け取った政府は、それを給付として家計に渡す。"""
        n = len(regions)
        self.names = list(regions)
        if currency == 'common':
            cmap = {r: 'c0' for r in regions}
            regimes = {'c0': 'fixed'}
        elif currency in ('fixed', 'float'):
            cmap = {r: r for r in regions}
            regimes = {r: currency for r in regions}
        else:
            cmap = dict(currency)
            regimes = dict(regimes or {})
        self.curs = []
        for r in regions:
            if cmap[r] not in self.curs:
                self.curs.append(cmap[r])
        self.cur = np.array([self.curs.index(cmap[r]) for r in regions])
        self.regime = [regimes.get(c, 'fixed') if k > 0 else 'anchor' for k, c in enumerate(self.curs)]
        self.floating = [k for k, g in enumerate(self.regime) if g == 'float']
        self.e = np.ones(len(self.curs)) if e0 is None else np.array(e0, float)
        self.fx_tactic = fx_tactic
        common = dict(common or {})
        region_kw = region_kw or {}
        seeds = seeds or {r: 0 for r in regions}
        multi = len(self.curs) > 1
        extra = ('fx',) if multi else ()
        self.capital = bool(capital)
        if dest is None:
            dest = {r: (regions[1] if i == 0 else regions[0]) for i, r in enumerate(regions)}
        self.dest = [self.names.index(dest[r]) for r in regions]
        if self.capital:
            extra = extra + (('ext', 'nr') if multi else ('ext',))
        self.union = union
        self.kappa_q = kappa_q
        # 通貨ごとの帳簿
        self.ledgers = []
        led_of = {}
        for k in range(len(self.curs)):
            group = [r for i, r in enumerate(regions) if self.cur[i] == k]
            sectors, bank_of = [], {}
            for r in group:
                for x in tuple(im1.SECTORS) + extra:
                    sectors.append(f'{x}@{r}')
                    bank_of[f'{x}@{r}'] = f'bank@{r}'
                if self.capital:
                    d = regions[self.dest[regions.index(r)]]
                    if cmap[d] == cmap[r]:
                        bank_of[f'ext@{r}'] = f'bank@{d}'      # 同じ通貨: 窓口は行き先の銀行に預ける
            real = [f'{x}@{r}' for r in group for x in im1.REAL]
            L = Ledger(sectors, INSTRUMENTS, real=real, bank_of=bank_of if len(group) > 1 else None)
            self.ledgers.append(L)
            for r in group:
                led_of[r] = L
        self.R = []
        for i, r in enumerate(regions):
            kw = dict(common, **region_kw.get(r, {}))
            self.R.append(im1.Im1(households=households, Ns=Ns, seed=seeds[r], ledger=led_of[r], region=r,
                                  extra_sectors=extra, **kw))
        for L in self.ledgers:
            L.close(); L.t = 0
        self.L = self.ledgers[0] if len(self.ledgers) == 1 else None
        self.sigma = np.array(sigma, float)
        self.tau_trade = np.where(np.eye(n) > 0, 1.0, 1.0 + tc)          # [買い手 s, 産地 r]
        self.w = np.where(np.eye(n) > 0, home, (1 - home) / max(n - 1, 1))  # 好みの重み [s, r]
        self.prices = np.array([e.prices for e in self.R])                  # [r, j]
        for s, e in enumerate(self.R):
            e.route = self._router(s)
        self.log = []
        self.t = 0
        self.hold = np.zeros(n)          # 違う通貨の行き先で、nr@d が持つ地域 o のための預金(行き先の通貨)
        self.KF = np.zeros(n)            # 今年の資本の流れ(出し手の通貨。地域 o の家計が外の資産を買う額)
        self.INT = np.zeros(n)           # 今年の利子(行き先の通貨。地域 o の家計の外の資産への利子)
        self.dF = [None] * n
        self.sh_del = None
        self.crisis_now = None

    def plan_capital(self, X):
        """今年の為替 X のもとで、家計が外の資産をどれだけ買うか(資産の市場は為替と同じ年の中で釣り合う)。"""
        for o, e in enumerate(self.R):
            d = self.dest[o]
            fx = X[o, d] / self.X_last[o, d]
            self.dF[o] = e.plan_foreign(self.r_out[o], fx)
            self.KF[o] = float(self.dF[o].sum())

    def same_cur(self, o, d):
        return self.cur[o] == self.cur[d]

    def holding(self, o):
        """地域 o の家計の外の資産の裏の預金(行き先の通貨)。"""
        d = self.dest[o]
        if self.same_cur(o, d):
            return self.R[o].L.stock[(f'ext@{self.names[o]}', 'deposits')]
        return self.hold[o]

    # ------------------------------------------------ 価格の指数と支出の割合
    def xrate(self, e=None):
        """X[s, r]: 産地 r の通貨一単位が、買い手 s の通貨でいくらか。e は通貨ごとの値(基準の通貨で測る)。"""
        e = self.e if e is None else e
        v = e[self.cur]
        return v[None, :] / v[:, None]

    def buy_prices(self, p, e=None):
        """p[r, j] は産地 r の財 j の売る価格(産地の通貨)。戻り値: 買う価格の指数 Pb[s, j](買い手の通貨)と、
        支出の割合 sh[s, r, j]。"""
        n = len(self.R)
        X = self.xrate(e)
        Pb = np.zeros((n, 3)); sh = np.zeros((n, n, 3))
        for j in range(3):
            sg = self.sigma[j]
            c = self.w * (p[None, :, j] * X * self.tau_trade) ** (1 - sg)          # [s, r]
            c = np.where(self.w > 0, c, 0.0)
            tot = c.sum(1)
            Pb[:, j] = tot ** (1 / (1 - sg))
            sh[:, :, j] = c / tot[:, None]
        return Pb, sh

    def _router(self, s):
        def route(flow, payer, good, amount):
            for r, e in enumerate(self.R):
                # 危機時の生産の年: 必需品は、実際に届いた割合(産地ごとに満たせた割合)で払う
                a = amount * (self.sh_del[s, r] if (good == 0 and self.sh_del is not None) else self.sh[s, r, good])
                to = e.firms[good]
                if to == payer or a == 0:
                    continue
                if self.cur[r] == self.cur[s]:
                    self.R[s].L.pay(flow if r == s else flow + ' (import)', payer, to, a)
                else:
                    # 違う通貨: 買い手は自国の通貨で窓口に払い、産地の窓口が産地の通貨で企業に払う(あとでまとめて)
                    self.R[s].L.pay('imports', payer, self.R[s].sec['fx'], a)
                    self.pending[r][good] += a / self.X[s, r]
        return route

    # ------------------------------------------------ 年の中の釣り合い
    def crisis_on(self):
        """どれかの地域で危機時の生産(段 7)の道を使うか。使うなら、すべての地域を同じ道で解く(産地の品不足は、輸入する国にも届く)。"""
        return any(e.crisis_on() for e in self.R)

    def market(self, p, e=None):
        """売る価格 p のもとで、各地域の買い手の消費、生産(輸出を含む)、費用、マークアップを求める。"""
        if self.crisis_on():
            return self.market_crisis(p, e)
        self.crisis_now = None
        n = len(self.R)
        Pb, sh = self.buy_prices(p, e)
        X = self.xrate(e)
        qc = np.zeros((n, 2))
        for s, e_ in enumerate(self.R):
            qc[s] = e_.demand(Pb[s], e_._inv, e_._B)[:2]          # 消費の量(必需品、贅沢品)
        exp_c = qc * Pb[:, :2]                                   # 消費の支出 [s, j]
        x = np.array([e_.demand(p[r], e_._inv, e_._B) for r, e_ in enumerate(self.R)])   # 初めの値(閉じた経済の生産)
        P = [e_.p for e_ in self.R]
        for _ in range(500):
            exp_I = np.array([Pb[s, 2] * (P[s].alpha @ x[s] + self.R[s].kap_new @ self.R[s]._inv) for s in range(n)])
            exp = np.column_stack([exp_c, exp_I])           # [s, j]
            xn = np.einsum('srj,sj,sr->rj', sh, exp, 1.0 / X) / p
            if np.max(np.abs(xn - x) / (np.abs(x) + 1e-12)) < 1e-15:
                x = xn
                break
            x = xn
        out = []
        for r, e_ in enumerate(self.R):
            Pr = e_.p
            u = x[r] / e_.K
            geff = e_.gamma_eff() * (1 + Pr.omega * softplus(u - 1.0, 20.0))
            cost = e_.W * geff + Pb[r, 2] * Pr.alpha
            if Pr.land:
                cost = cost + np.array([Pr.rho_T * p[r, 0], 0.0, 0.0])
            m = e_.m0 + Pr.zeta * np.tanh(u - Pr.u_star)
            out.append((u, geff, cost, m))
        return Pb, sh, qc, x, out

    def market_crisis(self, p, e=None):
        """危機時の生産の年の中の釣り合い(一国の Im1.evaluate_crisis を、地域と貿易に広げたもの)。
        産地 r: 必需品への需要 D_r(買い手の家計と政府の備蓄の買い、アーミントンの割合で産地に分けたもの)に対し、在庫の目標を足した量を作る。
        上限で満たせない注文は、まだ満たせる産地へ元の割合で回る(品不足のこぼれ。値で数える近似: 回った先の価格の違いは数えない)。
        価格の上限が効けば、変動費が上限に届くところまでしか作らない(最後の生産者・損失の分担なら、自国の生存の量までは作る)。
        満たせた割合 φ_r = 売れた量 / D_r。買い手 s: 必需品の束のうち届くのは f_s = Σ_r 割合_sr φ_r。足りない分は自国の備蓄から放出し、
        残りの不足は家計が負う(家計の中では、生存の量から先に配る)。政府の備蓄の買いは先に満たす(一国のときと同じ)。
        一国の世界では、Im1.evaluate_crisis と同じ数になる。"""
        n = len(self.R)
        Pb, sh = self.buy_prices(p, e)
        X = self.xrate(e)
        R = self.R
        P = [e_.p for e_ in R]
        qc = np.zeros((n, 2))
        for s, e_ in enumerate(R):
            qc[s] = e_.demand(Pb[s], e_._inv, e_._B)[:2]          # 望む消費の量(必需品、贅沢品。買い手の束の単位)
        capped = np.array([P[r].cap_N > 0 and p[r, 0] >= P[r].cap_N * (1 - 1e-9) for r in range(n)])
        gb = np.array([0.0 if capped[s] else getattr(R[s], '_gbuy', 0.0) for s in range(n)])   # 政府の備蓄の買い(束の単位)
        GIn = np.array([getattr(R[s], '_GI_nom', 0.0) for s in range(n)])                        # 公共事業(名目、買い手の通貨)
        dN = qc[:, 0]
        D = dN + gb
        # 買い手 s が産地 r に置く必需品の注文の割合 w[s, r](初めはアーミントンの割合)。上限で満たせない分は、
        # まだ満たせる産地へ、元の割合で回す(品不足のこぼれ。値で数える近似)。回し先がなくなるまで繰り返す
        w = sh[:, :, 0].copy()
        xN, fsales, Inv, phi = np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n)
        for _ in range(60):
            DR = np.array([sum(w[s, r] * Pb[s, 0] * D[s] / X[s, r] for s in range(n)) / p[r, 0] for r in range(n)])
            DRh = np.array([sum(w[s, r] * Pb[s, 0] * dN[s] / X[s, r] for s in range(n)) / p[r, 0] for r in range(n)])
            for r, e_ in enumerate(R):
                Pr = P[r]
                Inv[r] = (Pr.inv_target * DRh[r] if e_.InvN is None else e_.InvN) if Pr.inventory else 0.0
                plan = max(DR[r] + Pr.chi_inv * (Pr.inv_target * DR[r] - Inv[r]), 0.0) if Pr.inventory else DR[r]
                if capped[r]:
                    xs = e_.supply_at_cap(Pr.cap_N, Pb[r, 2])
                    xN[r] = min(xs, plan)
                    if Pr.crisis_rule in ('last', 'share'):
                        need = float(Pr.q0 * e_.subs_heads().sum())
                        xN[r] = max(xN[r], min(plan, max(need - Inv[r] - e_.S_G, 0.0)))
                else:
                    xN[r] = plan
                fsales[r] = min(DR[r], xN[r] + Inv[r])
                phi[r] = fsales[r] / DR[r] if DR[r] > 0 else 1.0
            openr = phi >= 1 - 1e-12
            moved = 0.0
            w_new = w.copy()
            for s in range(n):
                unmet = float(sum(w[s, r] * (1 - phi[r]) for r in range(n)))
                base = float(sum(sh[s, r, 0] for r in range(n) if openr[r]))
                if unmet <= 1e-15 or base <= 0:
                    continue
                for r in range(n):
                    w_new[s, r] = w[s, r] * phi[r] + (unmet * sh[s, r, 0] / base if openr[r] else 0.0)
                moved = max(moved, unmet)
            if moved <= 1e-15:
                break
            w = w_new
        f = np.array([sum(w[s, r] * phi[r] for r in range(n)) for s in range(n)])
        deliv = f * D
        rel = np.array([min(D[s] - deliv[s], R[s].S_G) if deliv[s] < D[s] else 0.0 for s in range(n)])
        hhN = deliv + rel - gb
        # 届いた必需品の、産地ごとの割合(払う先)
        self.sh_del_now = np.array([[w[s, r] * phi[r] / f[s] if f[s] > 0 else sh[s, r, 0] for r in range(n)] for s in range(n)])
        exp_c = np.column_stack([Pb[:, 0] * deliv, qc[:, 1] * Pb[:, 1]])   # 必需品は届いた分(政府の買いを含む)、贅沢品は望む分
        x = np.array([e_.demand(p[r], e_._inv, e_._B) for r, e_ in enumerate(R)])
        x[:, 0] = xN
        for _ in range(500):
            exp_I = np.array([Pb[s, 2] * (P[s].alpha @ x[s] + R[s].kap_new @ R[s]._inv) + GIn[s] for s in range(n)])
            exp = np.column_stack([exp_c, exp_I])
            xn = np.einsum('srj,sj,sr->rj', sh, exp, 1.0 / X) / p
            xn[:, 0] = xN                                                # 必需品は、作った量(売れた量ではない。差は在庫)
            if np.max(np.abs(xn - x) / (np.abs(x) + 1e-12)) < 1e-15:
                x = xn
                break
            x = xn
        out, cxs = [], []
        for r, e_ in enumerate(R):
            Pr = e_.p
            u = x[r] / e_.K
            geff = e_.gamma_eff() * (1 + Pr.omega * softplus(u - 1.0, 20.0))
            cost = e_.W * geff + Pb[r, 2] * Pr.alpha
            if Pr.land:
                cost = cost + np.array([Pr.rho_T * p[r, 0], 0.0, 0.0])
            m = e_.m0 + Pr.zeta * np.tanh(u - Pr.u_star)
            if Pr.inventory:
                m = m + np.array([Pr.zeta_inv * np.tanh(1.0 - (Inv[r] + xN[r] - fsales[r]) / max(Pr.inv_target * DR[r], 1e-12)), 0.0, 0.0])
            out.append((u, geff, cost, m))
            cxs.append(dict(dN=dN[r], D=D[r], gbuy=gb[r], xN=xN[r], firm_sales=fsales[r], rel=rel[r], hhN=hhN[r],
                            short=max(dN[r] - hhN[r], 0.0), capped=bool(capped[r]), GI=GIn[r] / Pb[r, 2], Inv=Inv[r], phi=phi[r], fill=f[r]))
        self.crisis_now = dict(cx=cxs, deliv=deliv, gb=gb, GIn=GIn)
        qc2 = np.column_stack([hhN, qc[:, 1]])
        return Pb, sh, qc2, x, out

    def flows(self, Pb, sh, qc, x):
        """支出の流れ [買い手 s, 産地 r, 財 j](買い手の通貨)。"""
        n = len(self.R)
        if getattr(self, 'crisis_now', None) is not None:
            # 危機時の生産の年: 必需品は届いた分(政府の備蓄の買いを含む)を、届いた割合で産地に分ける。投資財には公共事業を足す
            C = self.crisis_now
            exp_I = np.array([Pb[s, 2] * (self.R[s].p.alpha @ x[s] + self.R[s].kap_new @ self.R[s]._inv) + C['GIn'][s] for s in range(n)])
            f = np.zeros((n, n, 3))
            for s in range(n):
                for r in range(n):
                    f[s, r, 0] = self.sh_del_now[s, r] * Pb[s, 0] * C['deliv'][s]
                    f[s, r, 1] = sh[s, r, 1] * qc[s, 1] * Pb[s, 1]
                    f[s, r, 2] = sh[s, r, 2] * exp_I[s]
            return f
        exp_c = qc * Pb[:, :2]
        exp_I = np.array([Pb[s, 2] * (self.R[s].p.alpha @ x[s] + self.R[s].kap_new @ self.R[s]._inv) for s in range(n)])
        return np.einsum('srj,sj->srj', sh, np.column_stack([exp_c, exp_I]))

    def area_balance(self, k, f, X):
        """通貨 k の圏の、圏の外との収支(通貨 k で)= 圏の外への売上 − 圏の外からの買い。"""
        inn = self.cur == k
        ex = sum(f[s, r, :].sum() / X[s, r] for r in np.nonzero(inn)[0] for s in np.nonzero(~inn)[0])
        im = sum(f[s, r, :].sum() for s in np.nonzero(inn)[0] for r in np.nonzero(~inn)[0])
        spend = sum(f[s].sum() for s in np.nonzero(inn)[0])
        cap = 0.0
        if self.capital:
            for o in range(len(self.R)):
                d = self.dest[o]
                if inn[o] and not inn[d]:                 # 圏から出る資本と、入る利子
                    cap += -self.KF[o] + self.INT[o] * X[o, d]
                elif inn[d] and not inn[o]:               # 圏に入る資本と、出る利子
                    cap += self.KF[o] * X[d, o] - self.INT[o]
        return (ex - im + cap) / max(spend, 1e-9)

    def _unpack(self, z):
        n = len(self.R)
        p = np.exp(z[:3 * n].reshape(n, 3))
        e = self.e.copy()
        for i, k in enumerate(self.floating):
            e[k] = np.exp(z[3 * n + i])
        return p, e

    def residual(self, z):
        n = len(self.R)
        p, e = self._unpack(z)
        Pb, sh, qc, x, out = self.market(p, e)
        res = []
        for r in range(n):
            u, geff, cost, m = out[r]
            target = (1 + m) * cost
            cap = self.R[r].p.cap_N
            if cap > 0:
                target = np.array([min(target[0], cap), target[1], target[2]])   # 価格の上限(段 7)
            res += list(np.log(p[r]) - np.log(target))
        if self.floating:
            X = self.xrate(e)
            if self.capital:
                self.plan_capital(X)
            f = self.flows(Pb, sh, qc, x)
            for k in self.floating:
                res.append(10.0 * self.area_balance(k, f, X))
        tac = self.R[0].p.tactic
        res += list(np.sqrt(tac) * (z[:3 * n] - np.log(self.prices).ravel()))
        for i, k in enumerate(self.floating):
            res.append(np.sqrt(self.fx_tactic) * (z[3 * n + i] - np.log(self.e[k])))
        return np.array(res)

    # ------------------------------------------------ 一年
    def step(self):
        n = len(self.R)
        for e in self.R:
            e.begin_year()
        if self.capital:
            X0 = self.xrate()
            self.X_last = X0
            self.r_out = np.zeros(n)
            for o, e in enumerate(self.R):
                d = self.dest[o]
                r_d = self.R[d].r_policy if self.R[d].p.credit else 0.0
                q = X0[o, d] * self.R[d].PN_prev / e.PN_prev        # 実質の為替(去年の値)
                self.r_out[o] = r_d - self.kappa_q * np.log(max(q, 1e-12))
                self.INT[o] = r_d * max(self.holding(o), 0.0)
            self.plan_capital(X0)
        ub = np.zeros(n); uin = np.zeros(n)
        if self.union:
            mem = [self.names.index(m) for m in self.union['members']]
            assert len({self.cur[m] for m in mem}) == 1, '財政同盟は同じ通貨の地域だけ'
            if self.log:
                w_ref = float(np.mean([self.R[m].W for m in mem]))
                U = np.array([self.log[-1][m]['unemp'] for m in mem])
                Yl = np.array([self.log[-1][m]['Y'] for m in mem])
                net = self.union['psi'] * w_ref * (U - U.mean())      # 同盟の平均より失業の多い地域へ、差に応じて移す(和は 0)
                for i, m in enumerate(mem):
                    uin[m] = net[i]
                    ub[m] = max(net[i], 0.0)                          # 受け取った分は、その地域の給付として家計に渡す
                pos = uin > 0
                L = self.R[mem[0]].L
                for a in mem:
                    if uin[a] < 0:
                        for b in mem:
                            if uin[b] > 0:
                                L.pay('union transfer', f'gov@{self.names[a]}', f'gov@{self.names[b]}', -uin[a] * uin[b] / uin[pos].sum())
        z0 = np.concatenate([np.log(self.prices).ravel(), np.log(self.e[self.floating])])
        if self.R[0].p.solver == 'portable':
            sol = lm_solve(self.residual, z0, xtol=1e-13, ftol=0.0)
            # 価格の上限が効く地域: 必需品の価格を上限にそろえ、残りを解き直す(im1.py の solve_prices と同じ。上限の角を解から外す)
            fix = [3 * r for r, e in enumerate(self.R) if e.p.cap_N > 0 and np.exp(sol.x[3 * r]) >= e.p.cap_N * (1 - 1e-6)]
            if fix:
                nz, nf = len(sol.x), len(self.floating)
                free = [k for k in range(nz) if k not in fix]
                drop = set(fix) | {3 * n + nf + k for k in fix}

                def full(y):
                    z = sol.x.copy()
                    z[fix] = [np.log(self.R[k // 3].p.cap_N) for k in fix]
                    z[free] = y
                    return z

                def sub_res(y):
                    r_ = self.residual(full(y))
                    return r_[[k for k in range(len(r_)) if k not in drop]]
                sub = lm_solve(sub_res, sol.x[free], xtol=1e-13, ftol=0.0)
                zf = full(sub.x)
                sol = LMResult(zf, self.residual(zf), sol.nfev + sub.nfev, sol.nit + sub.nit)
        else:
            sol = least_squares(self.residual, z0, method='lm', xtol=1e-13, ftol=1e-13)
        p, self.e = self._unpack(sol.x)
        Pb, sh, qc, x, out = self.market(p)
        self.sh = sh
        C = getattr(self, 'crisis_now', None)
        self.sh_del = self.sh_del_now if C is not None else None
        for r, e in enumerate(self.R):
            e._cx = C['cx'][r] if C is not None else None                     # 危機時の生産の年: 地域ごとの年の中の結果を、finish_year に渡す
        self.X = self.xrate()
        self.pending = [np.zeros(3) for _ in range(n)]
        recs = []
        X = self.X
        if self.capital:
            self.plan_capital(X)
            for o, e in enumerate(self.R):
                d = self.dest[o]
                if self.same_cur(o, d):
                    continue
                # 家計の外の資産を、今年の為替で測り直してから売り買いする
                no = self.names[o]
                newV = self.hold[o] * X[o, d]
                g = newV - e.L.stock[(f'hh@{no}', 'foreign')]
                e.L.revalue('foreign', f'hh@{no}', g)
                e.L.revalue('foreign', f'ext@{no}', -g)
                if e.H.F.sum() > 0:
                    e.H.F = e.H.F * (newV / e.H.F.sum())
        for r, e in enumerate(self.R):
            e.xf = dict(union_ben=ub[r], union_in=uin[r])
            if self.capital:
                d = self.dest[r]
                e.xf.update(dF=self.dF[r], incF=self.INT[r] * X[r, d])
        if self.capital:
            for d in range(n):
                self.R[d].xf['int_nr'] = float(sum(self.INT[o] for o in range(n) if self.dest[o] == d))
        for r, e in enumerate(self.R):
            u, geff, cost, m = out[r]
            resid = float(np.linalg.norm(sol.fun[3 * r:3 * r + 3]))
            recs.append(e.finish_year(p[r], Pb[r], x[r], u, geff, cost, qc[r], resid))
        self.prices = p
        if len(self.curs) > 1:
            for r, e in enumerate(self.R):
                for j in range(3):
                    e.L.pay('exports', e.sec['fx'], e.firms[j], self.pending[r][j])
        if self.capital:
            for o, e in enumerate(self.R):
                d = self.dest[o]
                no, nd = self.names[o], self.names[d]
                Ld = self.R[d].L
                if self.same_cur(o, d):
                    Ld.pay('deposit interest (non-residents)', f'bank@{nd}', f'ext@{no}', self.INT[o])
                    continue
                Lo = e.L
                # 資本: 家計の買いで ext@o に入った預金を、窓口で行き先の通貨に替え、nr@d が預ける
                Lo.pay('capital outflow', f'ext@{no}', f'fx@{no}', self.KF[o])
                Ld.pay('capital inflow', f'fx@{nd}', f'nr@{nd}', self.KF[o] / X[o, d])
                self.hold[o] += self.KF[o] / X[o, d]
                # 利子: d の銀行が nr@d に払い、窓口を通して ext@o に届く(ext@o はそれを家計に払ってある)
                Ld.pay('deposit interest (non-residents)', f'bank@{nd}', f'nr@{nd}', self.INT[o])
                Ld.pay('income outflow', f'nr@{nd}', f'fx@{nd}', self.INT[o])
                Lo.pay('income inflow', f'fx@{no}', f'ext@{no}', self.INT[o] * X[o, d])
                # 値打ちの付け直し(為替): 家計の外の資産 = nr@d の預金 × 為替
                hh = f'hh@{no}'
                newV = self.hold[o] * X[o, d]
                g = newV - Lo.stock[(hh, 'foreign')]
                Lo.revalue('foreign', hh, g)
                Lo.revalue('foreign', f'ext@{no}', -g)
                H = e.H
                if H.F.sum() > 0:
                    H.F = H.F * (newV / H.F.sum())
        books = 0.0
        for L in self.ledgers:
            gb = max(abs(L.stock[(s, 'bonds')]) for s in L.sectors if s.startswith('gov@'))
            books = max(books, L.check(tol=1e-6 * max(1.0, gb))['max'])
        # 貿易の収支(産地の通貨)と、二地域の間の収支
        X = self.X
        f = self.flows(Pb, sh, qc, x)                       # [買い手 s, 産地 r, 財 j](買い手の通貨)
        e2 = self.e[1] / self.e[0] if len(self.curs) > 1 else 1.0
        for r, rec in enumerate(recs):
            exp_r = np.array([sum(f[s, r, j] / X[s, r] for s in range(n) if s != r) for j in range(3)])
            imports = float(f[r, :, :].sum() - f[r, r, :].sum())
            nm = self.names[r]
            L = self.R[r].L
            ib = L.stock[(f'bank@{nm}', 'interbank')] if 'interbank' in L.instruments else 0.0
            fxpos = -L.stock[(f'fx@{nm}', 'deposits')] if len(self.curs) > 1 else 0.0   # 窓口の外貨の持ち高(自国の通貨で、累積の輸出 − 輸入)
            bil = {f'bil_{self.names[q]}': float(f[q, r, :].sum() / X[q, r] - f[r, q, :].sum()) for q in range(n) if q != r}
            if self.capital:
                rec.update(F=float(self.R[r].H.F.sum()), F_Y=float(self.R[r].H.F.sum()) / rec['Y'], KF=self.KF[r], KF_Y=self.KF[r] / rec['Y'],
                           incF_Y=self.INT[r] * X[r, self.dest[r]] / rec['Y'],
                           F_gap=abs(self.R[r].H.F.sum() - L.stock[(f'hh@{nm}', 'foreign')]))
            rec.update(union_in=uin[r], union_ben=ub[r])
            rec.update(exports=float(exp_r.sum()), imports=imports, tb=float(exp_r.sum()) - imports,
                       tb_Y=(float(exp_r.sum()) - imports) / rec['Y'],
                       expN=float(exp_r[0]), expL=float(exp_r[1]), expI=float(exp_r[2]), ib=ib, fxpos=fxpos,
                       interbank=ib + fxpos, books=books, e=e2, e_cur=float(self.e[self.cur[r]]), **bil)
        for L in self.ledgers:
            L.close()
        self.log.append(recs)
        self.t += 1
        return recs


def simulate(T=300, events=(), **kw):
    """events: (年, 地域の名前 または None, dict) の列。None なら両方の地域のパラメータを変える。"""
    w = World(**kw)
    for t in range(T):
        for ts, reg, ch in events:
            if ts == t:
                for nm, e in zip(w.names, w.R):
                    if reg is None or reg == nm:
                        e.p.set(**ch)
        w.step()
    out = {}
    for i, nm in enumerate(w.names):
        recs = [rs[i] for rs in w.log]
        out[nm] = {k: np.array([r[k] for r in recs], dtype=float) for k in recs[0] if k != 'inv_limited'}
    return out, w
