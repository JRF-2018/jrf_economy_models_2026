# me_c: micro_economy の「現代化」(v0.1、2026-10-27)
#
# 元の micro_economy_*.py(ノート 1)の骨格を残し、ノート 1 の「残る問い」に沿って作り直した、理解しやすさを優先した模型。
#   * 財は三つ: 必需品 N、贅沢品 L、原料 I(原料は原料にも要る。レオンチェフの投入)。
#   * 企業は「社の数」ではなく、分野ごとの生産能力 K_j(連続量)で表す。参入・退出は、投資と減耗による能力の増減になる。
#     新しい能力ほど一製品あたりの労働が少ない(機械の高度化。元の 0.95〜0.99 倍の連続版)。
#   * 人の数・働く人数・贅沢品の需要は、すべて連続でなめらか(段差がない)。
#   * 三つの価格は、毎期「均衡の残差」の最小化で決める(作者の「価格を最適化で決める」を、ノート 1 第 8 章の
#     教訓どおり、「いくらでも大きく」の項を入れず、0 になれば均衡という条件の二乗和で書く)。
#     条件は 3 本: 三つの分野の「価格 = (1 + マークアップ) × 単位費用」。マークアップは稼働率(需要の強さ)で上下する。
#     解の選び方の戦術として「前の期の価格に近い解」を小さな重みで加える。
#   * 賃金は粘着的で、労働の需給のずれに応じて期ごとに動く(フィリップス曲線)。これが名目の水準の錨になる
#     (ノート 1 第 1.3 節: 名目の水準は、価格の最適化の中では決まらない)。
#   * 家計の消費の支出は、前の期の所得と手持ちのお金で決まる(期の中の循環をなくし、解を安定させる)。
#   * 資本市場はない。企業の利潤は資本家にそのまま渡り、投資は資本家の手元のお金で賄う。お金は政府の負債(社会的負債)だけ。
#   * 政府は、失業者に必需品の生存量を給付し、賃金と配当に課税する。税率は、赤字の対所得比を目標(既定は 0 = 均衡予算)に動く。
#
# 将来の版で足す候補: 輸出入(二地域)、危機時の生産(在庫、最後の生産者)、家計の分布、信用(ノート 4)。
import numpy as np
from scipy.optimize import least_squares

SECT = ('N', 'L', 'I')


def softplus(x, k=1.0):
    return np.logaddexp(0.0, k * x) / k


def softmin(a, b, k=50.0):
    """なめらかな min(段差をなくすため)。min(a, b) = a − max(a − b, 0) の max を softplus でなめらかにする。"""
    return a - softplus(a - b, k / max(abs(b), 1e-9))


class Params:
    def __init__(self, **kw):
        # 人
        self.pop_w = 1000.0         # 労働者(働く人と働かない人の合計)
        self.n_cap = 5.0            # 資本家
        self.q0 = 1.0               # 一人あたり必需品の生存量
        # 技術(初期値は元のモデルの各分野の最初の企業に合わせる)
        self.gamma0 = np.array([0.1, 0.3, 0.3])    # 一製品あたり労働
        self.alpha = np.array([0.3, 0.5, 0.3])     # 一製品あたり原料
        self.kappa = np.array([1.0, 1.0, 1.0])     # 生産能力一単位に要る原料(固定資産)
        self.delta = 0.1                           # 能力の減耗(元の固定資産の寿命 10 期)
        self.g_gamma = 0.01                        # 新しい能力の一製品あたり労働が、毎期下がる率(機械の高度化)
        # 価格と稼働
        self.m0 = np.array([0.3, 0.3, 0.3])        # 基準のマークアップ(初期値)
        self.zeta = 0.5                            # 稼働率がマークアップに効く強さ(tanh で上下に頭打ち)
        self.u_star = 0.85                         # 目標の稼働率
        self.omega = 1.0                           # 稼働率が 1 を超えたときの割増(残業)の強さ
        self.eta = 0.05                            # 競争で基準のマークアップが利潤率の差に寄る速さ
        self.r_bar = 0.06                          # 標準の利潤率(元の「市場利子率」。経済の中では決まらない: ノート 1 第 8.3 節)
        # 投資(連続の参入・退出)
        self.phi_r = 0.5                           # 利潤率の差への反応
        self.phi_u = 0.3                           # 稼働率の差への反応
        # 賃金(フィリップス曲線)
        self.phi_w = 0.1                           # 労働の需給のずれへの反応
        self.chi = 0.2                             # 前の期の必需品の値上がりへの連動
        # 家計
        self.c_y = 0.9                             # 労働者の可処分所得からの消費性向
        self.c_m = 0.2                             # 労働者の手持ちのお金からの消費性向
        self.c_yc = 0.5                            # 資本家の可処分所得からの消費性向
        self.c_mc = 0.05                           # 資本家の手持ちのお金からの消費性向
        self.b_N = 0.3                             # 生存量を超えた支出のうち、必需品に回る割合(残りは贅沢品)
        self.beta_s = 3.0                          # 労働の供給の、実質賃金への反応
        self.w_res = 2.0                           # 留保の実質賃金(必需品の単位)
        # 政府
        self.tau0 = 0.2
        self.deficit_target = 0.0                  # 赤字 / 名目の最終需要 の目標(0 なら均衡予算)
        self.k_tau = 0.2                           # 税率の調整の速さ
        self.benefit = 1.0                         # 失業者への給付(必需品の生存量の何倍か)
        # 解き方
        self.tactic = 1e-4                         # 前の期の価格に近い解を選ぶ重み
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


class Economy:
    def __init__(self, p=None, **kw):
        self.p = p or Params(**kw)
        P = self.p
        self.t = 0
        self.K = np.array([1250.0, 400.0, 1100.0])          # 生産能力
        self.gamma = P.gamma0.copy()                        # 能力の平均の一製品あたり労働
        self.gamma_new = P.gamma0.copy()                    # 新しい能力の一製品あたり労働
        self.m0 = P.m0.copy()
        self.W = 10.0
        self.prices = np.array([3.0, 8.0, 6.0])             # P_N, P_L, P_I
        self.PN_prev = self.prices[0]
        self.M_w, self.M_c = 3000.0, 1000.0                 # 手持ちのお金(労働者、資本家)
        self.G = self.M_w + self.M_c                        # 政府の負債(= 家計のお金。銀行はない)
        self.tau = P.tau0
        self.Yd_w, self.Yd_c = 4500.0, 500.0                # 前の期の可処分所得
        self.rho_prev = np.full(3, P.r_bar)
        self.u_prev = np.full(3, P.u_star)
        self.profit_prev = 0.0
        self.log = []

    # ---------------------------------------------------------------- 一期のなかの計算
    def plan_investment(self):
        """投資は、前の期の利潤率と稼働率で決める(期のはじめに決めるので、今期の価格の計算とは切り離される)。"""
        P = self.p
        g = P.delta + P.phi_r * (self.rho_prev - P.r_bar) + P.phi_u * (self.u_prev - P.u_star)
        inv = self.K * softplus(g, 50.0)                      # 投資は負にならない(退出は減耗で起きる)
        # 資本市場がないので、投資は、前の期の利潤(内部留保)と、資本家の手持ちのお金の半分までで賄う
        cost = self.prices[2] * P.kappa @ inv
        avail = max(self.profit_prev, 0.0) + max(self.M_c, 0.0) * 0.5
        self.inv_limited = bool(cost > avail and cost > 0)
        if self.inv_limited:
            inv = inv * avail / cost
        return inv

    def budgets(self):
        """消費の支出(名目)。前の期の可処分所得と、手持ちのお金で決まる。"""
        P = self.p
        B_w = P.c_y * self.Yd_w + P.c_m * max(self.M_w, 0.0)
        B_c = P.c_yc * max(self.Yd_c, 0.0) + P.c_mc * max(self.M_c, 0.0)
        return max(B_w, 0.0), max(B_c, 0.0)

    def demand(self, price, inv, B):
        """価格のもとでの需要(なめらか)。生存量までは必需品、それを超えた支出を必需品と贅沢品に分ける(線形支出体系)。"""
        P = self.p
        PN, PL, PI = price
        xN = xL = 0.0
        for Bk, heads in zip(B, (P.pop_w, P.n_cap)):
            base = PN * heads * P.q0
            disc = softplus(Bk - base, 10.0 / max(base, 1e-9))     # 生存量を超えた支出(なめらか)
            xN += (Bk - disc) / PN + P.b_N * disc / PN
            xL += (1 - P.b_N) * disc / PL
        xI = (P.alpha[0] * xN + P.alpha[1] * xL + P.kappa @ inv) / (1 - P.alpha[2])
        return np.array([xN, xL, xI])

    def evaluate(self, price, inv, B):
        P = self.p
        x = self.demand(price, inv, B)
        u = x / self.K
        geff = self.gamma * (1 + P.omega * softplus(u - 1.0, 20.0))          # 能力を超えると、残業で一製品あたり労働が増える
        cost = self.W * geff + price[2] * P.alpha
        m = self.m0 + P.zeta * np.tanh(u - P.u_star)
        return x, u, geff, cost, m

    def residual(self, z, inv, B):
        P = self.p
        price = np.exp(z)
        x, u, geff, cost, m = self.evaluate(price, inv, B)
        res = list(z - np.log((1 + m) * cost))                             # 価格 = (1 + マークアップ) × 単位費用
        res += list(np.sqrt(P.tactic) * (z - np.log(self.prices)))         # 戦術: 前の期の価格に近い解
        return np.array(res)

    def solve_prices(self, inv, B):
        sol = least_squares(self.residual, np.log(self.prices), args=(inv, B), method='lm', xtol=1e-13, ftol=1e-13)
        return np.exp(sol.x), sol

    # ---------------------------------------------------------------- 一期を進める
    def step(self):
        P = self.p
        inv = self.plan_investment()
        B = self.budgets()
        price, sol = self.solve_prices(inv, B)
        x, u, geff, cost, m = self.evaluate(price, inv, B)
        PN, PL, PI = price
        W = self.W
        # 労働
        S = P.pop_w / (1.0 + np.exp(-P.beta_s * (np.log(W / PN) - np.log(P.w_res))))
        D = float(geff @ x)
        E = softmin(D, S)
        unemp = P.pop_w - E
        # お金の動き
        wage_bill = W * D                                   # 残業も含めた賃金の総額
        ben = P.benefit * P.q0 * PN * unemp
        profit = (price - cost) * x
        div = profit.sum() - PI * P.kappa @ inv            # 投資は利潤(と資本家の手持ち)で賄う
        tax_w, tax_c = self.tau * wage_bill, self.tau * max(div, 0.0)
        spend = PN * x[0] + PL * x[1]                       # = B の合計(作り方から)
        self.Yd_w = wage_bill - tax_w + ben
        self.Yd_c = div - tax_c
        self.M_w += self.Yd_w - B[0]
        self.M_c += self.Yd_c - B[1]
        deficit = ben - tax_w - tax_c
        self.G += deficit
        # 利潤率(能力の再調達価格に対する、減耗を引いた純の利潤率)と、能力の更新(連続の参入・退出)
        rho = profit / (PI * P.kappa * self.K) - P.delta
        K_old = self.K.copy()
        self.K = (1 - P.delta) * self.K + inv
        self.gamma_new *= (1 - P.g_gamma)
        self.gamma = ((1 - P.delta) * K_old * self.gamma + inv * self.gamma_new) / self.K
        # 競争: 基準のマークアップは、利潤率が標準より高ければ下がる
        self.m0 = np.clip(self.m0 - P.eta * (rho - P.r_bar), 0.01, 2.0)
        # 賃金(フィリップス曲線)と財政のルール
        infl = np.log(PN / self.PN_prev)
        self.W = W * np.exp(P.phi_w * (D / S - 1.0) + P.chi * infl)
        Y = spend + PI * P.kappa @ inv                      # 名目の最終需要
        self.tau = float(np.clip(self.tau + P.k_tau * (deficit / Y - P.deficit_target), 0.0, 0.6))
        self.rho_prev, self.u_prev = rho, u
        self.profit_prev = float(profit.sum())
        self.PN_prev = PN
        self.prices = price
        rec = dict(t=self.t, W=W, PN=PN, PL=PL, PI=PI, real_wage=W / PN, S=S, D=D, E=E, unemp=unemp,
                   invol=S - E, nonpart=P.pop_w - S,
                   xN=x[0], xL=x[1], xI=x[2], uN=u[0], uL=u[1], uI=u[2], KN=K_old[0], KL=K_old[1], KI=K_old[2],
                   rhoN=rho[0], rhoL=rho[1], rhoI=rho[2], mN=m[0], mL=m[1], mI=m[2],
                   gN=self.gamma[0], gL=self.gamma[1], gI=self.gamma[2], div=div, tau=self.tau, G=self.G,
                   M_w=self.M_w, M_c=self.M_c, deficit=deficit, Y=Y, spend=spend, Btot=B[0] + B[1],
                   resid=float(np.linalg.norm(sol.fun[:3])), inv_limited=self.inv_limited, acct=self.M_w + self.M_c - self.G)
        self.log.append(rec)
        self.t += 1
        return rec


def simulate(T=200, shocks=(), **kw):
    """shocks: (期, dict) の列。その期のはじめに Params の値を変える。"""
    e = Economy(**kw)
    for t in range(T):
        for (ts, ch) in shocks:
            if ts == t:
                for k, v in ch.items():
                    setattr(e.p, k, v)
        e.step()
    return {k: np.array([r[k] for r in e.log]) for k in e.log[0]}, e
