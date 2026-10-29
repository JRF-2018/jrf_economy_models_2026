# im_firms.py: 部門の中の企業の集まり(段 5 の世界樹の案)。
# 各部門に N 社を置く。型は二つ: 規則的に研究する企業(型 0。見返りを見ず、売上の s_R を毎年研究に使う)と、
# 学習する企業(型 1。自分の研究は売上の s_L で、他社の最先端を真似る)。
# マクロの帳簿は部門の単位のまま(部門の利益・研究の賃金・投資は、企業の和)。企業の中の量は帳簿の外の内訳で、
# 部門の新しい生産能力の技術 = 占有率で重みをつけた企業の知識。
# 企業の知識 g(対数の向上)は、研究の当たり(パレート型)と、真似(最先端との差の割合 ι を毎年取り込む)で増える。
# 生産に乗った知識 ge は、投資の速さで g に追いつく。占有率は、ge が平均より高い(費用が低い)ほど増える(複製子の式)。
# 利益の率の記憶が退出の閾値を下回る企業は退出し、新しい企業が平均の知識で入る(型は、型ごとの利益で選ぶか、固定の混ぜ方)。
# 公の研究所(政府が研究の人を雇う)の当たりは、どの企業も真似られる(遅れなし)。知財の保護: 他社の知識は ip_lag 年前のものしか真似られない。
import numpy as np


class FirmPop:
    def __init__(self, P, rng, n_sec=3):
        self.P, self.rng = P, rng
        N = int(P.firms)
        self.N = N
        n_rule = int(round(P.rule_share0 * N))
        typ = np.array([0] * n_rule + [1] * (N - n_rule))
        self.typ = np.tile(typ, (n_sec, 1))
        self.g = np.zeros((n_sec, N))          # 知識(対数の向上。部門の基準の技術からの)
        self.ge = np.zeros((n_sec, N))         # 生産に乗った知識
        self.s = np.full((n_sec, N), 1.0 / N)  # 占有率
        self.ema = np.zeros((n_sec, N))        # 利益の率の記憶
        self.g_pub = np.zeros(n_sec)           # 公の研究所の知識
        self.hist = []                         # 最先端の知識の記録(知財の保護の遅れのため)
        self.G = self.front()
        self.exits = np.zeros(2)
        self.prize = np.zeros((n_sec, N))       # 国庫の案の報酬(去年の分。規則的な企業は研究に回す。段 14)
        # 段 15: 企業の質(悪い企業は知識が目減りし、貸倒れやすい)と、貸し手の型(関係融資 True / 市場型 False)
        self.bad = rng.random((n_sec, N)) < P.p_bad
        self.rel = rng.random((n_sec, N)) < (1.0 if P.lender == 'relation' else 0.0 if P.lender == 'market' else P.rel_share)
        self.credit = np.ones((n_sec, N))
        self.loss_pending = np.zeros(n_sec)     # 次の年に銀行が損として書く、企業の貸倒れ
        self.r_norm = None
        self.defaults = 0
        self.cbar = np.ones(n_sec)              # 部門の、占有率で重みをつけた信用の取り分(投資の資金に効く)
        self.n_spin = 0
        self.last = {}

    def front(self):
        return (self.s * self.g).sum(1)        # 部門の新しい生産能力の技術 = 占有率の重みの知識

    def budgets(self, rev):
        """rev[j]: 部門 j の売上。戻り値: 企業ごとの研究の額 [j, k]。"""
        P = self.P
        rate = np.where(self.typ == 0, P.s_R, P.s_L)
        return rate * self.s * rev[:, None] + (self.prize if P.prize_reinvest else 0.0)

    def _hits(self, L):
        P = self.P
        lam = np.maximum(L / P.trial_cost * P.p_hit, 0.0)
        hits = self.rng.poisson(lam)
        gain = np.zeros_like(L)
        for idx in zip(*np.nonzero(hits)):
            g = P.g0 * (1 - self.rng.random(hits[idx])) ** (-1 / P.alpha_R)
            gain[idx] = float(np.minimum(g, P.g_cap).sum())
        return gain

    def update(self, res_k, W, prof_j, wage_j, capval_j, L_pub, loans=None):
        """一年の更新。res_k: 企業ごとの研究の額、prof_j: 部門の利益(利子のあと、研究の前)、wage_j: 部門の生産の賃金、
        capval_j: 部門の資本の値打ち、L_pub: 公の研究所の研究の人(部門ごとに等分)。戻り値: 部門の技術の変化 dG[j]。"""
        P = self.P
        n_sec, N = self.g.shape
        gain = self._hits(res_k / W)
        self.g += gain
        if L_pub > 0:
            self.g_pub += self._hits(np.full(n_sec, L_pub / n_sec))
        # 真似: 他社の最先端(ip_lag 年前)と公の知識
        self.hist.append(self.g.max(1).copy())
        lag = int(P.ip_lag)
        seen = self.hist[-1 - lag] if len(self.hist) > lag else np.zeros(n_sec)
        self.hist = self.hist[-(lag + 2):]
        target = np.maximum(seen, self.g_pub)[:, None]
        iota = np.where(self.typ == 0, P.iota_R, P.iota_L)
        self.g += iota * np.maximum(target - self.g, 0.0)
        # 生産に乗る
        self.ge += P.embody * (self.g - self.ge)
        # 企業の利益: 部門の利益を占有率で分け、費用の差(乗った知識の、平均からのずれ)を足し、研究の額を引く
        Ge = (self.s * self.ge).sum(1, keepdims=True)
        adv = self.s * wage_j[:, None] * (1 - np.exp(-(self.ge - Ge)))
        adv -= self.s * adv.sum(1, keepdims=True)              # 和を 0 に(部門の利益の和は変わらない)
        pi = self.s * prof_j[:, None] + adv - res_k
        # 国庫の案(段 14): 部門の利益への税を、その年の研究の当たりに比例して配る(部門の中の移転。部門の利益の和は変わらない)
        self.prize = np.zeros_like(pi)
        if P.t_prize > 0:
            pool = P.t_prize * np.maximum(prof_j, 0.0)
            gs = gain.sum(1, keepdims=True)
            self.prize = np.where(gs > 0, pool[:, None] * gain / np.maximum(gs, 1e-300), 0.0)
            pi = pi - P.t_prize * self.s * np.maximum(prof_j, 0.0)[:, None] + self.prize
        rate = pi / np.maximum(self.s * capval_j[:, None], 1e-12)
        self.ema = P.omega_x * self.ema + (1 - P.omega_x) * rate
        # 占有率(複製子の式)
        if P.lender != 'none':
            # 段 15: 貸し手が決める信用の取り分 c(0〜1)。関係融資は質を知り、良い企業には 1、悪い企業にも zombie だけ貸し続ける。
            # 市場型は雑音のある信号で決め、部門の利益の率が平常を下回ると、貸す基準を上げる(信用の収縮)
            rr = prof_j / np.maximum(capval_j, 1e-12)
            self.r_norm = rr.copy() if self.r_norm is None else 0.9 * self.r_norm + 0.1 * rr
            thr = P.thr0 + P.crunch * np.maximum(0.0, (self.r_norm - rr) / 0.05)
            sig = np.where(self.bad, -1.0, 1.0) + P.sig_noise * self.rng.standard_normal(self.bad.shape)
            c_mkt = 1 / (1 + np.exp(-(sig - thr[:, None]) / 0.25))
            c_rel = np.where(self.bad, P.zombie, 1.0)
            self.credit = np.where(self.rel, c_rel, c_mkt) if P.lender != 'blind' else np.ones_like(c_mkt)
            self.cbar = (self.s * self.credit).sum(1)
            cbar = (self.s * self.credit).sum(1, keepdims=True)
            self.s = self.s * np.exp(P.beta_m * (self.ge - Ge) + P.beta_c * (self.credit - cbar))
            # 悪い企業: 知識の目減りと、借りている分に応じた貸倒れ(倒産して退出し、損は次の年に銀行が書く)
            self.g = self.g - P.bad_drift * self.bad
            # 関係融資の銀行は、悪い企業の借金を借り換えで延ばし(追い貸し)、倒産を zombie の割合だけ先送りする
            hz = np.where(self.rel & (P.lender != 'blind'), P.d_bad * (1 - P.zombie), P.d_bad) * self.credit
            dflt = self.bad & (self.rng.random(self.bad.shape) < hz)
            if loans is not None:
                share = self.s * self.credit / np.maximum((self.s * self.credit).sum(1, keepdims=True), 1e-12)
                self.loss_pending = P.lgd * loans * (share * dflt).sum(1)
            self.ema = np.where(dflt, -1.0, self.ema)               # 倒産した企業は、下の退出で入れ替わる
            self.defaults += int(dflt.sum())
        else:
            self.s = self.s * np.exp(P.beta_m * (self.ge - Ge))
        self.s /= self.s.sum(1, keepdims=True)
        # 退出と参入
        typ_profit = np.array([[(pi[j][self.typ[j] == t].sum() / max(self.s[j][self.typ[j] == t].sum(), 1e-12))
                                if (self.typ[j] == t).any() else np.nan for t in (0, 1)] for j in range(n_sec)])
        # 作り手の流出(段 14): 毎年、企業ごとに確率 p_spin で研究者が独立し、知識を持って新しい企業を起こす(占有率の最も小さい企業と入れ替わる)
        if P.p_spin > 0:
            for j in range(n_sec):
                for k in np.nonzero(self.rng.random(N) < P.p_spin)[0]:
                    m = int(np.argmin(np.where(np.arange(N) == k, np.inf, self.s[j])))
                    self.exits[self.typ[j, m]] += 1
                    self.g[j, m] = self.g[j, k]                       # 知識は持ち出せる(元の企業の知識も減らない)
                    self.ge[j, m] = float((self.s[j] * self.ge[j]).sum())   # 生産に乗せるのは、これから
                    self.typ[j, m] = self.typ[j, k]
                    self.s[j, m] = P.s_entry
                    self.ema[j, m] = 0.0
                    self.prize[j, m] = 0.0
                    self.n_spin += 1
            self.s /= self.s.sum(1, keepdims=True)
        out = (self.ema < P.r_exit) | (self.s < P.s_min)
        for j, k in zip(*np.nonzero(out)):
            self.exits[self.typ[j, k]] += 1
            if P.entry == 'select':
                # 型ごとの、占有率あたりの利益が高い型を選びやすい(と、少しの突然変異)
                tp = typ_profit[j]
                if np.all(np.isfinite(tp)):
                    p1 = 1 / (1 + np.exp(-(tp[1] - tp[0]) / max(abs(tp).mean(), 1e-12) * 3))
                else:
                    p1 = 1.0 if np.isfinite(tp[1]) else 0.0
                p1 = (1 - P.mut) * p1 + P.mut * 0.5
                self.typ[j, k] = int(self.rng.random() < p1)
            else:
                self.typ[j, k] = int(self.rng.random() >= P.rule_share0)
            mean_ge = float((self.s[j] * self.ge[j]).sum())
            mean_g = float((self.s[j] * self.g[j]).sum())
            self.g[j, k], self.ge[j, k] = mean_g, mean_ge
            self.s[j, k] = P.s_entry
            self.ema[j, k] = 0.0
            self.bad[j, k] = self.rng.random() < P.p_bad
            self.rel[j, k] = self.rng.random() < (1.0 if P.lender == 'relation' else 0.0 if P.lender == 'market' else P.rel_share)
        self.s /= self.s.sum(1, keepdims=True)
        G = self.front()
        dG = G - self.G
        self.G = G
        rule_s = (self.s * (self.typ == 0)).sum(1)
        self.last = dict(rule_share=float(rule_s.mean()), n_rule=float((self.typ == 0).sum() / n_sec),
                         hhi=float((self.s ** 2).sum(1).mean()), G_firm=float(G.mean()), g_pub=float(self.g_pub.mean()),
                         rate_rule=float(np.nanmean(np.where(self.typ == 0, rate, np.nan))) if (self.typ == 0).any() else np.nan,
                         rate_learn=float(np.nanmean(np.where(self.typ == 1, rate, np.nan))) if (self.typ == 1).any() else np.nan,
                         exits_rule=float(self.exits[0]), exits_learn=float(self.exits[1]),
                         gain_rule=float(gain[self.typ == 0].sum()), gain_learn=float(gain[self.typ == 1].sum()),
                         n_spin=float(self.n_spin), prize=float(self.prize.sum()),
                         bad_share=float((self.s * self.bad).sum(1).mean()), credit_good=float(np.mean((self.s * self.credit * ~self.bad).sum(1) / np.maximum((self.s * ~self.bad).sum(1), 1e-12))),
                         loss=float(self.loss_pending.sum()), defaults=float(self.defaults))
        return dG
