# %%CELL model
"""v6a.py -- 資産市場つきの v6(信用循環モデルの部門 + 個人の家計 + 土地と知財の市場)。

一期 = 一年。物価は 1 に固定。
家計(N 人): m_i(銀行との残高。預金 − 借金)、land_i(土地)、ip_i(知財)、eq_i(企業と銀行の持分。固定)。
企業: 資本 K、借入 L_f。v6 と同じく、株式的な富 E、損耗の増幅 κ、投資の規則、担保の上限 λ(E + K)。
銀行: 自己資本 0。利益(損失)はすべて持分に応じて家計に配る。だから Σ m_i = L_f + B がいつも成り立つ。
政府: 支出 G、失業者への給付、税、国債 B(銀行が持つ)。
"""
import numpy as np

DEFAULT = dict(
    N=1000, T=150, burn=150, seed=0,
    a=1.0,          # 一人あたりの生産(雇用 = Y / a)
    v=2.5,          # 資本係数: 生産能力 Y_p = u_max K / v
    u_max=1.1,
    omega=0.6,      # 賃金分配率
    rho_T=0.05,     # 地代: 所得 Y のこの割合を、土地の持ち主に払う
    phi_R=0.05,     # 知財の使用料: 操業利潤のこの割合を、知財の持ち主に払う
    d0=0.05,        # 平時の損耗率
    kappa=3.0,      # E が縮むときの損耗の増幅
    f_R=0.5,        # 純投資(I − δ0 K)のうち、内部留保で賄う割合。利潤のうち、それを除いた残りを配当にする
    g0=-0.02, g1=0.75, g2=0.1, g_q=0.0, q_ref=0.9,   # 投資の規則(v6 と同じ形。g2 は稼働率 u − 0.85 への反応)
    prem=0.02,      # E を評価するときのリスクプレミアム
    lam=0.6,        # 企業の担保の上限 λ(E + K)
    T_adj=2.0,      # 上限を超えたときの返済の速さ(年)
    tau_e=2.0,      # 期待利潤の適応の時定数
    default_share=0.5,
    r_cb=0.03, spread=0.01,
    gG=0.15,        # 政府支出 G = gG × 生産能力 / u_max
    tau=0.25,       # 税率(賃金・地代・使用料・配当・預金利子に)
    fiscal=0.05,    # 財政ルール: 税率を (B/Y − b_target) に比例して上げる強さ(v6 と同じ形)
    b_target=0.6,
    u_b=0.3,        # 失業者への給付(一人)
    n_c=0.3,        # 必需品(一人、実物の単位)
    c_y=0.8,        # 可処分所得の余りからの贅沢品の消費性向
    c_m=0.05,       # 預金からの贅沢品の消費性向
    c_h=0.005,      # 土地の値打ち(借金を引いた分)からの贅沢品の消費性向(資産効果)
    sigma_R=0.1,    # 贅沢品のうち、自主研究にあてる割合
    eta=1.0,        # 自主研究一単位が生む知財
    delta_IP=0.2,   # 知財の陳腐化の率
    T_land=1000.0,  # 土地の総量
    theta0=0.15, theta1=3.0,   # 土地に置きたい財産の割合 θ = θ0 exp(θ1 x)、x = 地代の利回り + 値上がりの期待 − 貸出金利(±0.2 で頭打ち)
    nu=0.2,         # 値上がりの期待の適応の速さ(外挿)
    chi=0.2,        # 望む土地へ、一期に近づく割合
    turnover=0.025, # 毎年、この割合の人が去り、財産のない若い人と入れ替わる(世代の入れ替わり)
    inherit='random',  # 去った人の財産の行き先: 'random'(くじで他の一人)、'equal'(全員に等しく)
    h_y=10.0,       # 土地の望む量に数える「稼ぐ力」(労働所得の何年分か。住宅ローンで買う働き手の需要)
    lam_T=0.6,      # 土地の担保の掛け目
    mu_u=0.5,       # 無担保で借りられる上限(平均の賃金の何年分か)
    shocks=(),      # [(年, {パラメータ: 値}, かける年数)]
)

REC = ['tau', 'u', 'Y', 'C', 'I', 'G', 'K', 'L_f', 'B', 'E', 'q', 'delta', 'p_T', 'HL', 'D', 'theta', 'gT_e', 'emp',
       'wo_f', 'wo_h', 'lux', 'res', 'IP', 'Pi', 'div_f', 'pi_b', 'gini_nw', 'top10_nw', 'n_over', 'resid', 'land_resid', 'nw_mean']


def _softmin(a, b, w):
    return -w * np.logaddexp(-a / w, -b / w)


def _sp(x, w):
    return w * np.logaddexp(0.0, x / w)


def gini(x):
    x = np.sort(np.asarray(x, float))
    x = x - min(0.0, x.min())
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


class Economy:
    def __init__(self, **kw):
        p = dict(DEFAULT, **kw)
        self.p = p
        self.rng = np.random.default_rng(p['seed'])
        N = p['N']
        self.K = p['v'] * 0.9 * p['a'] * N / 0.9          # はじめは稼働率 0.9 で、9 割が働く程度
        self.L_f = 0.3 * self.K
        self.B = 0.6 * 0.9 * p['a'] * N
        r = self.rng
        m = r.exponential(1.0, N); self.m = m / m.sum() * (self.L_f + self.B)   # Σ m = L_f + B から始める
        ld = r.exponential(1.0, N); self.land = ld / ld.sum() * p['T_land']
        self.ip = r.exponential(1.0, N)
        eq = r.exponential(1.0, N) ** 2; self.eq = eq / eq.sum()
        self.order = np.arange(N)                          # 働く順(前の期に働いた人が先)
        self.p_T = 1.0
        self.gT_e = 0.0
        Y0 = 0.9 * p['a'] * N
        self.Pi_e = (1 - p['omega'] - p['rho_T']) * Y0 - p['d0'] * self.K - (p['r_cb'] + p['spread']) * self.L_f
        self.Pi_prev = self.Pi_e
        self.E_prev = None
        self.Y_prev = Y0
        self.t = 0
        self._shock_base = {}

    def apply_shocks(self):
        p = self.p
        for k_, (ts, ch, dur) in enumerate(p['shocks']):
            if self.t >= ts:
                base = self._shock_base.setdefault(k_, {k: p[k] for k in ch})
                w = 1.0 if dur <= 0 else min(1.0, (self.t - ts + 1) / dur)
                for k, v in ch.items():
                    p[k] = base[k] + w * (v - base[k])

    def step(self):
        p = self.p
        self.apply_shocks()
        N = p['N']
        r_L = p['r_cb'] + p['spread']; r_D = r_B = p['r_cb']
        K, L_f, B, m = self.K, self.L_f, self.B, self.m
        # 1. 株式的な富 E と損耗(v6 と同じ形)
        disc = r_L + p['prem']
        wE = 0.001 * K
        E = _sp(self.Pi_e, wE) / (_sp(disc, 0.005) + 0.005)
        delta = p['d0']
        if self.E_prev is not None:
            delta = p['d0'] * (1 + p['kappa'] * max(0.0, -(E - self.E_prev) / max(self.E_prev, 1e-9)))
        self.E_prev = E
        # 2. 投資と、その借入(担保の上限つき)
        u_prev = self.Y_prev / (K / p['v'])
        g_net = p['g0'] + p['g1'] * self.Pi_prev / K + p['g2'] * (u_prev - 0.85) + p['g_q'] * (E / K - p['q_ref'])
        I_des = K * max(0.0, p['d0'] + g_net)
        bor_des = (1 - p['f_R']) * (I_des - p['d0'] * K)          # 純投資のうち、借りて賄う分
        bor = _softmin(bor_des, (p['lam'] * (E + K) - L_f) / p['T_adj'], 0.01 * K)   # 担保の上限
        I = max(0.0, p['d0'] * K + bor / (1 - p['f_R']))
        Y_k = p['u_max'] * K / p['v']
        Y_n = p['a'] * N                                    # 完全雇用で作れる量
        Y_p = _softmin(Y_k, Y_n, 0.01 * Y_n)                # 生産能力は、資本と労働の小さいほう
        G = p['gG'] * Y_p / p['u_max']
        tau = min(max(p['tau'] + p['fiscal'] * (B / self.Y_prev - p['b_target']), 0.0), 0.8)
        # 3. 所得の決定(家計の消費が所得に依存するので反復で解く)
        HL0 = np.maximum(-m, 0.0); D0 = np.maximum(m, 0.0)
        pi_b_pre = r_L * (L_f + HL0.sum()) + r_B * B - r_D * D0.sum()   # 銀行の利子の利益(貸倒れの前)
        ip_tot = self.ip.sum()
        Y = self.Y_prev
        for it in range(200):
            n_emp = int(min(N, round(Y / p['a'])))
            W = p['omega'] * Y
            wage = W / max(n_emp, 1)
            R_T = p['rho_T'] * Y
            OP = Y - W - R_T - p['d0'] * K
            Roy = p['phi_R'] * max(OP, 0.0)
            Pi = OP - Roy - r_L * L_f
            div_f = Pi - p['f_R'] * (I - p['d0'] * K)       # 純投資の f_R を留保し、残りを配当(赤字なら持ち主が補う)
            emp = np.zeros(N, bool); emp[self.order[:n_emp]] = True
            inc_w = np.where(emp, wage, 0.0)
            inc_u = np.where(emp, 0.0, p['u_b'])
            inc_T = R_T * self.land / p['T_land']
            inc_R = Roy * self.ip / ip_tot if ip_tot > 0 else np.zeros(N)
            inc_d = (div_f + pi_b_pre) * self.eq
            inc_i = r_D * D0 - r_L * HL0
            taxable = inc_w + inc_T + inc_R + np.maximum(inc_d, 0.0) + r_D * D0
            tax = tau * taxable
            y_d = inc_w + inc_u + inc_T + inc_R + inc_d + inc_i - tax
            lux = p['c_y'] * np.maximum(0.0, y_d - p['n_c']) + p['c_m'] * D0 + p['c_h'] * np.maximum(0.0, self.p_T * self.land - HL0)
            C = N * p['n_c'] + lux.sum()
            Yd = C + I + G
            Y_new = max(_softmin(Yd, Y_p, 0.01 * Y_p), 1e-9)
            if abs(Y_new - Y) < 1e-10 * Y:
                Y = Y_new
                break
            Y = 0.5 * Y + 0.5 * Y_new
        gap = max(0.0, Yd - Y)
        if gap > 0 and lux.sum() > 0:                       # 生産能力を超えた需要は、贅沢品を比例して削る
            lux = lux * max(0.0, 1 - gap / lux.sum())
        C = N * p['n_c'] + lux.sum()
        Y = C + I + G                                       # 実際の生産 = 実際の支出
        cons = p['n_c'] + lux
        res = p['sigma_R'] * lux                            # 自主研究(消費であり、知財を生む)
        # 4. 家計の受け払い(土地の市場の前)
        self.m = m + (inc_w + inc_u + inc_T + inc_R + inc_d + inc_i - tax - cons)
        self.ip = self.ip * (1 - p['delta_IP']) + p['eta'] * res
        # 5. 企業: 借入の変化は会計そのもの(投資 + 支払い − 売り上げ)。キャッシュフローが赤字なら一部が貸倒れ
        dL_f = I + W + R_T + Roy + r_L * L_f + div_f - Y
        CF = Y - W - R_T - r_L * L_f
        wo_f = min(p['default_share'] * max(0.0, -CF), 0.5 * L_f)
        # 6. 政府
        U = p['u_b'] * (N - n_emp)
        dB = G + U + r_B * B - tax.sum()
        # 7. 土地の市場
        wbar = W / max(n_emp, 1)
        U_lim = p['mu_u'] * wbar
        eqval = E
        v_IP = Roy / (max(self.ip.sum(), 1e-9) * (r_L + p['delta_IP']))
        y_T = R_T / p['T_land'] / self.p_T
        x_T = 0.2 * np.tanh((y_T + self.gT_e - r_L) / 0.2)
        theta = float(min(p['theta0'] * np.exp(p['theta1'] * x_T), 0.9))
        other = self.m + v_IP * self.ip + eqval * self.eq
        human = p['h_y'] * (inc_w + inc_u) * (1 - tau)       # 稼ぐ力(税引き後の労働所得の h_y 年分)
        land, chi, lT = self.land, p['chi'], p['lam_T']
        def holdings(pr):
            nw = other + pr * land
            d = np.maximum(0.0, theta * (nw + human) / pr)
            cap = (U_lim + self.m + pr * land) / ((1 - lT) * pr)
            return np.clip(land + chi * (d - land), 0.0, np.maximum(cap, 0.0))
        lo, hi = np.log(self.p_T) - 8, np.log(self.p_T) + 8
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if holdings(np.exp(mid)).sum() > p['T_land']:
                lo = mid
            else:
                hi = mid
        pT = float(np.exp(0.5 * (lo + hi)))
        new = holdings(pT)
        new = new * (p['T_land'] / new.sum())
        self.m = self.m - pT * (new - land)
        self.land = new
        self.gT_e += p['nu'] * (np.log(pT / self.p_T) - self.gT_e)
        self.p_T = pT
        # 8. 借入の上限の検査: 超えた分は貸倒れ
        limit = lT * pT * self.land + U_lim
        over = np.maximum(0.0, -self.m - limit)
        wo_h = over.sum()
        self.m = self.m + over
        # 9. 銀行: 貸倒れの損失を、持分に応じて家計が負担する(自己資本 0)
        self.m = self.m - (wo_f + wo_h) * self.eq
        # 10. ストックの更新
        self.L_f = L_f + dL_f - wo_f
        self.B = B + dB
        self.K = max(K + I - delta * K, 1e-3)
        self.Pi_e += (Pi - self.Pi_e) / p['tau_e']
        self.Pi_prev = Pi
        self.Y_prev = Y
        # 世代の入れ替わり: 去る人の財産(預金・借金、土地、知財、持分)を相続させ、財産のない若い人と入れ替える
        n_out = self.rng.binomial(N, p['turnover']) if p['turnover'] > 0 else 0
        if n_out:
            out_idx = self.rng.choice(N, n_out, replace=False)
            for i in out_idx:
                if p['inherit'] == 'equal':
                    heirs = np.delete(np.arange(N), i); share = 1.0 / (N - 1)
                    for arr in (self.m, self.land, self.ip, self.eq):
                        arr[heirs] += arr[i] * share
                else:
                    h = self.rng.integers(N - 1); h = h + (h >= i)
                    for arr in (self.m, self.land, self.ip, self.eq):
                        arr[h] += arr[i]
                for arr in (self.m, self.land, self.ip, self.eq):
                    arr[i] = 0.0
        # 働く順の並べ直し(働いた人を前に、それぞれの中はランダム)
        e_idx = self.order[:n_emp]; u_idx = self.order[n_emp:]
        self.order = np.concatenate([self.rng.permutation(e_idx), self.rng.permutation(u_idx)])
        self.t += 1
        nw = self.m + pT * self.land + v_IP * self.ip + eqval * self.eq
        resid = self.m.sum() - (self.L_f + self.B)
        srt = np.sort(nw)
        return dict(tau=tau, u=Y / (K / p['v']), Y=Y, C=C, I=I, G=G, K=self.K, L_f=self.L_f, B=self.B, E=E, q=E / K, delta=delta, p_T=pT,
                    HL=np.maximum(-self.m, 0).sum(), D=np.maximum(self.m, 0).sum(), theta=theta, gT_e=self.gT_e,
                    emp=n_emp / N, wo_f=wo_f, wo_h=wo_h, lux=lux.sum(), res=res.sum(), IP=self.ip.sum(), Pi=Pi,
                    div_f=div_f, pi_b=pi_b_pre - wo_f - wo_h, gini_nw=gini(nw), top10_nw=srt[-N // 10:].sum() / srt.sum(),
                    n_over=int((over > 0).sum()), resid=resid, land_resid=self.land.sum() - p['T_land'], nw_mean=nw.mean())


def simulate(**kw):
    """予備走行(burn 年。ショックなし)のあと T 年走らせ、年ごとの記録(dict of arrays)と、最後の経済を返す。
    ショックの年は、予備走行のあとから数える。"""
    p = dict(DEFAULT, **kw)
    shocks = p['shocks']
    e = Economy(**dict(kw, shocks=()))
    for _ in range(p['burn']):
        e.step()
    e.p['shocks'] = [(ts + e.t, ch, dur) for ts, ch, dur in shocks]
    out = {k: np.empty(p['T']) for k in REC}
    for t in range(p['T']):
        r = e.step()
        for k in REC:
            out[k][t] = r[k]
    return out, e
