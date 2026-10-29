# %%CELL model
"""v6b.py -- 資産市場つきの v6 の別の版: 個人を「標本」として扱い、その統計だけをマクロに渡す。

v6a は、1000 人の個人そのものが経済の全員で、マクロの量は個人の和だった(個人とマクロを「直接」つないだ)。
v6b では、二つを分ける(物理学の多重スケールの方法: 制限 restriction と持ち上げ lifting。Kevrekidis らの equation-free、
E と Engquist の heterogeneous multiscale method。経済学では Krusell と Smith の近似集計に近い)。

* マクロ: 人口 Npop(連続量)の経済。企業・銀行・政府は v6a と同じ。家計部門のお金は、銀行の自己資本が 0 なので、
  いつも M_h = L_f + B(マクロの帳簿だけで閉じる)。土地の総量、知財の総量もマクロの量として持つ。
* ミクロ: Ns 人の「標本」。一人は人口の Npop / Ns 人を代表する。標本の大きさは Npop と独立に選べる。
* 制限(ミクロ → マクロ): 標本で「ミクロの実験」をして、マクロが使う関係式を推定する。
    - 一人あたりの贅沢品の消費の、所得 Y への応答(三つの Y で標本を動かし、二次式を当てはめる)
    - 土地の需要曲線(価格の格子で、標本の望む保有を数える)
    - 借入の上限を超える人の割合と額(貸倒れの率)、預金と借金の総額の比
* 持ち上げ(マクロ → ミクロ): 毎期、標本の平均を、マクロの総量(一人あたりのお金、土地、知財)に合わせ直す。
  だから、標本の誤差はマクロに積もらない。標本は、マクロの「場」(賃金の水準、雇用の率、地代、使用料、配当、金利、税率、土地の値段)
  のもとで、一人ずつ動く。
"""
import numpy as np
from v6a import DEFAULT as DEFAULT_A, _softmin, _sp, gini

DEFAULT = dict(DEFAULT_A, Npop=1000.0, Ns=1000, T_land_pc=1.0)
REC = ['tau', 'u', 'Y', 'C', 'I', 'G', 'K', 'L_f', 'B', 'E', 'q', 'delta', 'p_T', 'HL', 'D', 'theta', 'gT_e', 'emp',
       'wo_f', 'wo_h', 'lux', 'res', 'IP', 'Pi', 'gini_nw', 'top10_nw', 'fit_err', 'land_err']


class Sample:
    """ミクロの標本。量はすべて「一人あたり」の単位(平均がマクロの一人あたりの量に一致するように持ち上げる)。"""

    def __init__(self, p, rng, M_pc, IP_pc):
        Ns = p['Ns']
        self.Ns, self.rng = Ns, rng
        m = rng.exponential(1.0, Ns); self.m = m / m.mean() * M_pc
        ld = rng.exponential(1.0, Ns); self.land = ld / ld.mean() * p['T_land_pc']
        ip = rng.exponential(1.0, Ns); self.ip = ip / ip.mean() * IP_pc
        eq = rng.exponential(1.0, Ns) ** 2; self.eq = eq / eq.mean()      # 持分(平均 1。人口全体の配当を一人あたりに割ったものに掛ける)
        self.order = np.arange(Ns)

    def lift(self, M_pc, T_pc, IP_pc):
        """持ち上げ: 標本の平均をマクロの一人あたりの量に合わせる。お金は全員を同じ額だけずらし(散らばりを保つ)、
        土地と知財は比で合わせる。"""
        self.m += M_pc - self.m.mean()
        if self.land.mean() > 0:
            self.land *= T_pc / self.land.mean()
        if self.ip.mean() > 0:
            self.ip *= IP_pc / self.ip.mean()

    def incomes(self, f):
        """マクロの場 f のもとでの、一人ずつの所得と贅沢品の消費。f は一人あたりの量の dict。"""
        p = f['p']
        Ns = self.Ns
        n_emp = f['e'] * Ns                                   # 雇用の率 × 標本の人数(端数は、境目の一人が一部だけ働く)
        k = int(np.floor(n_emp)); frac = n_emp - k
        work = np.zeros(Ns); work[self.order[:k]] = 1.0
        if k < Ns:
            work[self.order[k]] = frac
        HL0 = np.maximum(-self.m, 0.0); D0 = np.maximum(self.m, 0.0)
        inc_w = work * f['wage']
        inc_u = (1 - work) * p['u_b']
        inc_T = f['rent'] * self.land
        inc_R = f['roy'] * self.ip
        inc_d = f['div'] * self.eq
        inc_i = f['r_D'] * D0 - f['r_L'] * HL0
        taxable = inc_w + inc_T + inc_R + np.maximum(inc_d, 0.0) + f['r_D'] * D0
        tax = f['tau'] * taxable
        y_d = inc_w + inc_u + inc_T + inc_R + inc_d + inc_i - tax
        lux = p['c_y'] * np.maximum(0.0, y_d - p['n_c']) + p['c_m'] * D0 + p['c_h'] * np.maximum(0.0, f['p_T'] * self.land - HL0)
        return dict(work=work, inc=inc_w + inc_u + inc_T + inc_R + inc_d + inc_i - tax, lux=lux, tax=tax, HL0=HL0, D0=D0,
                    labor=(inc_w + inc_u) * (1 - f['tau']))


class Economy:
    def __init__(self, **kw):
        p = dict(DEFAULT, **kw)
        self.p = p
        self.rng = np.random.default_rng(p['seed'])
        Np = p['Npop']
        self.T_land = p['T_land_pc'] * Np
        self.K = p['v'] * 0.9 * p['a'] * Np / 0.9
        self.L_f = 0.3 * self.K
        self.B = 0.6 * 0.9 * p['a'] * Np
        self.IP = 1.0 * Np
        self.S = Sample(p, self.rng, (self.L_f + self.B) / Np, self.IP / Np)
        self.p_T = 1.0
        self.gT_e = 0.0
        Y0 = 0.9 * p['a'] * Np
        self.Pi_e = (1 - p['omega'] - p['rho_T']) * Y0 - p['d0'] * self.K - (p['r_cb'] + p['spread']) * self.L_f
        self.Pi_prev = self.Pi_e
        self.E_prev = None
        self.Y_prev = Y0
        self.HL_share = 0.0                                   # 標本から推定した、家計の借金の総額(一人あたり)
        self.t = 0
        self._shock_base = {}

    apply_shocks = __import__('v6a').Economy.apply_shocks

    def fields(self, Y, tau, I, r_L, r_D):
        """所得 Y のもとでの、マクロの一人あたりの「場」。"""
        p = self.p; Np = p['Npop']
        e = min(1.0, Y / (p['a'] * Np))
        W = p['omega'] * Y
        R_T = p['rho_T'] * Y
        OP = Y - W - R_T - p['d0'] * self.K
        Roy = p['phi_R'] * max(OP, 0.0)
        Pi = OP - Roy - r_L * self.L_f
        div_f = Pi - p['f_R'] * (I - p['d0'] * self.K)
        return dict(p=p, e=e, W=W, R_T=R_T, OP=OP, Roy=Roy, Pi=Pi, div_f=div_f,
                    wage=W / max(e * Np, 1e-9), rent=R_T / self.T_land, roy=Roy / max(self.IP, 1e-9),
                    div=(div_f + self.pi_b_pre) / Np, r_D=r_D, r_L=r_L, tau=tau, p_T=self.p_T)

    def step(self):
        p = self.p
        self.apply_shocks()
        Np = p['Npop']
        r_L = p['r_cb'] + p['spread']; r_D = r_B = p['r_cb']
        K, L_f, B = self.K, self.L_f, self.B
        M = L_f + B                                           # 家計部門のお金(マクロの帳簿)
        # 0. 持ち上げ
        self.S.lift(M / Np, self.T_land / Np, self.IP / Np)
        S = self.S
        # 1〜2. E、損耗、投資(v6a と同じ。マクロだけ)
        disc = r_L + p['prem']
        E = _sp(self.Pi_e, 0.001 * K) / (_sp(disc, 0.005) + 0.005)
        delta = p['d0']
        if self.E_prev is not None:
            delta = p['d0'] * (1 + p['kappa'] * max(0.0, -(E - self.E_prev) / max(self.E_prev, 1e-9)))
        self.E_prev = E
        u_prev = self.Y_prev / (K / p['v'])
        g_net = p['g0'] + p['g1'] * self.Pi_prev / K + p['g2'] * (u_prev - 0.85) + p['g_q'] * (E / K - p['q_ref'])
        I_des = K * max(0.0, p['d0'] + g_net)
        bor_des = (1 - p['f_R']) * (I_des - p['d0'] * K)
        bor = _softmin(bor_des, (p['lam'] * (E + K) - L_f) / p['T_adj'], 0.01 * K)
        I = max(0.0, p['d0'] * K + bor / (1 - p['f_R']))
        Y_p = _softmin(p['u_max'] * K / p['v'], p['a'] * Np, 0.01 * p['a'] * Np)
        G = p['gG'] * Y_p / p['u_max']
        tau = min(max(p['tau'] + p['fiscal'] * (B / self.Y_prev - p['b_target']), 0.0), 0.8)
        # 銀行の利子の利益(貸倒れの前)。家計の預金と借金の総額は、標本の分布から推定する
        HL_pc = np.maximum(-S.m, 0).mean(); D_pc = np.maximum(S.m, 0).mean()
        self.pi_b_pre = r_L * (L_f + HL_pc * Np) + r_B * B - r_D * D_pc * Np
        # 3. 制限: 標本で「ミクロの実験」をして、一人あたりの贅沢品の消費の Y への応答を二次式で推定する
        Ys = self.Y_prev * np.array([0.85, 1.0, 1.15])
        lux_pc = [S.incomes(self.fields(y, tau, I, r_L, r_D))['lux'].mean() for y in Ys]
        coef = np.polyfit(Ys, lux_pc, 2)
        lux_fn = lambda y: max(np.polyval(coef, y), 0.0)
        # マクロの所得の決定(推定した関係式だけを使う)
        Y = self.Y_prev
        for _ in range(200):
            C = Np * (p['n_c'] + lux_fn(Y))
            Yd = C + I + G
            Y_new = max(_softmin(Yd, Y_p, 0.01 * Y_p), 1e-9)
            if abs(Y_new - Y) < 1e-10 * Y:
                Y = Y_new; break
            Y = 0.5 * Y + 0.5 * Y_new
        f = self.fields(Y, tau, I, r_L, r_D)
        gap = max(0.0, Yd - Y)
        lux_tot = Np * lux_fn(Y)
        if gap > 0 and lux_tot > 0:
            lux_tot *= max(0.0, 1 - gap / lux_tot)
        C = Np * p['n_c'] + lux_tot
        Y = C + I + G
        f = self.fields(Y, tau, I, r_L, r_D)
        # 標本を、その場で一期動かす(推定の誤差を測るため、標本自身の贅沢品の消費も記録する)
        mi = S.incomes(f)
        lux_s = mi['lux'].mean()
        scale = (lux_tot / Np) / lux_s if lux_s > 0 else 1.0      # 標本の消費を、マクロの消費の総量に合わせる(制限と持ち上げの整合)
        lux_i = mi['lux'] * scale
        fit_err = (lux_s - lux_tot / Np) / max(lux_tot / Np, 1e-9)
        S.m = S.m + mi['inc'] - (p['n_c'] + lux_i)
        res_i = p['sigma_R'] * lux_i
        S.ip = S.ip * (1 - p['delta_IP']) + p['eta'] * res_i
        self.IP = self.IP * (1 - p['delta_IP']) + p['eta'] * p['sigma_R'] * lux_tot
        # 4〜6. マクロの帳簿: 企業、政府(税はマクロの所得の流れから)
        W, R_T, Roy, Pi, div_f = f['W'], f['R_T'], f['Roy'], f['Pi'], f['div_f']
        n_emp = f['e'] * Np
        D0_tot = D_pc * Np
        div_tot = div_f + self.pi_b_pre
        tax_tot = tau * (W + R_T + Roy + max(div_tot, 0.0) + r_D * D0_tot)
        dL_f = I + W + R_T + Roy + r_L * L_f + div_f - Y
        CF = Y - W - R_T - r_L * L_f
        wo_f = min(p['default_share'] * max(0.0, -CF), 0.5 * L_f)
        U = p['u_b'] * (Np - n_emp)
        dB = G + U + r_B * B - tax_tot
        # 7. 土地の市場: 標本の需要曲線(一人あたり)× 人口 = 土地の総量 となる値段を、マクロが解く
        wbar = W / max(n_emp, 1e-9)
        U_lim = p['mu_u'] * wbar
        v_IP = Roy / (max(self.IP, 1e-9) * (r_L + p['delta_IP']))
        y_T = R_T / self.T_land / self.p_T
        x_T = 0.2 * np.tanh((y_T + self.gT_e - r_L) / 0.2)
        theta = float(min(p['theta0'] * np.exp(p['theta1'] * x_T), 0.9))
        other = S.m + v_IP * S.ip + (E / Np) * S.eq
        human = p['h_y'] * mi['labor']
        land, chi, lT = S.land, p['chi'], p['lam_T']

        def holdings(pr):
            nw = other + pr * land
            d = np.maximum(0.0, theta * (nw + human) / pr)
            cap = (U_lim + S.m + pr * land) / ((1 - lT) * pr)
            return np.clip(land + chi * (d - land), 0.0, np.maximum(cap, 0.0))
        T_pc = self.T_land / Np
        lo, hi = np.log(self.p_T) - 8, np.log(self.p_T) + 8
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if holdings(np.exp(mid)).mean() > T_pc:
                lo = mid
            else:
                hi = mid
        pT = float(np.exp(0.5 * (lo + hi)))
        new = holdings(pT)
        land_err = new.mean() / T_pc - 1
        new = new * (T_pc / new.mean())
        S.m = S.m - pT * (new - land)
        S.land = new
        self.gT_e += p['nu'] * (np.log(pT / self.p_T) - self.gT_e)
        self.p_T = pT
        # 8. 貸倒れの率を標本から推定し、マクロの額にする
        limit = lT * pT * S.land + U_lim
        over = np.maximum(0.0, -S.m - limit)
        wo_h = over.mean() * Np
        S.m = S.m + over
        S.m = S.m - (wo_f + wo_h) / Np * S.eq                   # 銀行の損失を持分に応じて負担(一人あたり)
        # 10. マクロのストック
        self.L_f = L_f + dL_f - wo_f
        self.B = B + dB
        self.K = max(K + I - delta * K, 1e-3)
        self.Pi_e += (Pi - self.Pi_e) / p['tau_e']
        self.Pi_prev = Pi
        self.Y_prev = Y
        # 世代の入れ替わり(標本の中で)
        n_out = self.rng.binomial(S.Ns, p['turnover']) if p['turnover'] > 0 else 0
        for i in (self.rng.choice(S.Ns, n_out, replace=False) if n_out else []):
            h = self.rng.integers(S.Ns - 1); h = h + (h >= i)
            for arr in (S.m, S.land, S.ip, S.eq):
                arr[h] += arr[i]; arr[i] = 0.0
        S.eq *= 1.0 / S.eq.mean()
        k = int(round(f['e'] * S.Ns))
        S.order = np.concatenate([self.rng.permutation(S.order[:k]), self.rng.permutation(S.order[k:])])
        self.t += 1
        nw = S.m + pT * S.land + v_IP * S.ip + (E / Np) * S.eq
        srt = np.sort(nw)
        return dict(tau=tau, u=Y / (K / p['v']), Y=Y, C=C, I=I, G=G, K=self.K, L_f=self.L_f, B=self.B, E=E, q=E / K, delta=delta,
                    p_T=pT, HL=HL_pc * Np, D=D_pc * Np, theta=theta, gT_e=self.gT_e, emp=f['e'], wo_f=wo_f, wo_h=wo_h,
                    lux=lux_tot, res=p['sigma_R'] * lux_tot, IP=self.IP, Pi=Pi, gini_nw=gini(nw),
                    top10_nw=srt[-max(S.Ns // 10, 1):].sum() / srt.sum(), fit_err=fit_err, land_err=land_err)


def simulate(**kw):
    """v6a.simulate と同じ使い方。予備走行のあと T 年。"""
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
