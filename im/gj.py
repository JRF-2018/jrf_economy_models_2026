# %%CELL head
# gj.py: 第 2 章 必需品と贅沢品のマクロ経済を、帳簿を閉じて書き直した小さな模型
# 元のノート(gj_6)のプログラムは使わない。ここで新しく書く。
# 賃金を 1(物差し)とし、一期ごとに、支出 = 所得 となる釣り合いを求める。
from dataclasses import dataclass, replace, asdict
import numpy as np


@dataclass
class GJ:
    """模型のパラメータ。労働者(大衆)は人口の Hw、資本家(富裕層)は Hk。働くのは労働者だけで、労働力は Lmax。"""
    Hw: float = 0.9          # 労働者の人口
    Hk: float = 0.1          # 資本家の人口
    gw: float = 0.3          # 労働者一人あたりの必需品の必要量
    gk: float = 0.5          # 資本家一人あたりの必需品の必要量
    bw: float = 0.3          # 労働者が、必要量を超えた所得のうち必需品に回す割合(限界の割合)
    bk: float = 0.1          # 資本家の同じ割合
    s: float = 0.5           # 資本家の貯蓄の割合
    iota: float = 0.5        # 貯蓄のうち、研究と文化への投資(人を雇う)に回す割合。残りは国債や現金で持つ(漏れ)
    psi: float = 0.0         # 文化(贅沢品の種類の数 n)が貯蓄の割合を下げる強さ: s / (1 + ψ n)
    rho: float = 0.75        # 贅沢品の種類の間の代替のしやすさ。値上げ幅(マークアップ)は 1/ρ
    aL: float = 1.0          # 贅沢品の生産性
    tau: float = 0.2         # 再分配: 資本家の所得への税率。全額を労働者に配る
    G: float = 0.02          # 公共事業(人を雇う)。国債で賄う
    ic: float = 0.03         # 企業の運転資金の金利(賃金を先に払うために借りる。利子は資本家の所得)
    iB: float = 0.02         # 国債の金利(利子は資本家の所得)
    Lmax: float = 1.0        # 労働力


def stone_geary(M, need, b, pN):
    """所得 M の家計の支出。まず必需品の必要量 need を買い、残り(余裕)の割合 b を必需品、1 − b を贅沢品に使う。
    所得が必要量に届かなければ、すべてを必需品に使う。戻り値は (必需品への支出, 贅沢品への支出)。"""
    if M <= pN * need:
        return M, 0.0
    extra = M - pN * need
    return pN * need + b * extra, (1 - b) * extra


# %%CELL period
def period(p, a, n=1.0, Lcap=np.inf, B=1.0, tol=1e-12):
    """一期の釣り合い。a は必需品の生産性(技術)、n は贅沢品の種類の数(文化)、Lcap は必需品の部門で
    すぐに働ける人の上限(短期の生産能力)、B は国債の残高。
    必需品は競争的で、価格は費用 (1 + ic)/a。需要が生産能力を超えると価格が上がり、差額は持ち主(資本家)のレントになる。
    贅沢品は種類ごとに独占的に競争し、価格は費用の 1/ρ 倍。働く人が足りなければ、価格がさらに上がる。
    すべての支払いは誰かの所得になる: 賃金は労働者に、レント・利潤・利子は資本家に。"""
    cN = (1 + p.ic) / a
    cL = (1 + p.ic) / (p.rho * p.aL)
    s = p.s / (1 + p.psi * n)
    pN, pL, Mw, Mk = cN, cL, 0.5, 0.2
    for it in range(20000):
        ENw, ELw = stone_geary(Mw, p.Hw * p.gw, p.bw, pN)
        Ck = (1 - s) * Mk
        ENk, ELk = stone_geary(Ck, p.Hk * p.gk, p.bk, pN)
        I = p.iota * s * Mk
        EN, EL = ENw + ENk, ELw + ELk
        LN = min(EN / pN / a, Lcap)
        pN_new = max(cN, EN / (a * Lcap)) if np.isfinite(Lcap) else cN
        free = max(p.Lmax - LN - I - p.G, 1e-9)
        LL = min(EL / pL / p.aL, free)
        pL_new = max(cL, EL / (p.aL * free))
        L = LN + LL + I + p.G
        rent = EN - (1 + p.ic) * LN          # 必需品の売上から、賃金と運転資金の利子を引いた残り
        profit = EL - (1 + p.ic) * LL        # 贅沢品の売上から、賃金と利子を引いた残り
        interest = p.ic * (LN + LL) + p.iB * B
        Mk_pre = rent + profit + interest
        T = p.tau * Mk_pre
        Mw_new, Mk_new = L + T, Mk_pre - T
        err = abs(Mw_new - Mw) + abs(Mk_new - Mk) + abs(pN_new - pN) + abs(pL_new - pL)
        d = 0.3
        Mw += d * (Mw_new - Mw); Mk += d * (Mk_new - Mk)
        pN += d * (pN_new - pN); pL += d * (pL_new - pL)
        if err < tol:
            break
    # 部門の収支: 労働者は貯めない。資本家の貯蓄の残り(投資に回さない分)と、政府の赤字は一致するはず
    Sk = Mk - Ck - I
    deficit = p.G + p.iB * B
    welfare_lux = (EL / pL) * n ** ((1 - p.rho) / p.rho)   # 種類の数が多いほど、同じ量の贅沢品でも満足が大きい(多様性への愛好)
    return dict(pN=pN, cN=cN, pL=pL, cL=cL, EN=EN, EL=EL, LN=LN, LL=LL, I=I, L=L, unemp=p.Lmax - L,
                rent=rent, profit=profit, interest=interest, Mw=Mw, Mk=Mk, T=T, Ck=Ck, Sk=Sk, deficit=deficit,
                books=Sk - deficit, QN=EN / pN, lux_index=welfare_lux, it=it)


def table(rows, keys, fmt='{:8.3f}'):
    print(' ' * 9 + ''.join(f'{k:>10}' for k in keys))
    for label, r in rows:
        print(f'{label:>9}' + ''.join(fmt.format(r[k]).rjust(10) for k in keys))
