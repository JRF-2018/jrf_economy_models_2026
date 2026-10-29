# results_stage14.py: 段 14 の実験(作り手の流出と、国庫の案)。python -W ignore results_stage14.py > results_stage14.txt
# 段 5 の世界樹の案(部門に 20 社、規則的な企業と真似る企業が半々、参入を選ぶ)の上で。標本 2000 記入、400 年の後半 300 年、種 12 個。
from results_stage5c import table, F20


def w5_spin_prize():
    table('W5 作り手の流出(研究者が知識を持って独立する確率)と、国庫の案(部門の利益への税を、研究の当たりに比例して配り、研究に回す)',
          (('支えなし', dict(F20)),
           ('流出 2%/年', dict(F20, p_spin=0.02)),
           ('流出 5%/年', dict(F20, p_spin=0.05)),
           ('国庫 10%', dict(F20, t_prize=0.1)),
           ('国庫 20%', dict(F20, t_prize=0.2)),
           ('国庫 30%', dict(F20, t_prize=0.3)),
           ('国庫 30%、研究に回さない', dict(F20, t_prize=0.3, prize_reinvest=False)),
           ('国庫 30% + 流出 2%', dict(F20, t_prize=0.3, p_spin=0.02)),
           ('知財の保護 10 年(段 5 の W2)', dict(F20, ip_lag=10)),
           ('知財の保護 10 年 + 流出 2%', dict(F20, ip_lag=10, p_spin=0.02))))


def w6_monopoly():
    M = dict(F20, mu_hhi=0.2)
    table('W6 集中の値上げあり(段 5 の W4 と同じ mu_hhi = 0.2)で、知財の保護・公の研究所・国庫を比べる',
          (('支えなし', M), ('知財の保護 10 年', dict(M, ip_lag=10)), ('公の研究所 3%', dict(M, s_pub=0.03)),
           ('国庫 20%', dict(M, t_prize=0.2)), ('国庫 30%', dict(M, t_prize=0.3)), ('国庫 30% + 流出 2%', dict(M, t_prize=0.3, p_spin=0.02)),
           ('知財の保護 10 年 + 流出 2%', dict(M, ip_lag=10, p_spin=0.02))))


if __name__ == '__main__':
    w5_spin_prize()
    w6_monopoly()
