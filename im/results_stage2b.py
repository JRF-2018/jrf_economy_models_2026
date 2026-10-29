# results_stage2b.py: 段 2b の実験(知財の市場)。python -W ignore results_stage2b.py > results_stage2b.txt
# 問い(JRF): 2019 年の資産市場の模型で「知財は買えば不利」だった。統合模型では、それはどうなったか。
import numpy as np
import im1

BASE = dict(households='continuous', Ns=400, g_gamma=0.0, credit=True, land=True, c_h=0.005, ip_market=True, fiscal='deficit')


def summary(lab, **kw):
    a, e = im1.simulate(T=400, **dict(BASE, **kw))
    n = slice(-200, None)
    Y = a['Y'][n]
    R, rL = a['R_ip'][n], a['r_loan'][n]
    fund = np.maximum(a['ip_fund'][n], 1e-300)
    ratio = np.nanmedian(a['qIP'][n] / fund) if np.any(a['ip_fund'][n] > 0) else np.inf
    # 作り手が新しい知財を(その年の値段で)持つ値打ちと、持ち主の損(貸出の金利に比べた不足)
    create = np.mean(a['qIP'][n] * a['ip_new'][n] / Y)
    value_prev = np.concatenate([[np.nan], a['ip_value'][n][:-1]])
    loss = np.nanmean((rL - R) * value_prev / Y)
    print(f"  {lab:34s} 知財の値打ち/所得 {np.mean(a['ip_value'][n] / Y):.3f}  値段/使用料の割引の値 {ratio:9.3g}  "
          f"持ち主の利回り {np.nanmean(R):+.3f}(貸出の金利 {rL.mean():.3f}、物価 {a['infl'][n].mean():+.3f})  "
          f"新しい知財の値打ち/所得 {create:.4f}  持ち主の不足/所得 {loss:+.4f}  雇用 {a['E'][n].mean():.0f}  正味資産のジニ {a['gini_nw'][n].mean():.3f}")
    return a


if __name__ == '__main__':
    print('P1 知財の市場(400 年、最後の 200 年)。需要の決まり方は土地と同じ(利回りの差は ±0.2 で頭打ち、反応 3)')
    summary('使用料あり(φ_R = 0.1)')
    summary('使用料なし(φ_R = 0。2019 年と同じ)', phi_R=0.0)
    print('P2 需要が利回りに強く反応する(頭打ちを広げ、反応を強める)。値上がりの期待の外挿は切る(ν_IP = 0。利回りだけに反応)')
    for xc, t1 in ((0.2, 3.0), (0.2, 10.0), (1.0, 10.0), (2.0, 30.0)):
        summary(f'使用料あり、頭打ち ±{xc}、反応 {t1}', xcap_IP=xc, theta1_IP=t1, nu_IP=0.0)
    for xc, t1 in ((1.0, 10.0), (2.0, 30.0)):
        summary(f'使用料なし、頭打ち ±{xc}、反応 {t1}', phi_R=0.0, xcap_IP=xc, theta1_IP=t1, nu_IP=0.0)
    print('P3 世代の入れ替わり(新しく加わる人)を切る')
    summary('使用料あり、入れ替わりなし', turnover=0.0)
    summary('使用料なし、入れ替わりなし', phi_R=0.0, turnover=0.0)
    print('P5 中立の需要(利回りが貸出の金利と同じときに知財に置きたい割合 θ0_IP)を小さくする。反応 10、頭打ち ±1、外挿なし')
    for t0 in (0.05, 0.01, 0.003, 0.001, 0.0003):
        summary(f'使用料あり、θ0_IP = {t0}', theta0_IP=t0, xcap_IP=1.0, theta1_IP=10.0, nu_IP=0.0)
    summary('使用料なし、θ0_IP = 0.001', phi_R=0.0, theta0_IP=0.001, xcap_IP=1.0, theta1_IP=10.0, nu_IP=0.0)
    print('P6 裁定の価格(値段 = 使用料を、貸出の金利 + 減耗で割り引いた値打ち)。比べる基準')
    summary('使用料あり、裁定の価格', ip_pricing='fundamental')
    print('P4 値上がりの期待の外挿(ν_IP = 0.2)のまま、反応を強めると(泡の暴走)')
    summary('使用料あり、頭打ち ±1.0、反応 10、外挿あり', xcap_IP=1.0, theta1_IP=10.0)
