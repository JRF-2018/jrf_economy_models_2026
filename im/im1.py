# im1.py: 統合模型 im の段 1(me_c の価格と生産 + 信用 + 標本の家計)
# 1a: me_c v0.1 を骨組みの上に移す。お金は銀行の預金(銀行は国債を持ち、預金を負う)。家計は重みつきの標本。
#     信用を切り(貸出なし、金利 0)、標本を二つの型(労働者・資本家)にすれば、me_c v0.1 と同じ数になる。
import sys, os
import numpy as np
from scipy.optimize import least_squares
from im_core import Params, Ledger, Model, Component
from portable import make_rng, lm_solve, LMResult

HERE = os.path.dirname(os.path.abspath(__file__))
SECT = ('N', 'L', 'I')
FIRMS = ('fN', 'fL', 'fI')
SECTORS = ('hh',) + FIRMS + ('bank', 'gov')
INSTRUMENTS = ('deposits', 'loans', 'bonds', 'foreign')   # foreign: 家計が持つ外の資産(段 6。地域の外への投資の窓口 ext が出す請求権)
REAL = ('capital_N', 'capital_L', 'capital_I', 'stock_N', 'stock_G')   # stock_N: 必需品の企業の在庫、stock_G: 政府の備蓄(段 7)


def softplus(x, k=1.0):
    return np.logaddexp(0.0, k * x) / k


def softmin(a, b, k=50.0):
    return a - softplus(a - b, k / max(abs(b), 1e-9))


def default_params():
    p = Params()
    A = lambda *v: np.array(v, dtype=float)
    # 構造
    p.add('pop_w', 1000.0, 'structure', '労働者の人口(働く人と働かない人)')
    p.add('n_cap', 5.0, 'structure', '資本家の人数')
    p.add('q0', 1.0, 'structure', '一人あたり必需品の生存量')
    p.add('gamma0', A(0.1, 0.3, 0.3), 'structure', '一製品あたり労働(初期)')
    p.add('alpha', A(0.3, 0.5, 0.3), 'structure', '一製品あたり原料')
    p.add('kappa', A(1.0, 1.0, 1.0), 'structure', '生産能力一単位に要る原料(固定資産)')
    p.add('delta', 0.1, 'structure', '能力の減耗(一年あたり。寿命 10 年)')
    p.add('g_gamma', 0.01, 'structure', '新しい能力の一製品あたり労働が毎年下がる率')
    # 行動の仮定
    p.add('m0', A(0.3, 0.3, 0.3), 'behavior', '基準のマークアップ(初期値)')
    p.add('zeta', 0.5, 'behavior', '稼働率がマークアップに効く強さ')
    p.add('u_star', 0.85, 'behavior', '目標の稼働率')
    p.add('omega', 1.0, 'behavior', '能力を超えたときの残業の割増')
    p.add('eta', 0.05, 'behavior', '競争でマークアップが利潤率の差に寄る速さ')
    p.add('phi_r', 0.5, 'behavior', '投資の、利潤率の差への反応')
    p.add('phi_u', 0.3, 'behavior', '投資の、稼働率の差への反応')
    p.add('phi_w', 0.1, 'behavior', '賃金の、労働の需給のずれへの反応')
    p.add('chi', 0.2, 'behavior', '賃金の、前の年の必需品の値上がりへの連動')
    p.add('c_y', 0.9, 'behavior', '労働者の可処分所得からの消費性向')
    p.add('c_m', 0.2, 'behavior', '労働者の預金からの消費性向')
    p.add('c_yc', 0.5, 'behavior', '資本家の可処分所得からの消費性向')
    p.add('c_mc', 0.05, 'behavior', '資本家の預金からの消費性向')
    p.add('b_N', 0.3, 'behavior', '生存量を超えた支出のうち必需品に回る割合')
    p.add('beta_s', 3.0, 'behavior', '労働の供給の、実質賃金への反応')
    p.add('w_res', 2.0, 'behavior', '留保の実質賃金(必需品の単位)')
    p.add('tactic', 1e-4, 'behavior', '価格の解の戦術: 前の年の価格に近い解を選ぶ重み')
    # ブラウザの版との一致(段 B0)。既定は numpy の乱数と scipy の解き方(ノートの記録はこれで出した)
    p.add('rng', 'numpy', 'structure', "乱数: 'numpy'(numpy の Generator)か 'portable'(xoshiro128** と逆関数法。JS の版と同じ列。portable.py)")
    p.add('solver', 'scipy', 'structure', "価格を解く最小化: 'scipy'(least_squares の LM)か 'portable'(両方の言語で同じ手順の LM。portable.py)")
    # 政策
    p.add('r_bar', 0.06, 'policy', '標準の利潤率(信用を切ったとき。1c では金利から決める)')
    p.add('tau0', 0.2, 'policy', '税率の初期値')
    p.add('deficit_target', 0.0, 'policy', '赤字 / 名目の最終需要 の目標(財政のルール)')
    p.add('k_tau', 0.2, 'policy', '税率の調整の速さ')
    p.add('fiscal', 'burden', 'policy', "財政のルール(既定は 'burden': 利払いの重さを見る。JRF 2026-10-28 の案 C。段 1〜4 の記録は 'deficit' で出した): 'anchor'(最初の anchor_start 年は赤字のルールで動いて、その経済の自然な国債/所得の水準を見つけ、"
          "そのあとは、その水準を目標に国債/所得だけに反応する。打撃で国債が目減りしたら戻す。JRF 2026-10-28 の「国債の対所得比の目標」を、"
          "目標の水準を外から決めずに入れたもの)、'debt'(国債/所得を決まった目標 b_target に)、"
          "'deficit'(赤字全体を目標に。me_c と同じ。大きな打撃のあと国債の水準を戻さず、履歴が残る)、'structural'(失業の給付を除いた赤字を目標に)")
    p.add('anchor_start', 100, 'policy', "財政のルール 'anchor': この年までは赤字のルールで動き、この年の国債/所得を目標にする")
    p.add('b_target', 0.3, 'policy', "財政のルール 'debt' の目標: 国債/所得")
    p.add('k_debt', 0.2, 'policy', "財政のルール 'debt' の強さ: 税率を (国債/所得 − 目標) に比例して動かす")
    p.add('benefit', 1.0, 'policy', '失業者への給付(生存量の何倍か)')
    # 信用(1c)。credit = False なら、貸出も利子もなく、r̄ は r_bar(1a・1b と同じ)
    p.add('credit', False, 'structure', '信用(銀行の貸出と金利)を入れるか')
    p.add('r_cb', 0.02, 'policy', '政策金利(預金と国債の金利)。テイラー・ルールを入れたときは、その中立の値')
    p.add('phi_pi', 0.0, 'policy', 'テイラー・ルール: 政策金利 = max(0, r_cb + φ_π (物価上昇率 − 目標))。0 なら金利は固定')
    p.add('pi_target', 0.0, 'policy', 'テイラー・ルールの物価上昇率の目標')
    p.add('spread', 0.02, 'behavior', '貸出の金利の上乗せ(平時)')
    p.add('r_eq', 0.02, 'behavior', '持分の持ち主が貸出の金利の上に求める利回り(r̄ = 貸出の金利 + r_eq)')
    p.add('f_R', 0.3, 'behavior', '投資のうち内部留保で賄う割合(残りは借入)')
    p.add('lam', 0.6, 'behavior', '借入の担保の上限(資本の再調達価格に対する比)')
    p.add('amort', 0.2, 'behavior', '借入の返済の速さ(一年に残高のこの割合を返す)。平時の借入 = (1 − f_R) δ / amort × 資本の価値')
    p.add('mu_prem', 0.0, 'behavior', '貸し手の慎重さ: 所得の集中 Φ が θ を超えた分に掛ける係数(ρ_eff の形)')
    p.add('theta_prem', 0.3, 'behavior', '貸し手の慎重さの閾値(上位 10% の所得の取り分)')
    p.add('phi_fixed', -1.0, 'behavior', '貸し手が見る集中 Φ を固定する(負なら、前の年の標本の値。フィードバックを切る比較用)')
    p.add('tau_k', -1.0, 'policy', '資本所得(配当・銀行の利益)への税率。負なら賃金と同じ τ')
    p.add('redistrib', 0.0, 'policy', '資本所得の税収のうち、一人あたり同額で配る割合(再分配)')
    # 土地と知財(段 2)。land = False なら、すべて切る(段 1 と同じ数)
    p.add('land', False, 'structure', '土地と知財の市場を入れるか')
    p.add('T_land', 1000.0, 'structure', '土地の総量')
    p.add('rho_T', 0.1, 'structure', '地代: 必需品一単位あたり ρ_T × 必需品の価格(土地は必需品の投入)')
    p.add('phi_R', 0.1, 'behavior', '知財の使用料: 贅沢品の企業の利潤のこの割合を、知財の持ち主に払う')
    p.add('sigma_R', 0.1, 'behavior', '贅沢品の消費のうち、自主研究(知財を生む)の割合')
    p.add('eta_IP', 1.0, 'structure', '自主研究の実質の量から知財が生まれる比')
    p.add('delta_IP', 0.1, 'structure', '知財の減耗')
    p.add('theta0', 0.15, 'behavior', '土地に置きたい割合(正味資産 + 稼ぐ力 に対する)の基準')
    p.add('theta1', 3.0, 'behavior', '土地に置きたい割合の、利回りの差(地代利回り + 値上がりの期待 − 貸出の金利)への反応')
    p.add('chi_T', 0.2, 'behavior', '望む保有へ一年に近づく割合')
    p.add('h_y', 10.0, 'behavior', '稼ぐ力: 税引きの労働所得の何年分を、土地を買うときの資産に数えるか')
    p.add('nu_T', 0.2, 'behavior', '土地の値上がりの期待が、実際の値上がりに寄る速さ(外挿)')
    p.add('lam_T', 0.6, 'policy', '土地の担保の掛け目(借入の上限 = 掛け目 × 土地の値打ち + 無担保の上限)')
    p.add('mu_u', 0.5, 'policy', '無担保の借入の上限(一人あたりの賃金の何年分か)')
    p.add('c_h', 0.0, 'behavior', '資産効果: 土地の正味の値打ち(土地 − 借金)のこの割合を消費に足す')
    p.add('breakup_eps', 0.0, 'structure', '一家離散と数える棒引きの下限(その世帯の借入の上限に対する割合)。0 なら丸めの大きさ(1e-14 ほど)の棒引きも数える(ノートの記録はこれで出した)。ブラウザの版は 1e-9')
    p.add('haircut_max', 0.3, 'policy', '貸倒れの損を預金者が負う上限(預金の割合)。超えた分は政府が救済する')
    p.add('turnover', 0.025, 'structure', '世代の入れ替わり: 毎年この割合の世帯が去り、財産を相続させる')
    p.add('inherit', 'lottery', 'policy', "相続: 'lottery'(くじで他の一世帯)か 'equal'(全員に人数で等しく)")
    p.add('theta_shock', 0.0, 'behavior', '土地に置きたい割合への一時の上乗せ(ユーフォリアのシナリオ)')
    # 知財の市場(段 2b)。ip_market = False なら、知財は売り買いされない(段 2 と同じ)
    p.add('ip_market', False, 'structure', '知財の市場を入れるか(土地の市場と同じ仕組み。知財は減耗し、利回りは使用料から)')
    p.add('theta0_IP', 0.05, 'behavior', '知財に置きたい割合(正味資産 + 稼ぐ力 に対する)の基準')
    p.add('theta1_IP', 3.0, 'behavior', '知財に置きたい割合の、利回りの差(使用料の利回り + 値上がりの期待 − 減耗 − 貸出の金利)への反応')
    p.add('chi_IP', 0.2, 'behavior', '知財の望む保有へ一年に近づく割合')
    p.add('ip_pricing', 'demand', 'behavior', "知財の値段: 'demand'(標本の需要で決まる)か 'fundamental'(裁定: 使用料を貸出の金利 + 減耗で割り引いた値打ち。売り買いはしない)")
    p.add('xcap_IP', 0.2, 'behavior', '知財の利回りの差の頭打ち(±この値。土地と同じ 0.2 が既定。大きいほど、利回りに強く反応できる)')
    p.add('nu_IP', 0.2, 'behavior', '知財の値上がりの期待が、実際の値上がりに寄る速さ(外挿)')
    p.add('lam_IP', 0.0, 'policy', '知財の担保の掛け目(既定は担保にならない)')
    # 災害(段 4)。disaster = False なら、すべて切る(段 3 までと同じ数)
    p.add('disaster', False, 'structure', '災害の部品を入れるか')
    p.add('dis_rate', 0.0, 'structure', '大きな災害が一年に起きる確率(0 なら、実験で決めた年にだけ起こす)')
    p.add('dis_q', 0.0, 'structure', '歴史の重点標本化(段 17): 災害を、本当の確率 dis_rate の代わりにこの確率で引き、歴史の重み(尤度の比)を記録する。0 なら使わない')
    p.add('dis_min', 0.05, 'structure', '災害の大きさ(資本が壊れる割合)の下限。大きさはパレート型の裾を持つ')
    p.add('dis_alpha', 1.5, 'structure', '災害の大きさの裾の指数(小さいほど、巨大な災害が起きやすい)')
    p.add('dis_max', 0.8, 'structure', '災害の大きさの上限')
    p.add('dis_scope', 'common', 'structure', "打撃の相関: 'common'(全員が少しずつ)か 'idio'(一部の世帯だけが大きく。期待の損は同じ)")
    p.add('hit_share', 0.1, 'structure', "'idio' のとき、打撃を受ける世帯の割合(人数で)")
    p.add('inj_coef', 0.5, 'structure', '災害の大きさ 1 あたり、働く力を失う割合(怪我)')
    p.add('rec_inj', 0.3, 'structure', '怪我から一年に戻る割合')
    p.add('protect', 0.0, 'policy', '守り: 政府が前の年の所得のこの割合で人を雇い、守りの資本を積む')
    p.add('prot_scale', 200.0, 'structure', '守りの効き目の尺度: 被害 = 大きさ × exp(− 守りの資本 / 尺度)')
    p.add('prot_decay', 0.05, 'structure', '守りの資本の減耗')
    p.add('relief', 'none', 'policy', "救済: 'none'、'state'(政府が払い、国債で賄う)、'mutual'(打撃を受けなかった世帯が預金に応じて払う)")
    p.add('relief_rate', 0.5, 'policy', '救済が埋める、失われる労働所得(怪我が戻るまでの見込み)の割合')
    # 財政の行き詰まり(段 4b)。既定ではどれも切ってある
    p.add('mu_B', 0.0, 'behavior', '国債の上乗せ金利: 国債/所得が θ_B を超えた分に掛ける係数(0 なら上乗せなし)')
    p.add('theta_B', 1.0, 'behavior', '国債の上乗せ金利がつき始める国債/所得')
    p.add('burden_target', 0.016, 'policy', "財政のルール 'burden': 目標の利払いの重さ(国債の金利 × 国債 / 所得)。許される赤字/所得 = 目標 − 今の重さ")
    p.add('resolve', 'none', 'policy', "国債/所得が θ_resolve を超えたときの道: 'none'(何もしない)、'levy'(富裕税)、'austerity'(賃金への税の上乗せ)、"
          "'default'(国債の元本を削る。損は預金者へ)、'repression'(上乗せ金利を払わない)")
    p.add('theta_resolve', 1.0, 'policy', '行き詰まりの道を取り始める国債/所得')
    p.add('levy_rate', 0.02, 'policy', '富裕税: 一人あたりの正味資産が免除の額を超えた分に、毎年かける率')
    p.add('wealth_tax', 0.0, 'policy', '恒常の富裕税の率(毎年。免除の額は levy_exempt。行き詰まりの道の levy とは別)')
    p.add('enf_F', 1.0, 'policy', '外の資産のうち、税の当局が見える(富裕税と資本所得の税がかかる)割合')
    # 資本の移動(段 6。地域の模型 im3 の中でだけ使う)
    p.add('theta_F0', 0.0, 'behavior', '正味資産のうち外の資産に置きたい割合の基準')
    p.add('phi_K', 0.0, 'behavior', '外に置きたい割合の、預金の金利の差(外 − 内)への反応')
    p.add('phi_L', 0.0, 'behavior', '外に置きたい割合の、逃れられる富裕税の率(率 × 見えない割合)への反応')
    p.add('chi_F', 0.05, 'behavior', '外の資産の望む保有へ一年に近づく割合(速いと変動の為替が暴れる。段 6 の K0)')
    p.add('theta_F_max', 0.9, 'behavior', '外に置きたい割合の上限')
    p.add('levy_exempt', 3.0, 'policy', '富裕税の免除の額(一人あたりの正味資産の平均の何倍か)')
    p.add('austerity', 0.05, 'policy', '緊縮: 賃金への税率への上乗せ')
    p.add('haircut', 0.3, 'policy', '踏み倒し: 国債の元本を削る割合(一度。国債/所得が下がるまで次は起きない)')
    p.add('rbar_link', 1.0, 'behavior', 'r̄ の貸出の金利への連動(1 なら全部、0 なら平時の値に固定)。利下げがマークアップを下げる経路の強さ')
    # 学習(段 5): 災害の頻度を記録から学び、守りの量を決める。learn = False なら守りは protect のまま
    p.add('learn', False, 'structure', '災害の頻度を記録から学び、守りの量を決めるか')
    p.add('learn_omega', 1.0, 'behavior', '記録を一年ごとに割り引く割合(1 なら忘れない。0.95 なら実効の記録は 20 年ほど)')
    p.add('prior_a', 0.5, 'behavior', '信念の事前の災害の回数(ガンマ分布の形)')
    p.add('prior_b', 10.0, 'behavior', '信念の事前の年数(ガンマ分布の率)。事前の頻度 = prior_a / prior_b')
    p.add('k_learn', 0.2, 'policy', '守りの量 = min(上限, k × 信じている災害の頻度)')
    p.add('protect_max', 0.05, 'policy', '守りの量の上限(前の年の所得に対する割合)')
    p.add('learn_attitude', 'neutral', 'behavior', "信念の取り方: 'neutral'(平均)、'pessimistic'(上側 90%)、'optimistic'(下側 10%)、'interval'(信念の区間の上の α-maxmin。段 13)")
    p.add('amb_q', 0.05, 'behavior', "信念の区間('interval'): 事後の分布の下側 q から上側 q までの区間")
    p.add('amb_floor', 0.0, 'behavior', "信念の区間の幅の床: 区間は少なくとも平均 × (1 ± 床)(記録が増えても縮みきらない、構造の不確かさ。ナイト的不確実性)")
    p.add('amb_alpha', 1.0, 'behavior', "区間の上の決め方: 頻度 = (1 − α) × 下の端 + α × 上の端(α = 1 なら最悪に備える maxmin、0.5 ならハーヴィッツの真ん中)")
    # 政治(段 5): 資本所得への税を投票で決める。vote = False なら tau_k は外から与えたまま
    p.add('vote', False, 'structure', '資本所得への税を投票で決めるか(税収は一人あたり同じ額で配る)')
    p.add('vote_every', 10, 'structure', '投票の間隔(年)')
    p.add('tau_lo', 0.1, 'policy', '資本所得への税の低い案(資産家が好む)')
    p.add('tau_hi', 0.5, 'policy', '資本所得への税の高い案(労働者が好む)')
    p.add('rho_w', 1e-3, 'behavior', '労働者の群れの中の、投票の決定の相関(ノート M 第 4 章の ρ)')
    p.add('rho_k', 1e-3, 'behavior', '資産家の群れの中の、投票の決定の相関')
    p.add('c_vote', 0.01, 'behavior', '投票の費用の平均(一人あたりの所得に対する比)。費用は指数分布')
    p.add('vote_noise', False, 'structure', '票の揺れ: 群れの票の数に、群れの中の相関による揺れ(標準偏差 = 人数 × √(ρ t (1 − t)))を足す(段 19。ないと票の数が決定論で、差が 0 に張り付く)')
    p.add('rentier_cut', 0.5, 'structure', '資産家とみなす、所得のうち資産所得の割合の境目')
    # 探索(段 5): 企業の研究。research = False なら研究しない
    p.add('research', False, 'structure', '企業が研究する(探索)か。当たると最先端の技術が下がり、新しい生産能力に乗る')
    p.add('s_R', 0.02, 'behavior', '研究に使う、売上の割合(学ばないとき)')
    p.add('trial_cost', 1.0, 'structure', '研究の試行一回に要る労働')
    p.add('p_hit', 0.05, 'structure', '研究の試行が当たる確率')
    p.add('g0', 0.002, 'structure', '当たったときの技術の向上(対数)の下限。大きさはパレート型')
    p.add('alpha_R', 1.5, 'structure', '当たりの大きさの裾の指数(小さいほど、巨大な当たりが出やすい。1 以下なら平均がない)')
    p.add('g_cap', 0.3, 'structure', '一回の当たりの大きさの上限(対数)')
    p.add('cap_R', 0.5, 'behavior', '研究に使う額の上限: 去年の利益(利子を払ったあと、研究の前)の割合。赤字の企業は研究しない')
    p.add('adaptive_R', False, 'behavior', '研究の割合を、過去の当たりから学んで決めるか')
    p.add('omega_R', 0.9, 'behavior', '研究の当たりの記録を、一年ごとに割り引く割合')
    p.add('h_R', 5.0, 'behavior', '研究の見返りを数える年数。見込みの比 b = 労働一単位あたりの当たりの見込み × 生産の労働 × h_R')
    p.add('s_R_min', 0.0, 'behavior', '学ぶときも下回らない研究の割合(探索の床)')
    p.add('s_R_max', 0.1, 'behavior', '研究の割合の上限。学ぶときの割合 = s_R_max × b / (1 + b)')
    # 死蔵と準備率(段 16)。hoard = False なら死蔵しない
    p.add('hoard', False, 'structure', '富む家計が、預金の一部を死蔵する(銀行の外の現金として持つ。利子がつかず、銀行の準備を吸い取る。使うことはできる)。ノート 3 の S9・S10、v6 の ρ_eff の仕組みの版')
    p.add('hoard_cap', 5.0, 'structure', '死蔵の始まる預金: 一人あたりの預金の平均のこの倍を超えた分(ノート 3 の「上限まで埋まった種類の次」)')
    p.add('hoard_frac', 1.0, 'structure', '上限を超えた預金のうち、死蔵する割合')
    p.add('reserves', 'endog', 'structure', "準備の供給: 'endog'(中央銀行が求められるだけ出す。今の im)か 'fixed'(準備の量が所得の res_supply 倍に決まっている)")
    p.add('rho_res', 0.1, 'policy', '準備率')
    p.add('res_supply', 0.1, 'policy', "'fixed' のときの準備の供給(去年の所得に対する比)。死蔵は準備を吸い取る(現金の流出)")
    # 人間の有限の注意(段 12)。attention = 1 なら、毎年すべての家計が決め直す
    p.add('attention', 1.0, 'behavior', '一年に決め直す家計の割合(記入は世帯の群れなので、群れの中のこの割合が決め直す。'
          '予算は (1 − 割合) が去年の実質の予算のまま、資産の配分(土地・株・知財・外の資産)は望む保有へ近づく速さがこの割合だけ遅くなる)')
    # 自動化(段 11)。auto_share = 0 なら研究の当たりはすべて労働を補う技術
    p.add('auto_share', 0.0, 'structure', '研究の当たりのうち、仕事を置き換える(自動化の)割合。労働の係数を下げる代わりに、資本の係数を上げる')
    p.add('auto_kappa', 1.0, 'structure', '自動化の当たり g(対数)ごとに、新しい能力一単位の資本の係数が exp(auto_kappa × g) 倍になる')
    # 年齢と年金と BI(段 10)。どれも既定では切ってある
    p.add('age_track', False, 'structure', '引退がなくても世帯の年齢を数える(ブラウザの版: 途中で年金を入れたとき、すぐに引退する人がいるように)')
    p.add('ret_age', 0, 'structure', '引退の年齢(世帯が始まってからの年数)。0 なら引退はない。引退した世帯は働かない')
    p.add('pension', 'none', 'policy', "年金: 'none'、'payg'(賦課方式: 政府が引退した人に一人あたり同額を払う)、'points'(貢献に比例: 同じ総額を、働いた年の労働所得の累計に比例して配る)、'funded'(積立: 働く世帯が労働所得の一部を積み立て、引退してから取り崩す)")
    p.add('pen_rate', 0.4, 'policy', '賦課方式の年金の額: 働く人一人あたりの賃金の割合')
    p.add('fund_rate', 0.1, 'policy', '積立方式の積立の割合(労働所得の)')
    p.add('fund_draw', 0.1, 'policy', '積立の取り崩しの割合(引退した世帯の、一年の)')
    p.add('fund_eq', 0.0, 'policy', "積立の基金が株(持分)に置く割合(段 10b。pension = 'funded' と eq_market = True のとき)。0 なら積立はすべて預金")
    p.add('phi_q', 0.0, 'behavior', '投資の、株の値段の基本の値打ちからの開き(対数、トービンの q の代わり)への反応(段 10b)。株が高いと、持分で安く賄えるので投資が増える')
    p.add('child_allow', 0.0, 'policy', '児童手当: 子を育てる世帯に、一人あたり必需品の生存の量の値段のこの倍を払う')
    p.add('vat_bi', False, 'policy', '一人あたり同額の給付(g_tr、BI)を、消費税で賄う(消費税の定額還付の形)。False なら国債と財政のルールで賄う')
    # 家計の大バクチと養育の費用(段 9)。どれも既定では切ってある
    p.add('luck_sigma', 0.0, 'structure', '家計ごとの資産の運: 持分の取り分の対数が毎年この標準偏差で揺れる(和は 1 のまま。記入の人数が多いほど小さく)')
    p.add('luck_nu', 0.0, 'structure', '運の裾: 0 なら正規、2 より大きければ自由度 ν の t 分布(分散 1 にそろえる。裾が重い = 大バクチ)')
    p.add('luck_rho', 0.0, 'structure', '運用の腕(持続する利回りの差)の、一年の持続の割合。0 なら腕の差はない')
    p.add('luck_skill', 0.0, 'structure', '運用の腕の標準偏差(持分の取り分の対数の伸びの差、定常の値)')
    p.add('eq_init', 'pareto', 'structure', "初めの持分の分布: 'pareto'(裾の指数 1.5)か 'equal'(全員が同じ。格差が内から出るかを見る)")
    p.add('child_cost', 0.0, 'structure', '養育の費用: 子を育てる世帯の生存の量は、一人 q0 × (1 + child_cost)')
    p.add('child_years', 20, 'structure', '子を育てる年数(世代の入れ替わりで新しく始まる世帯が、その年数だけ育てる)')
    # 文化と多様性(段 8)。psi_v = 0 なら贅沢品は一種類
    p.add('psi_v', 0.0, 'behavior', '文化の効果 ψ: 贅沢品の種類の数 n = (知財の総量 / ip_ref)^ψ')
    p.add('culture_spill', 0.0, 'structure', '文化から技術への波及(段 8b): 研究の試行が当たる確率 × n^culture_spill(種類の数 n が多いほど、研究が当たりやすい。ノート I の「贅沢品 = 探索」)')
    p.add('spill_cap', 3.0, 'structure', '文化から技術への波及の倍率の上限(n^culture_spill をここで頭打ちにする。上限なしの冪では、当たり → 知財 → 種類 → 当たりの帰還で成長が発散する)')
    p.add('ip_ref', 400.0, 'structure', '種類の数の基準の知財の総量(基準の経済の定常の値。n = 1)')
    p.add('sigma_v', 3.0, 'behavior', '贅沢品の種類の間の代わりの利きやすさ(ディクシット–スティグリッツ)。実効の価格 = 価格 × n^(−1/(σ−1))')
    p.add('eps_v', 1.0, 'behavior', '生存量を超えた支出の、必需品と贅沢品(実効の価格)の間の代わりの利きやすさ。1 なら割合は b_N のまま(種類は効用だけに効く)')
    # 危機時の生産(段 7)。どれも既定では切ってある(一国の模型 Im1 だけで使う)
    p.add('crisis_track', False, 'structure', '段 7 の年の中の計算の道を使う(部品を入れなくても、生存の量に届かない人などを記録する。動きは同じ)')
    p.add('inventory', False, 'structure', '必需品の企業が在庫を持つか(生産と販売の差が在庫になる)')
    p.add('inv_target', 0.1, 'behavior', '在庫の目標(一年の販売に対する割合。0.1 は約 5 週)')
    p.add('chi_inv', 0.5, 'behavior', '在庫の目標からのずれを、一年に生産で埋める割合')
    p.add('zeta_inv', 0.3, 'behavior', '在庫が目標を下回るほど、値上げ幅を上げる強さ')
    p.add('cap_N', 0.0, 'policy', '必需品の価格の上限(名目。0 なら上限なし)。上限が効くと、企業は変動費が上限に届くところまでしか作らない')
    p.add('crisis_rule', 'none', 'policy', "必需品の企業の赤字の扱い: 'none'、'last'(最後の生産者: 政府が赤字を埋め、上限のもとでも生存の量(一人 q0)までは作らせ、能力を捨てさせない)、'share'(損失の分担: 贅沢品と原料の企業が売上に応じて埋める。元の micro_economy の規則)")
    p.add('g_stock', 0.0, 'policy', '政府の必需品の備蓄の目標(一年の家計の必需品の消費に対する割合)')
    p.add('chi_S', 0.2, 'policy', '備蓄の目標との差を、一年に買い足す割合')
    p.add('delta_S', 0.05, 'structure', '備蓄の傷み(一年)')
    p.add('g_pw', 0.0, 'policy', '公共事業: 政府が去年の所得のこの割合で、投資財を買う')
    p.add('g_tr', 0.0, 'policy', '一人あたり同額の給付: 去年の所得のこの割合(公共事業と同じ額で比べるため)')
    p.add('f_K', 0.0, 'structure', '生産能力の固定の維持費(能力一単位あたりの労働。稼働していなくても要る)')
    p.add('delta_scrap', 0.0, 'structure', '赤字の部門の能力の廃棄の率(利益の率が −0.1 で、この率だけ余分に捨てる)')
    # 部門の中の企業の集まり(段 5 の世界樹の案。im_firms.py)。firms = 0 なら、部門に企業は一つ
    p.add('firms', 0, 'structure', '部門ごとの企業の数(0 なら一つ。研究 research = True のときだけ使う)')
    p.add('rule_share0', 0.5, 'structure', '初めの、規則的に研究する企業の割合(固定の参入のときは、参入の型の割合)')
    p.add('s_L', 0.0, 'behavior', '学習する企業が自分の研究に使う、売上の割合')
    p.add('iota_L', 0.3, 'behavior', '学習する企業が、見える最先端との差を一年に取り込む割合(真似)')
    p.add('iota_R', 0.05, 'behavior', '規則的な企業が、見える最先端との差を一年に取り込む割合')
    p.add('embody', 0.15, 'structure', '知識が生産に乗る速さ(投資の速さ)')
    p.add('beta_m', 5.0, 'behavior', '占有率の、乗った知識の平均からのずれへの反応(費用が低い企業が客を取る)')
    p.add('omega_x', 0.8, 'behavior', '企業の利益の率の記憶の割り引き')
    p.add('r_exit', -0.05, 'structure', '利益の率の記憶がこれを下回ると退出')
    p.add('s_min', 1e-3, 'structure', '占有率がこれを下回ると退出')
    p.add('s_entry', 0.01, 'structure', '参入する企業の初めの占有率')
    p.add('entry', 'select', 'structure', "参入の型: 'select'(型ごとの利益で選ぶ。突然変異 mut)か 'fixed'(rule_share0 の割合)")
    p.add('mut', 0.05, 'structure', '参入の型の突然変異の割合')
    p.add('ip_lag', 0, 'policy', '知財の保護: 他社の知識は、この年数だけ前のものしか真似られない')
    # 段 15b: ノート M 第 3 章(縁故主義)の形の貸し手。投資の事業の成功の確率に外部性、市場型は広く雑に、関係融資は狭く確かに貸す
    p.add('lending', 'none', 'structure', "投資の事業への貸し手(段 15b): 'none'、'market'(市場型)、'relation'(関係融資)")
    p.add('p0_s', 0.75, 'structure', '投資の事業が成功する確率の基準(成功しない事業の能力は失われ、借りた分の一部が貸倒れる)')
    p.add('kappa_x', 0.3, 'structure', '投資の外部性: 成功の確率 = p0_s + kappa_x × (去年の投資 / 置き換えの投資 − 1)(みなが投資すると、事業は成功しやすい)')
    p.add('p_thr', 0.65, 'behavior', '市場型の貸し手が貸す成功の確率の目安')
    p.add('h_caut', 0.0, 'behavior', '市場型の貸し手の慎重さ(目安に上乗せ。ノート M の h)')
    p.add('sig_s', 0.05, 'behavior', '市場型の貸し手の見積りの揺れ(承認の割合 = sigmoid((成功の確率 − 目安 − 慎重さ) / sig_s))')
    p.add('g_rel', 0.6, 'behavior', '関係融資が見分けて貸す事業の割合(良い事業だけ。狭い)')
    p.add('p_good', 0.95, 'structure', '関係融資が貸す事業の成功の確率')
    p.add('lgd_p', 0.5, 'structure', '失敗した事業への貸出のうち失われる割合')
    # 段 15c: 貸し手と探索、株式的富のペテン
    p.add('res_loan', 0.0, 'structure', '研究(探索)の額のうち、借りて賄う割合(段 15c)。貸し手が承認した割合だけ実行する')
    p.add('g_rel_x', 0.2, 'behavior', '関係融資が、実績のない探索の事業に貸す割合(見分ける手がかりがない。狭い)')
    p.add('phi_sw', 0.0, 'behavior', '株式的富のペテン(段 15c): 市場型の貸し手は、株の値段の基本の値打ちからの開き(対数)を、事業が成功する印として読み、見積りに phi_sw × 開き × 信頼 を足す。関係融資は株の値段を見ない')
    p.add('k_mem', 0.0, 'behavior', '株式的富への信頼 = exp(−k_mem × 貸倒れの記憶/所得)。家計が株に置きたい割合と、市場型の貸し手の読みに掛かる(分配的損失の記憶。JRF 2026-04-02)')
    p.add('omega_mem', 0.9, 'behavior', '貸倒れの記憶を一年ごとに割り引く割合')
    # 関係融資と市場型融資(段 15。企業の集まり firms > 0 のときだけ)
    p.add('lender', 'none', 'structure', "企業への貸し手の型: 'none'(企業の質を入れない)、'blind'(悪い企業はいるが、どの企業にも同じく貸す)、'relation'(関係融資: 質を知る)、'market'(市場型: 雑音のある信号、景気で基準を上げる)、'mix'(企業ごとに rel_share の確率で関係融資)")
    p.add('rel_share', 0.5, 'structure', "'mix' のとき、関係融資の銀行と付き合う企業の割合")
    p.add('p_bad', 0.3, 'structure', '悪い企業の割合(参入のとき)')
    p.add('bad_drift', 0.001, 'structure', '悪い企業の知識の一年の目減り')
    p.add('d_bad', 0.1, 'structure', '悪い企業の一年の倒産の確率(借りている信用の取り分に比例)')
    p.add('lgd', 0.5, 'structure', '倒産した企業への貸出のうち、失われる割合')
    p.add('zombie', 0.8, 'behavior', '関係融資の銀行が、悪い企業に貸し続ける信用の取り分(ゾンビ融資)。追い貸しで、悪い企業の倒産もこの割合だけ先送りする')
    p.add('sig_noise', 1.0, 'behavior', '市場型の貸し手が見る、企業の質の信号の雑音')
    p.add('thr0', 0.0, 'behavior', '市場型の貸し手の基準(信号)')
    p.add('crunch', 1.0, 'behavior', '市場型の貸し手が、部門の利益の率が平常を 5% 下回るごとに上げる基準')
    p.add('beta_c', 1.0, 'behavior', '占有率の、信用の取り分への反応(借りられない企業は能力を増やせない)')
    p.add('p_spin', 0.0, 'structure', '作り手の流出: 毎年、企業ごとに、研究者が知識を持って独立し新しい企業を起こす確率(段 14)')
    p.add('t_prize', 0.0, 'policy', '国庫の案: 部門の利益(正の分)にこの率で課し、その年の研究の当たりの大きさに比例して企業に配る(段 14。2021〜22 年の作者の構想)')
    p.add('prize_reinvest', True, 'behavior', '国庫の報酬を、次の年の研究に回すか')
    p.add('mu_hhi', 0.0, 'structure', '集中の値上げ: 部門の利益の率の目標 = r̄ + mu_hhi × (HHI − 1/企業の数)(集中した部門は、競争で利益が r̄ まで下がらない)')
    p.add('s_pub', 0.0, 'policy', '公の研究所: 政府が去年の所得のこの割合で研究の人を雇う。当たりはどの企業も遅れなく真似られる')
    # 株式の市場(段 5)。eq_market = False なら、持分は売り買いされない
    p.add('eq_market', False, 'structure', '持分(株式)の市場を入れるか(土地の市場と同じ仕組み)')
    p.add('eq_pricing', 'fundamental', 'structure', "株の値段の決め方: 'fundamental'(均した配当 ÷ r̄。市場は持ち主を入れ替えるだけ)、'demand'(標本の望む保有の和 = 1 となる値段)")
    p.add('theta0_E', 0.12, 'behavior', '株式に置きたい割合(正味資産 + 稼ぐ力 に対する)の基準')
    p.add('theta1_E', 3.0, 'behavior', '株式に置きたい割合の、利回りの差(配当の利回り + 値上がりの期待 − 貸出の金利)への反応')
    p.add('chi_E', 0.2, 'behavior', '株式の望む保有へ一年に近づく割合')
    p.add('human_E', 0.0, 'behavior', '株式に置きたい額を決める元に、稼ぐ力(人の値打ち)をどれだけ入れるか。0 なら正味資産だけ(持ち主が偏る)、1 なら土地と同じ')
    p.add('kappa_E', 0.5, 'behavior', '株式に置きたい割合の、一人あたりの正味資産(平均との比)への弾力性。富むほど株に置く割合が高い(Wachter-Yogo)')
    p.add('lam_E', 0.0, 'structure', '株を買うのに借りられる割合(信用取引)。0 なら手元の預金でしか買えない')
    p.add('nu_E', 0.2, 'behavior', '株式の値上がりの期待が、実際の値上がりに寄る速さ(外挿)')
    # 個人として生きる箱庭(段 19)。player = False なら、プレイヤーの記入はない
    p.add('player', False, 'structure', '標本の中に、プレイヤーの記入(一人の世帯)を切り出す。規則の代わりに、外から選び方を与えられる(e.act)')
    p.add('player_src', 0, 'structure', 'プレイヤーの記入を切り出す元の記入の番号(同じ割合の財産と働く力を、人数に比例して分ける)')
    p.add('player_heads', 0.01, 'structure', 'プレイヤーの記入の人数(im の人数の単位は全体で約 1000。一人の世帯はその中のごく小さな重み)')
    p.add('player_heir', 'self', 'structure', "プレイヤーの世帯が入れ替わるとき: 'self'(自分の次の代が財産を継ぎ、遊びを続ける。ワールドネバーランドの代替わり)か 'rule'(ほかの世帯と同じく、くじで他の世帯へ)")
    p.add('k_sway', 1e-3, 'behavior', '一人の見本の効き目: プレイヤーが呼びかけ(campaign、0〜1)をすると、自分の群れの投票の決定の相関 ρ に k_sway × 呼びかけ を足す(ノート M 第 4 章のダウンズへの反論)')
    # 遊びの出来事(ブラウザの版の段 B7)。どちらも既定では切ってある
    p.add('revolt', False, 'structure', '体制の交代(革命): 移り変わりの痛み(雇用の急な落ち込み、実質賃金の下がり、生存の量に届かない人の増え、一家離散)が積もった不満が閾値を超えると、政策の組(資本所得の税、再分配、富裕税)が、もう一つの体制の組と入れ替わる(ノート 4 第 15.2 節の案)')
    p.add('rev_wE', 3.0, 'behavior', '不満: 雇用の一年の落ち込みの割合の重み')
    p.add('rev_ww', 3.0, 'behavior', '不満: 実質賃金の一年の下がり(対数)の重み')
    p.add('rev_wh', 2.0, 'behavior', '不満: 生存の量に届かない人の割合の増えの重み(危機時の生産の道で数えるとき)')
    p.add('rev_wb', 0.02, 'behavior', '不満: 一家離散の人数の重み')
    p.add('rev_mem', 0.8, 'behavior', '不満の記憶(一年ごとに残る割合)')
    p.add('rev_thr', 1.0, 'behavior', '革命の閾値(積もった不満。既定の世界で、景気の波だけなら 200 年に 2 回ほど、大きな災害が年 5% で起きれば 7 回ほど)')
    p.add('rev_cool', 15, 'behavior', '革命のあと、次が起きうるまでの年数')
    p.add('rev_tau_k', 0.5, 'policy', '新しい体制の資本所得の税率(初めの革命で入る組)')
    p.add('rev_redistrib', 1.0, 'policy', '新しい体制の、資本の税の再分配の割合')
    p.add('rev_wealth_tax', 0.01, 'policy', '新しい体制の富裕税')
    p.add('up_gen', 0, 'structure', '格上げ(新しい財の工房、ノート 4 第 15 章の upgrade.py を小さくしたもの): 0 なら切る。値が変わった年に、工房を割合 up_s0 で開き直す')
    p.add('up_s0', 0.02, 'structure', '開いたときの工房の割合(贅沢品のうち)')
    p.add('up_g0', 0.6, 'structure', '開いたときの工房の一製品あたり労働(古い贅沢品は 0.3 から)')
    p.add('up_floor', 0.1, 'structure', '工房の一製品あたり労働の下限')
    p.add('up_wright', 0.3, 'structure', 'ライトの法則の指数(作った量の累計が倍になると、労働が 2^(−指数) 倍に)')
    p.add('up_cum0', 50.0, 'structure', '開いたときの、作った量の累計(ライトの法則の基準)')
    p.add('up_speed', 0.5, 'behavior', '工房の割合が、費用の差(対数)に応じて広がる・縮む速さ')
    p.add('up_sub', 0.0, 'policy', '工房への補助: 工房の労働の費用のこの割合を政府が埋める(割合の広がりは、補助のあとの費用で決まる)')
    # 重点標本化(分割と併合)。is_split = 0 なら使わない
    p.add('is_split', 0, 'structure', '重点標本化: 毎年、借金の上限に近い記入をいくつ二つに分けるか(同じ数だけ、似た安全な記入を併せる)')
    p.add('is_risk', 0.5, 'structure', '分ける記入の条件: 借金 / 借入の上限 がこの値以上')
    p.add('is_min_heads', 0.1, 'structure', '分けたあとの記入の人数の下限')
    p.add('is_max_ratio', 3.0, 'structure', '併せたあとの記入の人数の上限(平均の何倍か)。重みが際限なく大きくならないように')
    p.add('is_mode', 'average', 'structure', "併せ方: 'roulette'(どちらか一方を人数に比例した確率で残し、重みを二つ分に。総量のずれは全員に配って直す)か 'average'(二つを足して平均する)")
    return p


class Households:
    """重みつきの標本。一つの記入が、heads 人の世帯の集まりを代表する(量はその集まりの合計)。
    計算する記入の全員が標本で、重み(heads)を変えれば、危ない層を多めにとる重点標本化もできる。
    lab: 労働の取り分(Σ = 1。賃金と失業を、この割合で分ける)、eq: 持分の取り分(Σ = 1。配当を分ける)。
    消費性向は、propensity = 'fixed' なら与えた値、'continuous' なら前の年の所得のうち資産所得の割合 a で
    労働者の値(a = 0)と資本家の値(a = 1)の間を連続に動く(階級を連続の分布として持つ。JRF)。"""

    def __init__(self, M, Yd, heads, lab, eq, c_y, c_m, propensity='fixed', cw=(0.9, 0.2), cc=(0.5, 0.05)):
        self.M = np.array(M, float)            # 預金
        self.Yd = np.array(Yd, float)          # 前の年の可処分所得
        self.heads = np.array(heads, float)
        self.lab = np.array(lab, float)
        self.eq = np.array(eq, float)
        self.c_y = np.array(c_y, float)
        self.c_m = np.array(c_m, float)
        self.propensity = propensity
        self.cw, self.cc = cw, cc
        self.asset_share = np.where(self.eq > 0, 1.0, 0.0) if propensity == 'fixed' else self.eq * 0.0
        self.land = np.zeros_like(self.M)
        self.ip = np.zeros_like(self.M)
        self.inj = np.zeros_like(self.M)          # 怪我で失った働く力の割合(段 4)
        self.extra = np.zeros_like(self.M)        # 資産効果など、予算への上乗せ(段 2)
        self.F = np.zeros_like(self.M)            # 外の資産(自分の通貨で測った値打ち。段 6)
        self.kids = np.zeros_like(self.M)         # 子を育てる残りの年数(段 9)
        self.skill = np.zeros_like(self.M)        # 運用の腕(段 9)
        self.age = np.zeros_like(self.M)          # 世帯の年齢(段 10)
        self.pf = np.zeros_like(self.M)           # 積立の年金の残高(預金のうち、引き出せない分。段 10)
        self.pts = np.zeros_like(self.M)          # 年金の貢献の累計(段 10)
        self.hd = np.zeros_like(self.M)           # 死蔵している預金(使わない。段 16)

    def budgets(self):
        if self.propensity == 'continuous':
            a = np.clip(self.asset_share, 0.0, 1.0)
            self.c_y = self.cw[0] + (self.cc[0] - self.cw[0]) * a
            self.c_m = self.cw[1] + (self.cc[1] - self.cw[1]) * a
        B = self.c_y * np.maximum(self.Yd, 0.0) + self.c_m * np.maximum(self.M + self.F, 0.0) + self.extra   # 外の資産も、使える財産に数える(段 6)
        return np.maximum(B, 0.0)

    def split(self, k):
        """どの記入も k 個の同じ記入に分ける(重みは 1/k)。総量は変わらない(重みの付け方によらないことのテスト)。"""
        for name in ('M', 'Yd', 'heads', 'lab', 'eq', 'c_y', 'c_m', 'asset_share', 'land', 'ip', 'extra', 'inj', 'F', 'kids', 'skill', 'age', 'pf', 'pts', 'hd'):
            v = getattr(self, name)
            rep_ = np.repeat(v, k)
            setattr(self, name, rep_ / k if name in ('M', 'Yd', 'heads', 'lab', 'eq', 'land', 'ip', 'extra', 'F', 'pf', 'pts', 'hd') else rep_)
        return self


def make_households(P, kind='two', Ns=500, seed=0, eq_tail=1.5):
    """kind = 'two': me_c と同じ二つの型。kind = 'continuous': Ns 世帯の連続の分布。
    持分はパレート型の裾(指数 eq_tail)、労働の取り分は持分の大きい世帯ほど小さい。初めの預金は、
    二つの型の版と同じ総額(労働者 3000、資本家 1000)を、労働と持分の取り分で配る。"""
    if kind == 'two':
        return Households(M=[3000.0, 1000.0], Yd=[4500.0, 500.0], heads=[P.pop_w, P.n_cap],
                          lab=[1.0, 0.0], eq=[0.0, 1.0], c_y=[P.c_y, P.c_yc], c_m=[P.c_m, P.c_mc])
    rng = make_rng(getattr(P, 'rng', 'numpy'), seed)
    heads = np.full(Ns, (P.pop_w + P.n_cap) / Ns)
    eq = rng.pareto(eq_tail, Ns) + 1e-3
    if getattr(P, 'eq_init', 'pareto') == 'equal':
        eq = heads.copy()
    eq = eq / eq.sum()
    lab = heads * np.exp(-30.0 * eq)                 # 持分の大きい世帯(資産家)は働かない
    lab = lab / lab.sum()
    M = 3000.0 * lab + 1000.0 * eq
    Yd = 4500.0 * lab + 500.0 * eq
    H = Households(M=M, Yd=Yd, heads=heads, lab=lab, eq=eq, c_y=np.full(Ns, P.c_y), c_m=np.full(Ns, P.c_m),
                   propensity='continuous', cw=(P.c_y, P.c_m), cc=(P.c_yc, P.c_mc))
    H.asset_share = 500.0 * eq / Yd
    return H


class Im1:
    def __init__(self, p=None, households='two', Ns=500, seed=0, split=1, ledger=None, region='', extra_sectors=(), **kw):
        self.p = p or default_params()
        self.p.set(**kw)
        P = self.p
        sfx = '@' + region if region else ''
        self.sfx = sfx
        self.region = region
        self.sec = {k: k + sfx for k in tuple(SECTORS) + tuple(extra_sectors)}
        self.firms = [f + sfx for f in FIRMS]
        self.real = [r + sfx for r in REAL]
        standalone = ledger is None
        self.L = Ledger(list(self.sec.values()), INSTRUMENTS, real=self.real) if standalone else ledger
        self.K = np.array([1250.0, 400.0, 1100.0])
        self.gamma = P.gamma0.copy()
        self.gamma_new = P.gamma0.copy()
        self.m0 = P.m0.copy()
        self.W = 10.0
        self.prices = np.array([3.0, 8.0, 6.0])
        self.PN_prev = self.prices[0]
        self.tau = P.tau0
        self.loans = np.zeros(3)
        self.top10_prev = 0.0
        self.set_rates()
        self.rho_prev = np.full(3, self.rbar)
        self.u_prev = np.full(3, P.u_star)
        self.profit_prev = 0.0
        # 家計: 労働者と資本家(me_c の初期値)
        self.H = make_households(P, households, Ns, seed)
        if split > 1:
            self.H.split(split)
        # 期首の残高: 家計の預金 4000、銀行の国債 4000(政府の負債。me_c の「社会的負債」)
        L = self.L
        hh, bank, gov = self.sec['hh'], self.sec['bank'], self.sec['gov']
        L.revalue('deposits', hh, 4000.0); L.revalue('deposits', bank, -4000.0)
        L.revalue('bonds', bank, 4000.0); L.revalue('bonds', gov, -4000.0)
        for j, r in enumerate(self.real[:3]):
            L.revalue(r, self.firms[j], self.K[j])
        if P.credit:
            # 初めの借入を平時の水準に置く。借りたお金は、過去に家計の預金になったとする(労働と持分の取り分で配る)
            L0 = (1 - P.f_R) * P.delta / P.amort * self.prices[2] * P.kappa * self.K
            for j in range(3):
                L.revalue('loans', bank, L0[j]); L.revalue('loans', self.firms[j], -L0[j])
            L.revalue('deposits', hh, L0.sum()); L.revalue('deposits', bank, -L0.sum())
            share = 0.5 * self.H.lab + 0.5 * self.H.eq
            self.H.M = self.H.M + L0.sum() * share / share.sum()
            self.loans = L0.copy()
        if standalone:
            L.close(); L.t = 0
        self.rng = make_rng(P.rng, seed + 1)
        self.pT, self.gT_e, self.wo_h = 1.0, 0.0, 0.0
        self.qIP, self.gIP_e, self.ip_hold = None, 0.0, None
        self.prot, self.Y_last, self.shock_queue, self.dis_rec = 0.0, 0.0, [], {}
        self.v_eq_last = 0.0
        if P.ret_age > 0 or P.age_track:
            self.H.age = make_rng(P.rng, seed + 11).exponential(1.0 / max(P.turnover, 1e-3), len(self.H.M))   # 定常の年齢の分布
        self.kap_new = np.array(P.kappa, float).copy()   # 新しい能力一単位の資本の係数(段 11)
        self.kap_avg = np.array(P.kappa, float).copy()   # 据え付けた能力の、ヴィンテージで均した資本の係数
        self.res_rng = make_rng(P.rng, seed + 3)
        self.luck_rng = make_rng(P.rng, seed + 7)
        self.vote_rng = make_rng(P.rng, seed + 13)
        self.xf = {}                        # 地域の模型(im3)から渡す、外との流れ(段 6): dF、incF、int_nr、union_ben、union_in
        self.prof_R_last = None
        self.Y_prev = None
        self.InvN, self.S_G, self.N_last, self._cx = None, 0.0, None, None
        self.fp = None
        if P.firms and P.research:
            from im_firms import FirmPop
            self.fp = FirmPop(P, make_rng(P.rng, seed + 5))
        self.sR = np.full(3, P.s_R)
        self.g_hat = None                                          # 当たりの見込み(初めの年に、研究の割合が s_R になる値で置く)
        self.pE, self.gE_e, self.eq_hold = None, 0.0, None
        self.events_this_year = 0
        if P.vote:
            self.vote_rec = dict(vote_hi=0.0, vote_lo=0.0, turnout_w=0.0, turnout_k=0.0, n_rentier=0.0)
        self.dis_rng = make_rng(P.rng, seed + 2)
        if P.land:
            # 土地は初め、持分と人数の混ぜた割合で持つ。知財は 0 から
            share = 0.5 * self.H.eq + 0.5 * self.H.heads / self.H.heads.sum()
            self.H.land = P.T_land * share / share.sum()
            self.pT = max(P.rho_T * self.prices[0] * 1200.0 / (P.T_land * 0.08), 1.0)   # 地代の利回りがほぼ 8% になる値段から
        self.log = []
        self.t = 0
        self.pl = None
        self.act = {}
        self.fund_D, self.fund_e, self.q_gap = 0.0, 0.0, 0.0     # 段 10b: 積立の基金の預金と持分、株の値段の開き
        self.up_s, self.up_gX, self.up_cum, self.up_gen_seen, self.up_peak = 0.0, P.up_g0, 0.0, 0, 0.0   # 段 B7: 格上げの工房
        self.rev_G, self.regime, self.rev_last, self.rev_prev, self.rev_alt = 0.0, 0, -10 ** 6, None, None   # 段 B7: 革命
        if P.player:
            self.carve_player()

    def trust(self):
        """段 15c: 株式的富への信頼(貸倒れの記憶で下がる)。k_mem = 0 なら 1。"""
        return float(np.exp(-self.p.k_mem * getattr(self, 'loss_mem', 0.0))) if self.p.k_mem else 1.0

    def fund_mode(self):
        """段 10b: 積立の基金が株を持つか。そのとき、積立の残高 H.pf は基金への請求権で、基金の預金は fund_D。"""
        P = self.p
        return P.pension == 'funded' and P.fund_eq > 0 and P.eq_market

    def pf_deposits(self):
        """積立のうち、預金になっている額。"""
        return self.fund_D if self.fund_mode() else float(self.H.pf.sum())

    # ------------------------------------------------ 個人として生きる箱庭(段 19)
    def carve_player(self):
        """記入 player_src から、人数 player_heads の記入を切り出し、標本の最後に足す。足せる量は人数に比例して分け、
        割合の量は写す。総量は変わらない。"""
        P, H = self.p, self.H
        i = int(P.player_src)
        f = min(P.player_heads / H.heads[i], 0.5)
        for name in self.EXT:
            arr = getattr(H, name)
            new = arr[i] * f
            arr[i] = arr[i] * (1 - f)
            setattr(H, name, np.append(arr, new))
        for name in self.INT:
            arr = getattr(H, name)
            setattr(H, name, np.append(arr, arr[i]))
        self.pl = len(H.M) - 1
        self.pl_gen, self.pl_pivots = 1, 0

    def player_state(self):
        """プレイヤーの世帯の今の様子(一人あたりではなく、記入の量。人数 player_heads で割れば一人あたり)。"""
        H, k = self.H, self.pl
        if k is None:
            return {}
        v_eq = self.pE if (self.p.eq_market and self.pE is not None) else self.v_eq_last
        nw = H.M[k] + self.pT * H.land[k] + (self.qIP or 0.0) * H.ip[k] + v_eq * H.eq[k] + H.F[k] + H.pf[k]
        hd = H.heads[k]
        mean_nw = float((H.M + self.pT * H.land + v_eq * H.eq + H.F + H.pf).sum() / H.heads.sum())
        return dict(pl_M=H.M[k] / hd, pl_Yd=H.Yd[k] / hd, pl_nw=nw / hd, pl_nw_rel=nw / hd / max(mean_nw, 1e-12),
                    pl_age=H.age[k], pl_kids=H.kids[k], pl_asset=H.asset_share[k], pl_gen=self.pl_gen, pl_pivots=self.pl_pivots,
                    pl_rentier=bool(H.asset_share[k] > self.p.rentier_cut))

    def set_rates(self):
        """金利と標準の利潤率。貸し手の慎重さ(前の年の所得の集中 Φ による上乗せ)を、貸出の金利に足す。"""
        P = self.p
        if not P.credit:
            self.prem, self.r_loan, self.rbar = 0.0, 0.0, P.r_bar
            return
        phi = self.top10_prev if P.phi_fixed < 0 else P.phi_fixed
        self.prem = P.mu_prem * softplus(phi - P.theta_prem, 50.0)
        infl = getattr(self, 'infl_prev', 0.0)
        self.r_policy = max(0.0, P.r_cb + P.phi_pi * (infl - P.pi_target))
        self.r_loan = self.r_policy + P.spread + self.prem
        r_ref = P.r_cb + P.spread
        self.rbar = P.r_eq + P.rbar_link * self.r_loan + (1 - P.rbar_link) * r_ref
        # 国債の金利: 政策金利 + 上乗せ(国債/所得が θ_B を超えた分)。金融抑圧の道では上乗せを払わない
        b = getattr(self, 'b_last', 0.0)
        self.premB = P.mu_B * softplus(b - P.theta_B, 50.0)
        if P.resolve == 'repression' and b > P.theta_resolve:
            self.premB = 0.0
        self.r_bond = self.r_policy + self.premB

    # ------------------------------------------------ 年の中(速い層)
    def plan_investment(self):
        P = self.p
        g = P.delta + P.phi_r * (self.rho_prev - self.rbar) + P.phi_u * (self.u_prev - P.u_star)
        if P.phi_q:
            g = g + P.phi_q * self.q_gap                    # 段 10b: 株が基本の値打ちより高いほど、投資が増える
        inv = self.K * softplus(g, 50.0)
        cost = self.prices[2] * self.kap_new @ inv
        M_c = float(np.sum(self.H.asset_share * np.maximum(self.H.M, 0.0))) if self.H.propensity == 'continuous' \
            else float(self.H.eq @ self.H.M)
        avail = max(self.profit_prev, 0.0) + max(M_c, 0.0) * 0.5
        if P.credit:
            cb = self.fp.cbar if (self.fp is not None and P.lender not in ('none', 'blind')) else 1.0   # 段 15: 貸し手が貸す気のある割合
            cb = cb * getattr(self, 'cf', 1.0)                         # 段 16: 準備の縛り
            avail += float((np.maximum(P.lam * self.prices[2] * self.kap_avg * self.K - self.loans, 0.0) * cb).sum())
        self.inv_limited = bool(cost > avail and cost > 0)
        if self.inv_limited:
            inv = inv * avail / cost
        if P.lending != 'none' and P.credit:
            # 段 15b: 借りて賄う分 (1 − f_R) は、貸し手が承認した割合だけ。成功の確率は、去年のみなの投資で決まる(外部性)
            xr = getattr(self, 'x_inv_prev', 1.0)
            ps = float(np.clip(P.p0_s + P.kappa_x * (xr - 1.0), 0.05, 0.99))
            if P.lending == 'market':
                seen = ps + P.phi_sw * self.q_gap * self.trust()      # 段 15c: 株の値段の膨らみを、成功の印と読む(実際の成功は ps のまま)
                A = 1.0 / (1.0 + np.exp(-(seen - P.p_thr - P.h_caut) / P.sig_s))
                succ = ps
                self.lend_Ax = A
            else:
                A = P.g_rel
                succ = P.p_good
                self.lend_Ax = P.g_rel_x
            self.lend_A, self.lend_ps, self.lend_succ = A, ps, succ
            inv = inv * (P.f_R + (1 - P.f_R) * A)
        return inv

    def gamma_eff(self):
        """段 B7: 格上げの工房があれば、贅沢品の一製品あたり労働は、古い企業と工房の割合で混ぜたもの。なければ gamma そのもの。"""
        if self.p.up_gen > 0 and self.up_s > 0:
            g = self.gamma.copy()
            g[1] = (1 - self.up_s) * self.gamma[1] + self.up_s * self.up_gX
            return g
        return self.gamma

    def subs_heads(self):
        """段 9: 生存の量を数える人数(子を育てる世帯は 1 + child_cost 倍)。child_cost = 0 なら人数そのもの。"""
        H, P = self.H, self.p
        if P.child_cost == 0:
            return H.heads
        return H.heads * (1 + P.child_cost * (H.kids > 0))

    def variety(self):
        """段 8: 贅沢品の種類の数 n(知財の総量から)。psi_v = 0 なら 1。"""
        P = self.p
        if P.psi_v == 0 or not P.land:
            return 1.0
        tot = float(self.H.ip.sum())
        return max(tot / P.ip_ref, 1e-6) ** P.psi_v

    def b_share(self):
        """生存量を超えた支出のうち必需品に回る割合。種類が増えると、贅沢品の実効の価格が下がり、eps_v > 1 なら贅沢品へ寄る。"""
        P = self.p
        if P.psi_v == 0 or P.eps_v == 1.0:
            return P.b_N
        n = self.variety()
        c = (1 - P.b_N) * n ** ((P.eps_v - 1) / (P.sigma_v - 1))
        return P.b_N / (P.b_N + c)

    def b_vec(self):
        """記入ごとの、生存量を超えた支出のうち必需品に回る割合。プレイヤーが「何を削るか」(e.act['lux']: 超えた支出のうち
        贅沢品に回す割合)を選んだときだけ、その記入を替える(段 B6)。選ばなければ b_share() のまま(前と同じ数)。"""
        b = self.b_share()
        if self.pl is None or self.act.get('lux') is None:
            return b
        v = np.full(len(self.H.M), b)
        v[self.pl] = 1.0 - float(np.clip(self.act['lux'], 0.0, 1.0))
        return v

    def demand(self, price, inv, B):
        P = self.p
        PN, PL, PI = price
        base = PN * self.subs_heads() * P.q0
        disc = softplus(B - base, 10.0 / np.maximum(base, 1e-9))
        bN_ = self.b_vec()
        xN = np.sum((B - disc) / PN + bN_ * disc / PN)
        xL = np.sum((1 - bN_) * disc / PL)
        xI = (P.alpha[0] * xN + P.alpha[1] * xL + self.kap_new @ inv) / (1 - P.alpha[2])
        return np.array([xN, xL, xI])

    def crisis_on(self):
        P = self.p
        return bool(P.crisis_track or P.inventory or P.cap_N > 0 or P.g_stock > 0 or P.g_pw > 0 or P.f_K > 0 or self.S_G > 0
                    or P.crisis_rule != 'none' or P.delta_scrap > 0)

    def supply_at_cap(self, cap, PI):
        """上限の価格 cap のもとで、必需品の企業が赤字なく作れる量(変動費 = cap となる稼働率まで)。
        変動費 = W γ (1 + ω softplus(u − 1)) + 原料 + 地代。最も低い変動費(W γ + 原料 + 地代)が cap を超えれば、誰も作らない。"""
        P = self.p
        var0 = P.alpha[0] * PI + (P.rho_T * cap if P.land else 0.0)
        y = ((cap - var0) / (self.W * self.gamma[0]) - 1.0) / P.omega      # softplus(u − 1, 20) = y
        if y <= 1e-12:
            return 0.0
        u = 1.0 + np.log(np.expm1(min(y * 20.0, 700.0))) / 20.0
        return float(self.K[0] * max(u, 0.0))

    def evaluate_crisis(self, price, inv, B):
        """段 7 の年の中の釣り合い: 在庫、価格の上限と品不足、政府の備蓄の買いと放出、公共事業、固定費。"""
        P, H = self.p, self.H
        PN, PL, PI = price
        base = PN * self.subs_heads() * P.q0
        disc = softplus(B - base, 10.0 / np.maximum(base, 1e-9))
        bN_ = self.b_vec()
        dN = float(np.sum((B - disc) / PN + bN_ * disc / PN))
        xL = float(np.sum((1 - bN_) * disc / PL))
        capped = P.cap_N > 0 and PN >= P.cap_N * (1 - 1e-9)
        gbuy = 0.0 if capped else self._gbuy
        D = dN + gbuy
        Inv = (P.inv_target * dN if self.InvN is None else self.InvN) if P.inventory else 0.0
        plan = max(D + P.chi_inv * (P.inv_target * D - Inv), 0.0) if P.inventory else D
        rel = 0.0
        if capped:
            xs = self.supply_at_cap(P.cap_N, PI)
            xN = min(xs, plan)
            if P.crisis_rule in ('last', 'share'):
                # 最後の生産者・損失の分担が保つのは、生存の量(一人 q0)まで。それを超える需要は割り当てる
                need = float(P.q0 * self.subs_heads().sum())
                xN = max(xN, min(plan, max(need - Inv - self.S_G, 0.0)))
            firm_sales = min(D, xN + Inv)
            rel = min(D - firm_sales, self.S_G)                  # 政府の備蓄を放出する
        else:
            xN = plan
            firm_sales = min(D, xN + Inv)
            if firm_sales < D:
                rel = min(D - firm_sales, self.S_G)
        sales = firm_sales + rel
        hhN = sales - gbuy
        GI = self._GI_nom / PI
        xI = (P.alpha[0] * xN + P.alpha[1] * xL + self.kap_new @ inv + GI) / (1 - P.alpha[2])
        x = np.array([xN, xL, xI])
        u = x / self.K
        geff = self.gamma_eff() * (1 + P.omega * softplus(u - 1.0, 20.0))
        cost = self.W * geff + PI * P.alpha
        if P.land:
            cost = cost + np.array([P.rho_T * PN, 0.0, 0.0])
        # 固定費(P.f_K)は値段の元の費用に入れない(値段は変動費の上の値上げ幅。固定費は利益から引く。需要が落ちると赤字になる)
        m = self.m0 + P.zeta * np.tanh(u - P.u_star)
        if P.inventory:
            m = m + np.array([P.zeta_inv * np.tanh(1.0 - (Inv + xN - firm_sales) / max(P.inv_target * D, 1e-12)), 0.0, 0.0])
        self._cx = dict(dN=dN, D=D, gbuy=gbuy, xN=xN, firm_sales=firm_sales, rel=rel, hhN=hhN, short=max(dN - hhN, 0.0),
                        capped=capped, GI=GI, Inv=Inv)
        return x, u, geff, cost, m

    def evaluate(self, price, inv, B):
        P = self.p
        if self.crisis_on():
            return self.evaluate_crisis(price, inv, B)
        self._cx = None
        x = self.demand(price, inv, B)
        u = x / self.K
        geff = self.gamma_eff() * (1 + P.omega * softplus(u - 1.0, 20.0))
        cost = self.W * geff + price[2] * P.alpha
        if P.land:
            cost = cost + np.array([P.rho_T * price[0], 0.0, 0.0])   # 地代(必需品一単位あたり)
        m = self.m0 + P.zeta * np.tanh(u - P.u_star)
        return x, u, geff, cost, m

    def residual(self, z, inv, B):
        P = self.p
        price = np.exp(z)
        x, u, geff, cost, m = self.evaluate(price, inv, B)
        target = (1 + m) * cost
        if P.cap_N > 0:
            target = np.array([min(target[0], P.cap_N), target[1], target[2]])   # 価格の上限
        res = list(z - np.log(target))
        res += list(np.sqrt(P.tactic) * (z - np.log(self.prices)))
        return np.array(res)

    def solve_prices(self, inv, B):
        if self.p.solver == 'portable':
            sol = lm_solve(lambda z: self.residual(z, inv, B), np.log(self.prices), xtol=1e-13, ftol=0.0)
            P = self.p
            if P.cap_N > 0 and np.exp(sol.x[0]) >= P.cap_N * (1 - 1e-6):
                # 価格の上限が効くとき: 必需品の価格を上限にそろえ、残りの二つを解き直す。上限の角(残差が滑らかでない)では、
                # 数値の微分が丸めのずれを大きくし、JS の版と 1e-8 ほど違う解に止まるので、角を解から外す(portable の道だけ)
                z0 = np.log(P.cap_N)
                full = lambda y: self.residual(np.array([z0, y[0], y[1]]), inv, B)
                sub = lm_solve(lambda y: full(y)[[1, 2, 4, 5]], sol.x[1:], xtol=1e-13, ftol=0.0)
                x = np.array([z0, sub.x[0], sub.x[1]])
                sol = LMResult(x, full(sub.x), sol.nfev + sub.nfev, sol.nit + sub.nit)
        else:
            sol = least_squares(self.residual, np.log(self.prices), args=(inv, B), method='lm', xtol=1e-13, ftol=1e-13)
        return np.exp(sol.x), sol

    # ------------------------------------------------ 土地の市場、貸倒れ、世代の入れ替わり(段 2)
    def land_market(self, rent, roy, rL, W, wage_bill, E, Yd, lab_inc, div, bN=None, bL=None):
        """v6a・v6b の土地の市場を、重みつきの標本の上で行う。売り買いは家計の間の預金の移動なので、
        マクロの家計の預金は変わらない(帳簿は動かない)。貸倒れは、借り手の借金の棒引きと、銀行の持ち主(持分)の負担で、
        これも家計の間の移動になる。"""
        P, H = self.p, self.H
        T = P.T_land
        wbar = wage_bill / max(E, 1e-9)                       # 一人あたりの賃金
        U_lim = P.mu_u * wbar * H.heads * getattr(self, 'cf', 1.0)   # 無担保の借入の上限(記入の人数ぶん。準備の縛りで減る。段 16)
        ip_tot = H.ip.sum()
        v_IP = roy / max(ip_tot * (rL + P.delta_IP), 1e-9) if ip_tot > 0 else 0.0
        if P.ip_market and self.qIP is not None:
            v_IP = self.qIP                                     # 知財の市場があれば、市場の値段で数える
        v_eq = max(div, 0.0) / max(self.rbar, 1e-3)            # 持分の値打ち(配当の割引)
        if P.eq_market and self.pE is not None:
            v_eq = self.pE                                       # 株式の市場があれば、市場の値段で数える
        y_T = rent / (T * self.pT)
        x_T = 0.2 * np.tanh((y_T + self.gT_e - rL) / 0.2)
        theta = float(min(P.theta0 * np.exp(P.theta1 * x_T) + P.theta_shock, 0.9))
        other = H.M + v_IP * H.ip + v_eq * H.eq
        human = P.h_y * np.maximum(lab_inc * (1 - self.tau), 0.0)
        land, chi, lT = H.land, P.chi_T * P.attention, P.lam_T

        def holdings(pr):
            nw = other + pr * land
            d = np.maximum(0.0, theta * (nw + human) / pr)
            cap = (U_lim + H.M + pr * land) / ((1 - lT) * pr)
            return np.clip(land + chi * (d - land), 0.0, np.maximum(cap, 0.0))
        lo, hi = np.log(self.pT) - 8, np.log(self.pT) + 8
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if holdings(np.exp(mid)).sum() > T:
                lo = mid
            else:
                hi = mid
        pT = float(np.exp(0.5 * (lo + hi)))
        new = holdings(pT)
        land_err = new.sum() / T - 1
        new = new * (T / new.sum())
        H.M = H.M - pT * (new - land)
        H.land = new
        self.gT_e += P.nu_T * (np.log(pT / self.pT) - self.gT_e)
        self.pT = pT
        ip_rec = {}
        if P.ip_market:
            ip_rec = self.ip_trade(roy, rL, U_lim, v_eq, human, bL)
            v_IP = self.qIP
        if P.eq_market:
            ip_rec.update(self.eq_trade(div, rL, U_lim, v_IP, human))
            v_eq = self.pE
        # 貸倒れ: 借入の上限を超えた借金を棒引きにする。銀行の自己資本は 0 なので、損は預金者が預金に応じて負う
        # (預金の削減。損を負う人が新たに借り手にならないように)
        limit = lT * pT * H.land + U_lim + (P.lam_IP * self.qIP * H.ip if P.ip_market else 0.0)
        over = np.maximum(0.0, -H.M - limit)
        self.risk = np.maximum(-H.M, 0.0) / np.maximum(limit, 1e-12)   # 借金 / 借入の上限(重点標本化に使う)
        wo = over.sum()
        breakups = float(H.heads[over > P.breakup_eps * np.maximum(limit, 1e-12)].sum())   # 借金を棒引きにされた世帯の人数(一家離散)
        H.M = H.M + over
        bailout = 0.0
        if wo > 0:
            dep = np.maximum(H.M, 0.0)
            cut = min(wo, P.haircut_max * dep.sum())
            H.M = H.M - cut * dep / max(dep.sum(), 1e-12)
            bailout = wo - cut
            if bailout > 0:
                self.L.pay('bank bailout', self.sec['gov'], self.sec['hh'], bailout)      # 救済: 政府が銀行の損を埋め、その分の預金が家計に残る
                self.L.trade('bonds', self.sec['bank'], self.sec['gov'], bailout)
        # 世代の入れ替わり: 去る世帯の財産(預金、土地、知財、持分)を相続させる。新しい世帯は財産なしで始まる
        n = len(H.M)
        self.uniform_w = bool(np.allclose(H.heads, H.heads[0]))
        if P.turnover > 0:
            out = np.nonzero(self.rng.random(n) < P.turnover)[0]
            for i in out:
                if i == self.pl and P.player_heir == 'self':
                    # 段 19: プレイヤーの次の代が、財産を継いで遊びを続ける
                    if H.pf[i] != 0.0:
                        if self.fund_mode():
                            self.fund_D -= H.pf[i]
                        H.M[i] += H.pf[i]; H.pf[i] = 0.0
                    if P.child_cost > 0:
                        H.kids[i] = P.child_years
                    H.skill[i] = 0.0; H.age[i] = 0.0; H.pts[i] = 0.0
                    self.pl_gen += 1
                    continue
                if self.uniform_w:
                    h = self.rng.integers(n - 1); h = h + (h >= i)
                else:
                    pw = H.heads.copy(); pw[i] = 0.0
                    h = int(self.rng.choice(n, p=pw / pw.sum()))   # 人数に比例して相続人を選ぶ(重みが不揃いのとき)
                w_other = H.heads.copy(); w_other[i] = 0.0; w_other /= w_other.sum()
                if H.pf[i] != 0.0:
                    if self.fund_mode():
                        self.fund_D -= H.pf[i]                        # 段 10b: 基金が預金で払い出す
                    H.M[i] += H.pf[i]; H.pf[i] = 0.0              # 積立の残りは、引き出せる預金として相続する(段 10)
                for arr in (H.M, H.land, H.ip, H.eq, H.F):
                    val = arr[i]
                    arr[i] = 0.0
                    if P.inherit == 'lottery':
                        arr[h] += val
                    else:
                        arr += val * w_other
                if P.child_cost > 0:
                    H.kids[i] = P.child_years                  # 新しい世帯は、子を育てる年から始まる(段 9)
                H.skill[i] = 0.0
                H.age[i] = 0.0
                H.pts[i] = 0.0
        if P.is_split > 0:
            self.split_merge(pT, v_IP, v_eq)
        hl = np.maximum(-H.M, 0.0).sum()
        nw = H.M + pT * H.land + v_IP * H.ip + v_eq * H.eq + H.F         # 外の資産も正味資産に数える(段 6)
        return dict(pT=pT, theta=theta, land_err=land_err, wo_h=wo, bailout=bailout, breakups=breakups, hh_debt=hl, n_borrowers=float(H.heads[H.M < 0].sum()),
                    gini_nw=gini_w(nw / H.heads, H.heads), top10_nw=top_share_w(np.maximum(nw, 0) / H.heads, H.heads),
                    land_value=pT * T, ip_tot=float(H.ip.sum()), land_sum_err=abs(H.land.sum() - T), **ip_rec)

    def plan_foreign(self, r_out, fx=1.0):
        """段 6: 家計が今年、外の資産をどれだけ買うか(負なら売る)。去年の終わりの正味資産と、今年の内と外の預金の金利から決める。
        外に置きたい割合 = θ_F0 + φ_K (外の金利 − 内の金利) + φ_L × 逃れられる富裕税の率(率 × 見えない割合 × 免除を超えた分の割合)。
        買う額は手元の預金まで、売る額は持っている外の資産まで。"""
        P, H = self.p, self.H
        Fv = H.F * fx                         # fx: 今年の為替で測り直す比(外の資産の値打ちが為替で変わる)
        nw = H.M + self.pT * H.land + (self.qIP or 0.0) * H.ip + self.v_eq_last * H.eq + Fv
        pc = nw / H.heads
        lrate = P.wealth_tax + (P.levy_rate if (P.resolve == 'levy' and getattr(self, 'b_last', 0.0) > P.theta_resolve) else 0.0)
        ex = P.levy_exempt * max(nw.sum() / H.heads.sum(), 1e-12)
        eff = lrate * (1 - P.enf_F) * np.maximum(pc - ex, 0.0) / np.maximum(pc, 1e-12)
        rD = self.r_policy if P.credit else 0.0
        th = np.clip(P.theta_F0 + P.phi_K * (r_out - rD) + P.phi_L * eff, 0.0, P.theta_F_max)
        d = P.chi_F * P.attention * (th * np.maximum(nw, 0.0) - Fv)
        dF = np.clip(d, -Fv, np.maximum(H.M, 0.0))
        return dF

    def eq_trade(self, div, rL, U_lim, v_IP, human):
        """株式の市場(段 5)。土地の市場と同じ形で、持分の値段 pE(持分 1 = 全部の値段)を、標本の望む保有の和 = 1 となるよう解く。
        配当は持分に応じて配られている。持ち主の実際の利回り R = (配当 + 今年の値段) / 去年の値段 − 1 を記録する。"""
        P, H = self.p, self.H
        self.div_s = max(div, 0.0) if getattr(self, 'div_s', None) is None else 0.7 * self.div_s + 0.3 * div
        fund = max(self.div_s, 1e-6 * max(self.Y_last, 1.0)) / max(self.rbar, 1e-3)    # 均した配当の割引
        if self.pE is None:
            self.pE = fund
        p0 = self.pE
        y = max(div, 0.0) / p0
        x = 0.2 * np.tanh((y + self.gE_e - rL) / 0.2)
        th = float(min(P.theta0_E * np.exp(P.theta1_E * x), 0.9)) * self.trust()   # 段 15c: 信頼が下がると、株に置きたい割合が下がる
        eq = H.eq.copy()
        base = H.M + self.pT * H.land + v_IP * H.ip + P.human_E * human
        lT, chi = P.lam_T, P.chi_E * P.attention
        fm = self.fund_mode()
        fe, fD = self.fund_e, self.fund_D

        def fund_hold(pr):
            # 段 10b: 基金は、値打ち(預金 + 持分)の fund_eq を株に置こうとする(近づく速さは家計と同じ)。買えるのは預金の分まで
            if not fm:
                return 0.0
            want = P.fund_eq * (fD + fe * pr) / pr
            return float(np.clip(fe + P.chi_E * (want - fe), 0.0, min(fe + max(fD, 0.0) / pr, 0.9)))   # 基金の持分は 9 割まで

        def holdings(pr):
            w = base + pr * eq
            if P.kappa_E > 0:
                pc = np.maximum(w, 0.0) / H.heads
                rel = pc / max(float(np.maximum(w, 0.0).sum() / H.heads.sum()), 1e-12)
                thi = np.minimum(th * rel ** P.kappa_E, 0.95)
            else:
                thi = th
            d = np.maximum(0.0, thi * w / pr)
            cap = eq + np.maximum(H.M + P.lam_E * (U_lim + lT * self.pT * H.land), 0.0) / pr
            return np.clip(eq + chi * (d - eq), 0.0, np.maximum(cap, 0.0))
        if P.eq_pricing == 'fundamental':
            # 値段は基本の値打ち。買いたい側と売りたい側の少ないほうだけ取引する(多いほうは同じ割合で削る)
            pE = fund
            d = holdings(pE) - eq
            buy, sell = d[d > 0].sum(), -d[d < 0].sum()
            q = min(buy, sell)
            new = eq + np.where(d > 0, d * q / max(buy, 1e-300), d * q / max(sell, 1e-300))
            fe_new = fund_hold(pE)
        else:
            lo, hi = np.log(p0) - 8, np.log(p0) + 8
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if holdings(np.exp(mid)).sum() + fund_hold(np.exp(mid)) > 1.0:
                    lo = mid
                else:
                    hi = mid
            pE = float(np.exp(0.5 * (lo + hi)))
            new = holdings(pE)
            fe_new = fund_hold(pE)
        new = new * (1.0 - fe_new) / new.sum()                 # 家計の持分の和 = 1 − 基金の持分
        vol = float(np.abs(new - eq).sum() / 2)
        H.M = H.M - pE * (new - eq)
        H.eq = new
        if fm:
            self.fund_D = fD - pE * (fe_new - fe)               # 基金の売り買いは、家計との預金の移動(マクロの家計の預金は変わらない)
            self.fund_e = fe_new
            V = self.fund_D + fe_new * pE
            if H.pf.sum() > 0:
                H.pf = H.pf * V / H.pf.sum()                   # 積立の請求権を、基金の値打ちに合わせる(株の値上がり・値下がりを配る)
        self.q_gap = float(np.clip(np.log(pE / fund), -1.0, 1.0))
        R = np.nan
        if self.eq_hold is not None:
            R = (div + pE) / self.eq_hold - 1                # 配当が負なら、持ち主が払い込む
        self.eq_hold = pE
        self.gE_e += P.nu_E * (np.log(pE / p0) - self.gE_e)
        self.pE = pE
        return dict(pE=pE, R_eq=R, eq_vol=vol, eq_fund=fund, theta_eq=th, fund_e=self.fund_e, fund_D=self.fund_D, q_gap=self.q_gap)

    def ip_trade(self, roy, rL, U_lim, v_eq, human, bL):
        """知財の市場(段 2b)。土地の市場と同じ形で、値段 q は、標本の望む保有の和 = 知財の総量 となるよう二分法で解く。
        違いは、知財が減耗すること(δ_IP)と、利回りが使用料から来ること。持ち主の実際の利回り
        R = (使用料 + 今年の値段 × 減耗のあとの去年の保有) / (去年の値段 × 去年の保有) − 1 を記録する。"""
        P, H = self.p, self.H
        ip_tot = H.ip.sum()
        if ip_tot <= 0:
            return dict(qIP=0.0, ip_value=0.0, R_ip=np.nan, ip_vol=0.0, gIP_e=self.gIP_e, theta_ip=0.0)
        fund = roy / (ip_tot * (rL + P.delta_IP))
        if P.ip_pricing == 'fundamental':
            # 裁定の価格: 持ち主は、使用料と値段の変化で、ちょうど貸出の金利を得る(使用料が一定なら)。売り買いは起きない
            q = max(fund, 1e-12)
            R = np.nan
            if self.ip_hold is not None:
                units, value = self.ip_hold
                R = (roy + q * (1 - P.delta_IP) * units) / value - 1
            self.ip_hold = (ip_tot, q * ip_tot)
            self.qIP = q
            return dict(qIP=q, ip_value=q * ip_tot, R_ip=R, ip_vol=0.0, gIP_e=0.0, theta_ip=0.0, ip_fund=fund)
        if self.qIP is None:
            self.qIP = fund if fund > 0 else bL / P.eta_IP           # 初めは、使用料の割引の値(なければ作る費用)
        q0 = self.qIP
        y = roy / (ip_tot * q0)
        xc = P.xcap_IP
        x = xc * np.tanh((y + self.gIP_e - P.delta_IP - rL) / xc)
        th = float(min(P.theta0_IP * np.exp(P.theta1_IP * x), 0.9))
        lT, lI, chi = P.lam_T, P.lam_IP, P.chi_IP * P.attention
        ip = H.ip.copy()
        base = H.M + self.pT * H.land + v_eq * H.eq + human

        def holdings(q):
            d = np.maximum(0.0, th * (base + q * ip) / q)
            cap = (U_lim + lT * self.pT * H.land + H.M + q * ip) / ((1 - lI) * q)
            return np.clip(ip + chi * (d - ip), 0.0, np.maximum(cap, 0.0))
        lo, hi = np.log(q0) - 8, np.log(q0) + 8
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if holdings(np.exp(mid)).sum() > ip_tot:
                lo = mid
            else:
                hi = mid
        q = float(np.exp(0.5 * (lo + hi)))
        new = holdings(q)
        new = new * (ip_tot / new.sum())
        vol = float(np.abs(new - ip).sum() / 2)
        H.M = H.M - q * (new - ip)
        H.ip = new
        # 去年の持ち主の実際の利回り(使用料は去年の保有に応じて配られている)
        R = np.nan
        if self.ip_hold is not None:
            units, value = self.ip_hold
            R = (roy + q * (1 - P.delta_IP) * units) / value - 1
        self.ip_hold = (ip_tot, q * ip_tot)
        self.gIP_e += P.nu_IP * (np.log(q / q0) - self.gIP_e)
        self.qIP = q
        return dict(qIP=q, ip_value=q * ip_tot, R_ip=R, ip_vol=vol, gIP_e=self.gIP_e, theta_ip=th, ip_fund=fund)

    # ------------------------------------------------ 重点標本化(分割と併合)
    EXT = ('M', 'Yd', 'heads', 'lab', 'eq', 'land', 'ip', 'extra', 'F', 'pf', 'pts', 'hd')   # 記入の合計の量(足せる)
    INT = ('c_y', 'c_m', 'asset_share', 'inj', 'kids', 'skill', 'age')           # 記入の中の割合(人数や働く力で平均する)

    def split_merge(self, pT, v_IP, v_eq):
        """借金の上限に近い記入を二つに分け(重みは半分ずつ)、借金のない似た記入を二つ併せる。記入の数は一定。
        総量(預金、土地、知財、持分、人数、働く力)は変わらない。分けた二つは、そのあとの乱数(入れ替わり、打撃)で別々の道をたどる。
        併せるのは、借金がなく、正味資産が上位 1 割に入らない記入のうち、一人あたりの正味資産がもっとも近い二つ。"""
        P, H = self.p, self.H
        nw_pc = (H.M + pT * H.land + v_IP * H.ip + v_eq * H.eq) / H.heads
        for _ in range(P.is_split):
            cand = np.nonzero((self.risk >= P.is_risk) & (H.heads >= 2 * P.is_min_heads))[0]
            if self.pl is not None:
                cand = cand[cand != self.pl]                         # プレイヤーの記入は分けも併せもしない(段 19)
            if len(cand) == 0:
                break
            i = cand[np.argmax(self.risk[cand])]
            safe = np.nonzero((self.risk == 0) & (H.M >= 0) & (nw_pc < np.quantile(nw_pc, 0.9)))[0]
            if self.pl is not None:
                safe = safe[safe != self.pl]
            if len(safe) < 2:
                break
            o = safe[np.argsort(nw_pc[safe])]
            gaps = np.diff(nw_pc[o])
            gaps = np.where(H.heads[o[:-1]] + H.heads[o[1:]] <= P.is_max_ratio * H.heads.mean(), gaps, np.inf)
            if not np.isfinite(gaps).any():
                break
            k = int(np.argmin(gaps))
            a, b = o[k], o[k + 1]
            wa, wb = H.heads[a], H.heads[b]
            if P.is_mode == 'roulette':
                # ロシアン・ルーレット: 人数に比例した確率で一方を残し、重みを二つ分にする(期待値は変わらない)
                if self.dis_rng.random() < wb / (wa + wb):
                    for name in self.EXT + self.INT:
                        arr = getattr(H, name)
                        arr[a] = arr[b]
                    nw_pc[a] = nw_pc[b]
                    wa, wb = wb, wa
                f = (wa + wb) / wa
                tot = {name: getattr(H, name).sum() for name in self.EXT}
                for name in self.EXT:
                    getattr(H, name)[a] *= f
                    getattr(H, name)[b] = 0.0
            else:
                # 平均: a に b を足す(割合の量は人数で平均、怪我は働く力で平均)
                la, lb = H.lab[a], H.lab[b]
                for name in self.INT:
                    arr = getattr(H, name)
                    wA, wB = (la, lb) if name == 'inj' else (wa, wb)
                    arr[a] = (arr[a] * wA + arr[b] * wB) / max(wA + wB, 1e-300)
                for name in self.EXT:
                    arr = getattr(H, name)
                    arr[a] += arr[b]
                tot = None
            # 分ける: b の場所に i の半分を写す
            for name in self.EXT:
                arr = getattr(H, name)
                arr[i] *= 0.5
                arr[b] = arr[i]
            for name in self.INT:
                arr = getattr(H, name)
                arr[b] = arr[i]
            if tot is not None:
                # 総量のずれを全員に配って直す(持ち上げ)。預金は一人あたり同じ額、ほかの量は比で
                for name in self.EXT:
                    arr = getattr(H, name)
                    if name == 'M':
                        # 預金のずれは、預金のある世帯に預金に比例して配る(借金の上限の近くの借り手に揺れを足さない)
                        dep = np.maximum(arr, 0.0)
                        arr += (tot[name] - arr.sum()) * dep / max(dep.sum(), 1e-12)
                    elif name in ('Yd', 'extra'):
                        arr += (tot[name] - arr.sum()) * H.heads / H.heads.sum()
                    elif arr.sum() > 0:
                        arr *= tot[name] / arr.sum()
            self.risk[b] = self.risk[i]
            self.risk[a] = 0.0
            nw_pc[a] = (nw_pc[a] * wa + nw_pc[b] * wb) / (wa + wb)
            nw_pc[b] = nw_pc[i]
            self.n_splits = getattr(self, 'n_splits', 0) + 1

    # ------------------------------------------------ 災害(段 4)
    def apply_disaster(self, size, scope='common'):
        """災害を起こす。被害 d = 大きさ × exp(− 守りの資本 / 尺度)。資本の d(ばらばらの打撃では d × 打撃を受ける割合)が壊れ、
        家計は働く力を失う(共通: 全員が inj_coef × d、ばらばら: 一部の世帯だけが inj_coef × d / 割合、上限 1)。
        失われる働く力の期待値は、どちらも同じ。救済は、怪我が戻るまでに失う労働所得の見込みの relief_rate を、一度に埋める。"""
        P, H, L = self.p, self.H, self.L
        d = size * np.exp(-self.prot / P.prot_scale)
        n = len(H.M)
        if scope == 'common':
            hit = np.ones(n, bool)
            add = np.full(n, min(P.inj_coef * d, 1.0))
            dK = d
        else:
            order = self.dis_rng.permutation(n)
            cum = np.cumsum(H.heads[order]) / H.heads.sum()
            hit = np.zeros(n, bool); hit[order[cum <= P.hit_share + 1e-12]] = True
            if not hit.any():
                hit[order[0]] = True
            share = H.heads[hit].sum() / H.heads.sum()
            add = np.where(hit, min(P.inj_coef * d / share, 1.0), 0.0)
            dK = d * share
        H.inj = 1 - (1 - H.inj) * (1 - add)
        lost = dK * self.K
        self.K = self.K - lost
        for j in range(3):
            L.revalue(self.real[j], self.firms[j], -lost[j])
        # 救済: 怪我が戻るまでに失う労働所得の見込み(前の年の労働所得 × 怪我 / 戻る率)の relief_rate
        lab_income = np.maximum(getattr(self, 'lab_inc_last', np.zeros(n)), 0.0)
        need = P.relief_rate * lab_income * add / P.rec_inj
        relief = float(need.sum())
        if P.relief == 'state' and relief > 0:
            L.pay('disaster relief', self.sec['gov'], self.sec['hh'], relief)
            L.trade('bonds', self.sec['bank'], self.sec['gov'], relief)
            H.M = H.M + need
        elif P.relief == 'mutual' and relief > 0:
            dep = np.where(hit, 0.0, np.maximum(H.M, 0.0))
            pay = min(relief, dep.sum())
            H.M = H.M - pay * dep / max(dep.sum(), 1e-12) + need * (pay / relief)
            relief = pay
        elif P.relief == 'none':
            relief = 0.0
        self.last_hit = hit
        self.dis_rec = dict(dis_size=size, dis_d=d, relief=relief, K_lost=float((lost * self.prices[2] * self.kap_avg).sum()))

    # ------------------------------------------------ 学習と政治(段 5)
    def update_belief(self):
        """災害の頻度の信念を、その年の記録で更新し(古い記録は ω で割り引く)、守りの量を決める(ノート 3 第 7.7 節)。"""
        P = self.p
        n = getattr(self, 'events_this_year', 0)
        self.n_eff = P.learn_omega * getattr(self, 'n_eff', 0.0) + n
        self.t_eff = P.learn_omega * getattr(self, 't_eff', 0.0) + 1.0
        a, b = P.prior_a + self.n_eff, P.prior_b + self.t_eff
        if P.learn_attitude == 'neutral':
            lam = a / b
        elif P.learn_attitude == 'interval':
            # 段 13: 信念の区間(ノート I 第 5 章)。事後の区間を、構造の不確かさの床で広げ、α-maxmin で一つに決める
            from scipy.stats import gamma
            mean = a / b
            lo, hi = gamma.ppf(P.amb_q, a, scale=1 / b), gamma.ppf(1 - P.amb_q, a, scale=1 / b)
            lo, hi = min(lo, mean * (1 - P.amb_floor)), max(hi, mean * (1 + P.amb_floor))
            lam = (1 - P.amb_alpha) * max(lo, 0.0) + P.amb_alpha * hi
            self.belief_lo, self.belief_hi = lo, hi
        else:
            from scipy.stats import gamma
            lam = gamma.ppf(0.9 if P.learn_attitude == 'pessimistic' else 0.1, a, scale=1 / b)
        self.belief = lam
        P.protect = min(P.protect_max, P.k_learn * lam)

    def hold_vote(self):
        """資本所得への税を投票で決める(ノート M 第 4 章)。群れ g の一人あたりの賭け金 A_g(二つの案の間の所得の差、
        一人あたりの所得に対する比)と、群れの中の相関 ρ_g から、投票の効き目 √(ρ_g/π)、投票率 1 − exp(−A_g √(ρ_g/π) / c)。
        人数を重みにした票の多い方の案になる。"""
        P, H = self.p, self.H
        cap = np.maximum(self.cap_inc_last, 0.0)
        rent = H.asset_share > P.rentier_cut
        heads = H.heads
        inc_pc = max(np.maximum(H.Yd, 0.0).sum() / heads.sum(), 1e-12)
        dtau = P.tau_hi - P.tau_lo
        transfer_pc = dtau * cap.sum() / heads.sum()                 # 高い案で増える、一人あたりの配りもの
        stake = (transfer_pc * heads - dtau * cap) / heads / inc_pc  # 高い案で増える所得(一人あたり、所得に対する比)
        k = self.pl
        pl_g = None
        if k is not None:
            side0 = self.act.get('vote')
            # 呼びかけは、自分の投じる案を好む群れの相関を高める(投じなければ自分の群れ)
            pl_g = 'w' if side0 == 'hi' else 'k' if side0 == 'lo' else ('k' if rent[k] else 'w')
            mask_pl = np.ones(len(heads), bool); mask_pl[k] = False      # プレイヤーの票は、規則の票から外して別に数える
        else:
            mask_pl = np.ones(len(heads), bool)

        def tally(sway):
            out = {}
            for g, mask, rho, sign in (('w', ~rent, P.rho_w, +1), ('k', rent, P.rho_k, -1)):
                mask = mask & mask_pl
                if not mask.any():
                    out[g] = (0.0, 0.0); continue
                rho = rho + (sway if g == pl_g else 0.0)
                A = max(sign * np.average(stake[mask], weights=heads[mask]), 0.0)
                if P.solver == 'portable' and A < 1e-12:
                    A = 0.0                      # 賭け金の和は作り方から 0 なので、資産家がいない年は平均が丸めの大きさになる(portable の道では 0 に)
                xv = A * np.sqrt(rho / np.pi) / P.c_vote
                # portable の道では expm1 で(投票率がごく小さいとき、1 − exp の引き算で桁が落ち、JS の版と割れるので)
                t = -np.expm1(-xv) if P.solver == 'portable' else 1 - np.exp(-xv)
                v = float(heads[mask].sum() * t)
                if P.vote_noise:
                    v = max(v + float(heads[mask].sum() * np.sqrt(rho * t * (1 - t))) * z[g], 0.0)   # 相関のある群れの票の揺れ
                out[g] = (v, float(t))
            return out
        z = {}
        if P.vote_noise:
            z = dict(w=float(self.vote_rng.standard_normal()), k=float(self.vote_rng.standard_normal()))
        out = tally(0.0)
        if P.vote_noise and out['w'][0] == 0.0 and out['k'][0] == 0.0:
            # どちらの群れにも賭け金がなく、誰も投票しない年は、選挙が成り立たず、今の政策を保つ(段 19)
            P.redistrib = 1.0
            self.vote_rec = dict(vote_hi=0.0, vote_lo=0.0, turnout_w=0.0, turnout_k=0.0, n_rentier=float(heads[rent].sum()))
            return
        if k is not None:
            # 段 19: プレイヤーの一票(人数 player_heads)と、呼びかけによる群れの相関の高まり(見本の効き目)
            camp = float(np.clip(self.act.get('campaign', 0.0), 0.0, 1.0))
            side = self.act.get('vote')
            base_hi = out['w'][0] > out['k'][0]
            if camp > 0:
                out = tally(P.k_sway * camp)
            hi_v, lo_v = out['w'][0], out['k'][0]
            if side == 'hi':
                hi_v += heads[k]
            elif side == 'lo':
                lo_v += heads[k]
            win_hi = hi_v > lo_v
            pivot = win_hi != base_hi
            self.pl_pivots += int(pivot)
            self.vote_pl = dict(pl_vote=side or '', pl_campaign=camp, pl_pivot=float(pivot), pl_rentier_vote=bool(rent[k]))
            out = {'w': (hi_v, out['w'][1]), 'k': (lo_v, out['k'][1])}
        P.tau_k = P.tau_hi if out['w'][0] > out['k'][0] else P.tau_lo
        P.redistrib = 1.0
        self.vote_rec = dict(vote_hi=out['w'][0], vote_lo=out['k'][0], turnout_w=out['w'][1], turnout_k=out['k'][1],
                             n_rentier=float(heads[rent].sum()))

    # ------------------------------------------------ 一年(基本の層)
    def begin_year(self):
        """年のはじめ: 投資の計画と、家計の予算(前の年の所得と預金から)。"""
        P, H = self.p, self.H
        self.dis_rec = {}
        if P.up_gen != self.up_gen_seen:
            # 段 B7: 工房を開く(開き直す)
            self.up_gen_seen = P.up_gen
            self.up_s, self.up_gX, self.up_cum, self.up_peak = (P.up_s0, P.up_g0, P.up_cum0, P.up_s0) if P.up_gen > 0 else (0.0, P.up_g0, 0.0, 0.0)
        if P.disaster:
            for (ts, size, scope) in [q for q in self.shock_queue if q[0] == self.t]:
                self.apply_disaster(size, scope)
            q = P.dis_q if P.dis_q > 0 else P.dis_rate
            if P.dis_rate > 0 and self.dis_rng.random() < q:
                size = min(P.dis_min * (1 - self.dis_rng.random()) ** (-1 / P.dis_alpha), P.dis_max)
                self.apply_disaster(size, P.dis_scope)
                if P.dis_q > 0:
                    self.logw = getattr(self, 'logw', 0.0) + np.log(P.dis_rate / q)
            elif P.dis_rate > 0 and P.dis_q > 0:
                self.logw = getattr(self, 'logw', 0.0) + np.log((1 - P.dis_rate) / (1 - q))
            self.events_this_year = 1 if self.dis_rec else 0
        if P.learn:
            self.update_belief()
        if P.vote and self.t % P.vote_every == 0 and self.t > 0:
            self.hold_vote()
        self._inv = self.plan_investment()
        if self.crisis_on():
            hhN0 = self.N_last if self.N_last is not None else None
            if P.inventory and self.InvN is None:
                self.InvN = None                           # 初めの年は、需要を見てから目標の量で置く
            self._gbuy = P.chi_S * max(P.g_stock * hhN0 - self.S_G, 0.0) if (P.g_stock > 0 and hhN0) else 0.0
            self._GI_nom = P.g_pw * self.Y_prev if (P.g_pw > 0 and self.Y_prev) else 0.0
        if P.land:
            H.extra = P.c_h * np.maximum(self.pT * H.land - np.maximum(-H.M, 0.0), 0.0)
        self._B = H.budgets()
        if P.attention < 1.0:
            # 決め直さない世帯は、去年の実質の予算(必需品の値段で測る)を保つ
            if getattr(self, '_B_real_last', None) is not None and len(self._B_real_last) == len(self._B):
                self._B = P.attention * self._B + (1 - P.attention) * self._B_real_last * self.PN_prev
            self._B_real_last = self._B / self.PN_prev
        if self.pl is not None:
            # 段 19: プレイヤーの支出(規則の予算の倍率)と、子を持つ選び方
            self._B[self.pl] = max(self._B[self.pl] * float(self.act.get('spend', 1.0)), 0.0)
            if self.act.get('child') and P.child_cost > 0 and H.kids[self.pl] == 0:
                H.kids[self.pl] = P.child_years
                self.act['child'] = False
        self._vat_i = None
        if P.vat_bi and P.g_tr > 0 and self.Y_prev and getattr(self, '_Bsum_last', None):
            vat = P.g_tr * self.Y_prev / self._Bsum_last          # 去年の消費の総額で、給付の額を賄う税率
            self._vat_i = self._B * vat / (1 + vat)
            self._B = self._B / (1 + vat)
            self._vat = vat
        self._Bsum_last = float(self._B.sum())
        return self._inv, self._B

    def step(self):
        """一国の一年: 年のはじめ、価格を解く(年の中の釣り合い)、支払いを書く、帳簿を閉じる。"""
        inv, B = self.begin_year()
        price, sol = self.solve_prices(inv, B)
        x, u, geff, cost, m = self.evaluate(price, inv, B)
        qc = x[:2] if self._cx is None else np.array([self._cx['hhN'], x[1]])
        rec = self.finish_year(price, price, x, u, geff, cost, qc, float(np.linalg.norm(sol.fun[:3])))
        rec['books'] = self.L.check(tol=1e-6 * max(1.0, abs(self.L.stock[(self.sec['gov'], 'bonds')])))['max']
        self.L.close()
        self.log.append(rec)
        return rec

    def route(self, flow, payer, good, amount):
        """支払いを、財 good(0 = 必需品、1 = 贅沢品、2 = 原料)を売る企業に振り分ける。一国では自国の企業だけ。
        自分から自分への支払い(原料の企業が自分の原料を使う)は、帳簿に書かない。世界の模型は、これを差し替える。"""
        to = self.firms[good]
        if to != payer:
            self.L.pay(flow, payer, to, amount)

    def finish_year(self, price, pbuy, x, u, geff, cost, qc, resid):
        """支払いを書き、標本と状態を更新する。price は自国の企業が売る価格、pbuy は自国の買い手が払う価格
        (一国では同じ。世界の模型では、輸入と混ぜた価格の指数)。x は自国の企業の生産(輸出を含む)、
        qc は自国の家計の消費の量(必需品、贅沢品。買う価格の単位)。"""
        P, L, H = self.p, self.L, self.H
        inv, B = self._inv, self._B
        hh, bank, gov = self.sec['hh'], self.sec['bank'], self.sec['gov']
        F = self.firms
        PN, PL, PI = price
        bN, bL, bI = pbuy
        W = self.W
        S = P.pop_w / (1.0 + np.exp(-P.beta_s * (np.log(W / bN) - np.log(P.w_res))))
        ret = (H.age >= P.ret_age) if P.ret_age > 0 else np.zeros(len(H.M), bool)
        pop_eff = P.pop_w
        if P.ret_age > 0:
            ws = float((H.lab * ~ret).sum() / H.lab.sum())
            S = S * ws                                         # 引退した世帯の分、働ける人が減る
            pop_eff = P.pop_w * ws                             # 失業者は、働く年齢の人の中で数える
        Dl = float(geff @ x)
        L_R = np.zeros(3)
        if P.research:
            budget = self.sR * price * x                       # 研究に使う額(売上の割合)
            if self.fp is not None:
                res_k = self.fp.budgets(price * x)            # 企業ごとの研究の額(規則的な企業は利益で縛らない)
                budget = res_k.sum(1)
            elif self.prof_R_last is not None:
                budget = np.minimum(budget, P.cap_R * np.maximum(self.prof_R_last, 0.0))   # 去年の利益で縛る
            if P.res_loan > 0 and P.lending != 'none' and P.credit:
                budget = budget * ((1 - P.res_loan) + P.res_loan * self.lend_Ax)   # 段 15c: 借りて賄う探索は、貸し手が承認した分だけ
            L_R = budget / W                                   # 研究に雇う人
            Dl = Dl + float(L_R.sum())
        prot_L = 0.0
        if P.disaster:
            inj_share = float(H.lab @ H.inj / H.lab.sum())
            S = S * (1 - inj_share)                            # 怪我をした分、働ける人が減る
            prot_L = P.protect * self.Y_last / W               # 守りの仕事に政府が雇う人
            Dl = Dl + prot_L
        pub_L = P.s_pub * self.Y_prev / W if (P.s_pub > 0 and self.Y_prev) else 0.0   # 公の研究所の人
        Dl = Dl + pub_L
        E = softmin(Dl, S)
        unemp = pop_eff - E
        wages_j = W * geff * x
        cx = self._cx
        if cx is not None and P.f_K > 0:
            wages_j = wages_j + W * P.f_K * self.K                # 維持の労働
            Dl_fix = float(P.f_K * self.K.sum())
            Dl = Dl + Dl_fix
            E = softmin(Dl, S)
            unemp = pop_eff - E
        ben = P.benefit * P.q0 * bN * unemp
        profit = (price - cost) * x
        subsidy, share_pay = 0.0, np.zeros(3)
        if cx is not None:
            profit[0] = PN * cx['firm_sales'] - cost[0] * x[0]   # 必需品の企業は、売った分で稼ぎ、作った分を払う(差は在庫)
            if P.f_K > 0:
                profit = profit - W * P.f_K * self.K              # 維持の労働(固定費)
            if P.crisis_rule in ('last', 'share') and profit[0] < 0:
                if P.crisis_rule == 'last':
                    subsidy = -profit[0]                          # 最後の生産者: 政府が赤字を埋める
                else:
                    rev = price[1:] * x[1:]
                    share_pay[1:] = -profit[0] * rev / rev.sum()  # 損失の分担: 他の部門が売上に応じて埋める
                profit[0] = 0.0
        inv_cost_j = bI * self.kap_new * inv
        # ---- 信用: 借入、利子、銀行の利益
        rL, rD = self.r_loan, (self.r_policy if P.credit else 0.0)
        loan_loss = np.zeros(3)
        if P.lending != 'none' and getattr(self, 'lend_loss_pending', None) is not None:
            loan_loss = np.minimum(self.lend_loss_pending, self.loans)   # 段 15b: 去年失敗した事業への貸出の損
            for j in range(3):
                L.revalue('loans', bank, -loan_loss[j]); L.revalue('loans', F[j], loan_loss[j])
            self.loans = self.loans - loan_loss
            self.lend_loss_pending = None
        if self.fp is not None and P.lender != 'none' and self.fp.loss_pending.sum() > 0:
            # 段 15: 去年倒産した企業への貸出の損を、銀行が書く(企業の部門の借金が減り、銀行の利益が減る)
            loan_loss = np.minimum(self.fp.loss_pending, self.loans)
            for j in range(3):
                L.revalue('loans', bank, -loan_loss[j]); L.revalue('loans', F[j], loan_loss[j])
            self.loans = self.loans - loan_loss
            self.fp.loss_pending = np.zeros(3)
        if P.k_mem:
            self.loss_mem = P.omega_mem * getattr(self, 'loss_mem', 0.0) + (1 - P.omega_mem) * float(loan_loss.sum()) / max(self.Y_prev or 1.0, 1e-9)
        Lf = self.loans
        B_bank = L.stock[(bank, 'bonds')]
        if P.credit:
            cap = P.lam * bI * self.kap_avg * self.K
            if self.fp is not None and P.lender not in ('none', 'blind'):
                cap = Lf + (cap - Lf) * self.fp.cbar                  # 段 15: 新しく借りられるのは、貸し手が貸す気のある割合だけ
            dL = np.clip((1 - P.f_R) * inv_cost_j - P.amort * Lf, -Lf, np.maximum(cap - Lf, 0.0))
            if getattr(self, 'cf', 1.0) < 1.0:
                dL = np.where(dL > 0, dL * self.cf, dL)                # 段 16: 準備が足りなければ、新しい貸出を絞る
        else:
            dL = np.zeros(3)
        up_pay = 0.0
        if P.up_gen > 0 and self.up_s > 0 and P.up_sub > 0:
            up_pay = P.up_sub * self.up_s * W * self.up_gX * x[1]      # 段 B7: 工房の労働の費用の一部を政府が埋める
            profit = profit + np.array([0.0, up_pay, 0.0])
        int_f = rL * Lf
        self.prof_R_last = profit - int_f
        rent = P.rho_T * PN * x[0] if P.land else 0.0
        roy = P.phi_R * max(profit[1], 0.0) if P.land else 0.0
        res_w = W * L_R
        divj = profit - int_f - inv_cost_j + dL - np.array([0.0, roy, 0.0]) - res_w - share_pay
        div = divj.sum()
        depo, debt = np.maximum(H.M, 0.0), np.maximum(-H.M, 0.0)
        rB = getattr(self, 'r_bond', rD) if P.credit else rD
        depo = np.maximum(depo - H.hd, 0.0) if P.hoard else depo            # 死蔵の分(銀行の外の現金)には利子がつかない(段 16)
        int_D, int_HL, int_B = rD * (depo.sum() + self.pf_deposits()), rL * debt.sum(), rB * B_bank   # 積立の残高も預金として利子がつく
        bank_profit = int_f.sum() + int_HL + int_B - int_D - self.xf.get('int_nr', 0.0) - loan_loss.sum()   # int_nr: 地域の外の持ち主の預金への利子
        cap_pre = div + bank_profit
        tau_k = self.tau if P.tau_k < 0 else P.tau_k
        tax_w, tax_c = self.tau * (wages_j.sum() + res_w.sum()), tau_k * max(cap_pre, 0.0)
        tax_c_eq = tax_c
        tax_c += tau_k * (rent + roy)
        tr = P.redistrib * tax_c
        if P.g_tr > 0 and self.Y_prev:
            tr = tr + P.g_tr * self.Y_prev                    # 一人あたり同額の給付(公共事業と比べるため。国債で賄う)
        # 行き詰まりの道(前の年の国債/所得が閾値を超えたとき)
        b_last = getattr(self, 'b_last', 0.0)
        crisis = b_last > P.theta_resolve
        levy_i = np.zeros(len(H.M))
        lrate = (P.levy_rate if (crisis and P.resolve == 'levy') else 0.0) + P.wealth_tax
        if lrate > 0:
            nw = H.M + self.pT * H.land + (self.qIP or 0.0) * H.ip + self.v_eq_last * H.eq + P.enf_F * H.F   # 外の資産は見える分だけ
            pc = nw / H.heads
            ex = P.levy_exempt * max(nw.sum() / H.heads.sum(), 1e-12)
            levy_i = lrate * np.maximum(pc - ex, 0.0) * H.heads
        if crisis and P.resolve == 'austerity':
            tax_w = tax_w + P.austerity * wages_j.sum()
        levy = float(levy_i.sum())
        # ---- 帳簿への書き込み(家計はマクロでは一つの部門 hh。標本ごとの内訳は H で持つ)
        cons = [bN * qc[0], bL * qc[1]]
        for j in range(3):
            L.pay('wages', F[j], hh, wages_j[j])
            L.pay('research wages', F[j], hh, res_w[j])
        gov_buy = 0.0
        if cx is not None:
            # 家計の必需品の支払いは、企業の売った分と、政府の備蓄の放出の分に分かれる
            rel_hh = min(cx['rel'], qc[0])
            self.route('consumption', hh, 0, bN * (qc[0] - rel_hh))
            L.pay('stockpile release', hh, gov, bN * rel_hh)
            self.route('stockpile purchase', gov, 0, bN * cx['gbuy'])        # 政府も買い手の価格で買う(一国では売る価格と同じ)
            self.route('public works', gov, 2, bI * cx['GI'])
            L.pay('production subsidy', gov, F[0], subsidy)
            for j in (1, 2):
                L.pay('loss sharing', F[j], F[0], share_pay[j])
            gov_buy = bN * cx['gbuy'] + bI * cx['GI'] + subsidy - bN * rel_hh
            dInv = x[0] - cx['firm_sales'] if P.inventory else 0.0
            self.InvN = (cx['Inv'] + dInv) if P.inventory else None
            L.produce_real('stock_N' + self.sfx, F[0], dInv)
            dS = cx['gbuy'] - cx['rel'] - P.delta_S * (self.S_G + cx['gbuy'] - cx['rel'])
            L.produce_real('stock_G' + self.sfx, gov, dS)
            self.S_G = self.S_G + dS
        else:
            self.route('consumption', hh, 0, cons[0])
        self.route('consumption', hh, 1, cons[1])
        for j in range(3):
            self.route('intermediate', F[j], 2, bI * P.alpha[j] * x[j])
            self.route('investment goods', F[j], 2, inv_cost_j[j])
            L.produce_real(self.real[j], F[j], -P.delta * self.K[j] + inv[j])
            L.trade('loans', bank, F[j], dL[j])
            L.pay('loan interest', F[j], bank, int_f[j])
            L.pay('dividends', F[j], hh, divj[j])
        L.pay('land rent', F[0], hh, rent)
        L.pay('royalties', F[1], hh, roy)
        L.pay('deposit interest', bank, hh, int_D)
        L.pay('loan interest', hh, bank, int_HL)
        L.pay('bond interest', gov, bank, int_B)
        L.pay('dividends', bank, hh, bank_profit)
        L.pay('benefits', gov, hh, ben)
        L.pay('transfers', gov, hh, tr)
        # 段 6: 外の資産の利子(窓口 ext から)、外の資産の売り買い、財政同盟の給付
        xf = self.xf
        incF, dF = xf.get('incF', 0.0), xf.get('dF')
        ben_U, union_in = xf.get('union_ben', 0.0), xf.get('union_in', 0.0)
        incF_i = incF * H.F / H.F.sum() if H.F.sum() > 0 else np.zeros(len(H.M))
        tax_F = tau_k * P.enf_F * incF
        if 'ext' in self.sec:
            L.pay('foreign income', self.sec['ext'], hh, incF)
            if dF is not None:
                L.trade('foreign', hh, self.sec['ext'], float(dF.sum()))
        L.pay('union benefits', gov, hh, ben_U)
        tax_c = tax_c + tax_F
        prot_w = W * prot_L
        L.pay('protection', gov, hh, prot_w)
        pub_w = W * pub_L
        L.pay('public research', gov, hh, pub_w)
        prot_w = prot_w + pub_w                               # 以下、政府が雇う人の賃金として、守りと同じに扱う
        tax_w = tax_w + self.tau * prot_w
        # 段 10: 年金(賦課方式・貢献に比例)、児童手当、消費税
        pen_i = np.zeros(len(H.M))
        if P.pension in ('payg', 'points') and ret.any():
            wh = float(H.heads[~ret].sum())
            wbar = wages_j.sum() / max(wh, 1e-12)               # 働く人一人あたりの賃金
            pen_tot = P.pen_rate * wbar * float(H.heads[ret].sum())
            if P.pension == 'payg':
                pen_i = np.where(ret, pen_tot * H.heads / H.heads[ret].sum(), 0.0)
            else:
                w_ = np.where(ret, H.pts, 0.0)
                pen_i = pen_tot * w_ / w_.sum() if w_.sum() > 0 else np.where(ret, pen_tot * H.heads / H.heads[ret].sum(), 0.0)
        ca_i = P.child_allow * P.q0 * bN * H.heads * (H.kids > 0) if P.child_allow > 0 else np.zeros(len(H.M))
        L.pay('pensions', gov, hh, float(pen_i.sum()))
        L.pay('child allowance', gov, hh, float(ca_i.sum()))
        vat_i = self._vat_i if self._vat_i is not None else np.zeros(len(H.M))
        L.pay('consumption tax', hh, gov, float(vat_i.sum()))
        gov_buy = gov_buy + float(pen_i.sum() + ca_i.sum() - vat_i.sum())
        L.pay('workshop subsidy', gov, F[1], up_pay)
        gov_buy = gov_buy + up_pay
        L.pay('taxes', hh, gov, tax_w + tax_c)
        L.pay('wealth tax', hh, gov, levy)
        deficit = ben + tr + int_B + prot_w - tax_w - tax_c - levy + ben_U - union_in + gov_buy
        L.trade('bonds', bank, gov, deficit)          # 赤字は国債で賄い、銀行が買う(預金が増える)
        self.loans = Lf + dL
        # ---- 標本の所得と預金(マクロの総額を、労働と持分の取り分で配る)
        labw = H.lab
        if P.disaster:
            labw = H.lab * (1 - H.inj)
            labw = labw / labw.sum()                           # 働く力の取り分(怪我の分を除く)
        if P.ret_age > 0:
            labw = labw * ~ret
            labw = labw / labw.sum()                           # 引退した世帯は、賃金も失業の給付も受け取らない
        if self.pl is not None and float(self.act.get('work', 1.0)) != 1.0:
            labw = labw.copy()
            labw[self.pl] *= float(np.clip(self.act.get('work', 1.0), 0.0, 1.0))   # 段 19: プレイヤーが働く度合い(働かない分は、ほかの人が働く)
            labw = labw / labw.sum()
        lab_inc = (wages_j.sum() + res_w.sum() + prot_w + ben + ben_U) * labw + tr * H.heads / H.heads.sum() + pen_i + ca_i
        cap_inc = cap_pre * H.eq + rD * depo - rL * debt
        tax_i = tax_c_eq * H.eq
        cap_inc = cap_inc + incF_i
        tax_i = tax_i + tau_k * P.enf_F * incF_i
        if P.land:
            land_inc = rent * H.land / P.T_land + (roy * H.ip / H.ip.sum() if H.ip.sum() > 0 else roy * H.eq)
            cap_inc = cap_inc + land_inc
            tax_i = tax_i + tau_k * land_inc
        self.cap_inc_last = cap_inc
        Yd = lab_inc - self.tau * (wages_j.sum() + res_w.sum() + prot_w) * labw + cap_inc - tax_i - levy_i
        if crisis and P.resolve == 'austerity':
            Yd = Yd - P.austerity * wages_j.sum() * labw
        spend = B                                           # 予算はすべて使う(作り方から、cons の和 = B の和)
        hungry, sub_gap, hungry_kids, hungry_ret = 0.0, 0.0, 0.0, 0.0
        if cx is not None:
            # 割り当て: 品不足なら、まず生存の量(一人 q0)を配り、残りを足りない分に比例して配る。買えなかった分は預金に残る(強いられた貯蓄)
            sh = self.subs_heads()
            base = bN * sh * P.q0                                  # 家計が払うのは買い手の価格(一国では売る価格と同じ)
            disc = softplus(B - base, 10.0 / np.maximum(base, 1e-9))
            dN_i = (B - disc) / bN + self.b_vec() * disc / bN
            avail = qc[0]
            got = dN_i
            if avail < dN_i.sum() * (1 - 1e-12):
                sub = np.minimum(dN_i, sh * P.q0)
                if avail <= sub.sum():
                    got = sub * avail / sub.sum()
                else:
                    got = sub + (dN_i - sub) * (avail - sub.sum()) / (dN_i - sub).sum()
                spend = B - (dN_i - got) * bN
            # 生存の量に届かない人: 品不足の割り当てでも、値段が高くて買えないのでも数える
            hungry = float(H.heads[got < sh * P.q0 * (1 - 1e-6)].sum() / H.heads.sum())
            sub_gap = float(np.maximum(sh * P.q0 - got, 0.0).sum() / (sh.sum() * P.q0))
            kid = H.kids > 0
            hungry_kids = float(H.heads[kid & (got < sh * P.q0 * (1 - 1e-6))].sum() / max(H.heads[kid].sum(), 1e-12)) if kid.any() else 0.0
            hungry_ret = float(H.heads[ret & (got < sh * P.q0 * (1 - 1e-6))].sum() / max(H.heads[ret].sum(), 1e-12)) if ret.any() else 0.0
            self.N_last = qc[0]
        if P.pension == 'funded' and P.ret_age > 0:
            # 積立: 働く世帯は労働所得の一部を積み立て(引き出せない預金)、引退した世帯は取り崩して所得にする。預金の総額は変わらない
            contrib = P.fund_rate * np.maximum(lab_inc, 0.0) * ~ret
            draw = P.fund_draw * H.pf * ret
            if self.fund_mode():
                # 段 10b: 基金の預金に、利子・持分の配当(税を引いて)・積立が入り、取り崩しが出る。請求権は株の売り買いのあとで値打ちに合わせる
                self.fund_D = self.fund_D * (1 + rD) + self.fund_e * (cap_pre - tax_c_eq) + float(contrib.sum() - draw.sum())
                H.pf = H.pf + contrib - draw
            else:
                H.pf = H.pf * (1 + rD) + contrib - draw
            Yd = Yd - contrib + draw
        elif H.pf.sum() > 0:
            H.pf = H.pf * (1 + rD)
        if P.pension == 'points' and P.ret_age > 0:
            H.pts = H.pts + np.maximum(lab_inc - pen_i - ca_i, 0.0) * ~ret
        H.M = H.M + Yd - spend - vat_i
        if dF is not None:
            H.M = H.M - dF                                  # 外の資産を買う(負なら売る)
            H.F = H.F + dF
        H.Yd = Yd
        if P.hoard:
            # 段 16: 一人あたりの預金が平均の hoard_cap 倍を超えた分を死蔵する(ノート 3 の S9: 死蔵 = max(s − mB/W, 0))
            dep = np.maximum(H.M, 0.0)
            mean_pc = dep.sum() / H.heads.sum()
            H.hd = P.hoard_frac * np.maximum(dep - P.hoard_cap * mean_pc * H.heads, 0.0)
            self.hoard_tot = float(H.hd.sum())
            D_tot = float(dep.sum() + self.pf_deposits())
            if P.reserves == 'fixed':
                Rbar = P.res_supply * Y_for_res if (Y_for_res := (self.Y_prev or 0.0)) else P.res_supply * 1e9
                self.cf = float(np.clip((Rbar - self.hoard_tot) / max(P.rho_res * (D_tot - self.hoard_tot), 1e-12), 0.0, 1.0))
            else:
                self.cf = 1.0
        H.asset_share = np.where(np.abs(lab_inc) + np.abs(cap_inc) > 0,
                                 np.maximum(cap_inc, 0.0) / np.maximum(lab_inc + np.maximum(cap_inc, 0.0), 1e-12), 0.0)
        if P.luck_sigma > 0 or P.luck_skill > 0:
            # 段 9: 家計ごとの資産の運と腕。持分の取り分の対数を揺らし、和を 1 に戻す(配当の総額は変わらず、配り先だけが変わる)
            n_ = len(H.M)
            if P.luck_nu > 2:
                z = self.luck_rng.standard_t(P.luck_nu, n_) * np.sqrt((P.luck_nu - 2) / P.luck_nu)
            else:
                z = self.luck_rng.standard_normal(n_)
            sc = np.sqrt(H.heads.mean() / H.heads)            # 人数の多い記入ほど、運は平均されて小さい
            if self.pl is not None:
                sc[self.pl] = 1.0                              # 段 19: プレイヤーの運は、ふつうの記入と同じ大きさ(遊びの都合)
            if P.luck_skill > 0:
                H.skill = P.luck_rho * H.skill + P.luck_skill * np.sqrt(1 - P.luck_rho ** 2) * self.luck_rng.standard_normal(n_)
            g_ = np.clip(P.luck_sigma * sc * z + H.skill, -5.0, 5.0)
            tot = H.eq.sum()
            H.eq = H.eq * np.exp(g_)
            H.eq = H.eq * tot / H.eq.sum()
        if P.child_cost > 0:
            H.kids = np.maximum(H.kids - 1, 0.0)
        if P.ret_age > 0 or P.age_track:
            H.age = H.age + 1.0
        land_rec = {}
        if P.land:
            shq = self.subs_heads() * P.q0
            lux_i = (1 - self.b_vec()) * softplus(B - bN * shq, 10.0 / np.maximum(bN * shq, 1e-9))
            ip_new = P.eta_IP * P.sigma_R * lux_i / bL
            H.ip = H.ip * (1 - P.delta_IP) + ip_new
            land_rec = self.land_market(rent, roy, rL, W, wages_j.sum(), E, Yd, lab_inc, div, bN, bL)
            land_rec['ip_new'] = float(ip_new.sum())
        # ---- 能力、技術、競争、賃金、財政(遅くない層だが、年ごとの更新)
        rho = profit / (bI * self.kap_avg * self.K) - P.delta
        K_old = self.K.copy()
        self.K = (1 - P.delta) * self.K + inv
        if P.lending != 'none' and P.credit:
            # 段 15b: 借りて賄った事業のうち、失敗した分の能力は失われ、その借りた分の lgd_p が次の年に貸倒れる
            fail = (1 - P.f_R) * inv * (1 - self.lend_succ)
            self.K = self.K - fail
            for j in range(3):
                L.revalue(self.real[j], F[j], -fail[j])
            self.lend_loss_pending = P.lgd_p * bI * self.kap_new * fail
            self.x_inv_prev = float(inv.sum() / max((P.delta * K_old).sum(), 1e-12))
        scrap = np.zeros(3)
        if P.delta_scrap > 0:
            # 赤字の部門は能力を捨てる(最後の生産者のもとでは、必需品の能力は捨てない)
            scrap = P.delta_scrap * np.clip(-(rho + P.delta) / 0.1, 0.0, 1.0) * self.K
            if P.crisis_rule == 'last':
                scrap[0] = 0.0
            self.K = self.K - scrap
            for j in range(3):
                L.revalue(self.real[j], F[j], -scrap[j])
        self.gamma_new *= (1 - P.g_gamma)
        gains = np.zeros(3)
        fp_rec = {}
        if self.fp is not None:
            dG = self.fp.update(res_k, W, profit - int_f, wages_j, bI * self.kap_avg * self.K, pub_L, loans=self.loans)
            self.gamma_new *= np.exp(-dG)
            gains = dG
            fp_rec = self.fp.last
        elif P.research:
            # 研究の当たり: 試行の数 × 当たる確率 の回数だけ、パレート型の大きさの向上が出る(上限つき)
            for j in range(3):
                lam_j = L_R[j] / P.trial_cost * P.p_hit * (min(self.variety() ** P.culture_spill, P.spill_cap) if P.culture_spill else 1.0)
                if not np.isfinite(lam_j):
                    raise FloatingPointError('研究の試行の数が数でない')
                hits = self.res_rng.poisson(max(lam_j, 0.0))
                if hits > 1_000_000:
                    # 当たりが多すぎると配列が膨らむので、上限つきのパレートの期待値で足す(メモリの安全弁)
                    a = P.alpha_R
                    gains[j] = hits * (P.g0 + P.g0 ** a * (P.g_cap ** (1 - a) - P.g0 ** (1 - a)) / (1 - a))
                elif hits > 0:
                    g = P.g0 * (1 - self.res_rng.random(hits)) ** (-1 / P.alpha_R)
                    gains[j] = float(np.minimum(g, P.g_cap).sum())
            if P.auto_share > 0:
                # 自動化の当たり(当たりの auto_share)は、費用が下がるときだけ使う: 労働の節約 W γ (1 − e^{−g}) が、
                # 資本の費用の増え P_I κ (e^{λg} − 1)(r̄ + δ) を上回るとき(賃金が資本財に比べて高いほど、自動化が引き合う)
                ga = P.auto_share * gains
                save = W * self.gamma_new * (1 - np.exp(-ga))
                extra = PI * self.kap_new * (np.exp(P.auto_kappa * ga) - 1) * (self.rbar + P.delta)
                take = save > extra
                self.auto_taken = getattr(self, 'auto_taken', 0.0) + float(ga[take].sum())
                self.auto_refused = getattr(self, 'auto_refused', 0.0) + float(ga[~take].sum())
                gains = np.where(take, gains, gains - ga)             # 引き合わない自動化の当たりは使わない
                self.kap_new = self.kap_new * np.exp(P.auto_kappa * np.where(take, ga, 0.0))   # 自動化: 新しい能力は資本をより多く使う
            self.gamma_new *= np.exp(-gains)
            if P.adaptive_R:
                # 見返りの比 b = 見込み × 生産の労働 × h_R(当たりは新しい生産能力の労働を減らすので、生産の労働に比例して効く)
                Lp = np.maximum(geff * x, 1e-9)
                if self.g_hat is None:
                    b0 = P.s_R / max(P.s_R_max - P.s_R, 1e-9)
                    self.g_hat = b0 / (Lp * P.h_R)
                per = gains / np.maximum(L_R, 1e-12)
                self.g_hat = P.omega_R * self.g_hat + (1 - P.omega_R) * np.where(L_R > 0, per, self.g_hat)
                b = self.g_hat * Lp * P.h_R
                self.sR = np.maximum(P.s_R_min, P.s_R_max * b / (1 + b))
        self.gamma = ((1 - P.delta) * K_old * self.gamma + inv * self.gamma_new) / self.K
        if P.auto_share > 0:
            self.kap_avg = ((1 - P.delta) * K_old * self.kap_avg + inv * self.kap_new) / np.maximum((1 - P.delta) * K_old + inv, 1e-12)
        excess = 0.0
        if self.fp is not None and P.mu_hhi > 0:
            excess = P.mu_hhi * ((self.fp.s ** 2).sum(1) - 1.0 / self.fp.N)   # 集中した部門の、上乗せの利益の率
        m0_old = self.m0.copy()
        self.m0 = np.clip(self.m0 - P.eta * (rho - self.rbar - excess), 0.01, 2.0)
        if cx is not None and cx['capped']:
            self.m0[0] = m0_old[0]                                 # 上限が効いている年は、必需品の値上げ幅の目標を動かさない(積み上がりを防ぐ)
        infl = np.log(bN / self.PN_prev)
        self.infl_prev = infl
        self.W = W * np.exp(P.phi_w * (Dl / S - 1.0) + P.chi * infl)
        Y = cons[0] + cons[1] + bI * self.kap_new @ inv
        if cx is not None:
            Y = bN * qc[0] + cons[1] + bI * self.kap_new @ inv + bN * cx['gbuy'] + bI * cx['GI'] - bN * min(cx['rel'], qc[0])
            if P.inventory:
                Y = Y + PN * (x[0] - cx['firm_sales'])
        self.Y_prev = Y
        if P.disaster:
            self.Y_last = Y
            self.lab_inc_last = lab_inc
            self.prot = self.prot * (1 - P.prot_decay) + prot_L
            H.inj = H.inj * (1 - P.rec_inj)
        if P.fiscal == 'debt':
            gdebt = -L.stock[(gov, 'bonds')]
            self.tau = float(np.clip(self.tau + P.k_debt * (gdebt / Y - P.b_target), 0.0, 0.6))
        elif P.fiscal == 'anchor':
            b = -L.stock[(gov, 'bonds')] / Y
            if self.t < P.anchor_start:
                # 最初の移り変わりは赤字のルールで動き、その経済の自然な国債/所得の水準を見つける
                self.b_ref = b
                self.tau = float(np.clip(self.tau + P.k_tau * (deficit / Y - P.deficit_target), 0.0, 0.6))
            else:
                # そのあとは、見つけた水準を目標に、国債/所得だけに反応する(打撃で目減りしたら戻す)
                self.tau = float(np.clip(self.tau + P.k_debt * (b - self.b_ref), 0.0, 0.6))
        elif P.fiscal == 'burden':
            # 利払いの重さを見る: 許される赤字/所得 = 目標の重さ − 今の重さ(金利が低いほど、借りる余地が広い)
            gdebt = -L.stock[(gov, 'bonds')]
            allowed = P.burden_target - rB * gdebt / Y
            self.tau = float(np.clip(self.tau + P.k_tau * (deficit / Y - allowed), 0.0, 0.6))
        elif P.fiscal == 'real':
            # 物価で直した赤字(赤字 − 物価上昇率 × 国債)を 0 に: 国債の実質の値打ちを保つ
            gdebt = -L.stock[(gov, 'bonds')]
            d_rule = deficit - (np.exp(infl) - 1) * (gdebt - deficit)
            self.tau = float(np.clip(self.tau + P.k_tau * (d_rule / Y - P.deficit_target), 0.0, 0.6))
        else:
            d_rule = deficit if P.fiscal == 'deficit' else deficit - ben
            self.tau = float(np.clip(self.tau + P.k_tau * (d_rule / Y - P.deficit_target), 0.0, 0.6))
        self.rho_prev, self.u_prev = rho, u
        self.profit_prev = float(profit.sum())
        self.PN_prev = bN
        self.prices = price
        self.pbuy = pbuy
        dist = self.dist_stats(Yd)
        rbar_used, prem_used, rL_used = self.rbar, self.prem, self.r_loan
        self.top10_prev = dist['top10_inc']
        # 国債/所得(来年の上乗せ金利と、行き詰まりの道の判断に使う)。踏み倒しは、ここで一度
        gdebt = -L.stock[(gov, 'bonds')]
        defaulted = 0.0
        if P.resolve == 'default' and gdebt / Y > P.theta_resolve and not getattr(self, 'in_default', False):
            cut = P.haircut * L.stock[(bank, 'bonds')]
            L.revalue('bonds', bank, -cut); L.revalue('bonds', gov, +cut)
            # 銀行の自己資本は 0 なので、損は預金者が預金に応じて負う
            dep = np.maximum(H.M, 0.0)
            take = min(cut, dep.sum())
            H.M = H.M - take * dep / max(dep.sum(), 1e-12)
            L.revalue('deposits', hh, -take); L.revalue('deposits', bank, +take)
            if cut > take:                                     # 預金で埋めきれない分は、銀行の持ち主(持分)の負担として借金に
                H.M = H.M - (cut - take) * H.eq
                L.revalue('deposits', hh, -(cut - take)); L.revalue('deposits', bank, +(cut - take))
            defaulted = cut
            self.in_default = True
            gdebt = -L.stock[(gov, 'bonds')]
        if gdebt / Y < 0.8 * P.theta_resolve:
            self.in_default = False
        self.b_last = gdebt / Y
        self.v_eq_last = max(div, 0.0) / max(self.rbar, 1e-3)
        self.set_rates()
        sample_gap = max(abs(H.M.sum() + self.pf_deposits() - L.stock[(hh, 'deposits')]), abs(H.F.sum() - L.stock[(hh, 'foreign')]))
        # 段 B7: 格上げの工房(ライトの法則で費用が下がり、古い企業より安ければ割合が広がる)
        up_adv, upgraded = 0.0, 0.0
        if P.up_gen > 0 and self.up_s > 0:
            self.up_cum += self.up_s * x[1]
            gX = max(P.up_g0 * (self.up_cum / P.up_cum0) ** (-P.up_wright), P.up_floor)
            up_adv = float(np.log(self.gamma[1] / (gX * (1 - P.up_sub))))
            s_new = min(max(self.up_s + P.up_speed * self.up_s * (1 - self.up_s) * up_adv, 0.0), 0.99)
            if self.up_peak < 0.5 <= s_new:
                upgraded = 1.0                                       # 格上げ: 工房が贅沢品の半分を超えた
            self.up_peak = max(self.up_peak, s_new)
            self.up_s = s_new if s_new >= 1e-4 else 0.0              # 小さくなりすぎた工房は消える
            self.up_gX = gX
        # 段 B7: 革命(移り変わりの痛みが積もった不満が閾値を超えると、政策の組が入れ替わる)
        revolution = 0.0
        if P.revolt:
            brk = land_rec.get('breakups', 0.0) if land_rec else 0.0
            if self.rev_prev is not None:
                E0, w0, h0 = self.rev_prev
                pain = (P.rev_wE * max(0.0, 1 - E / E0) + P.rev_ww * max(0.0, -np.log((W / bN) / w0))
                        + P.rev_wh * max(0.0, hungry - h0) + P.rev_wb * brk)
                self.rev_G = P.rev_mem * self.rev_G + pain
            self.rev_prev = (E, W / bN, hungry)
            if self.rev_G > P.rev_thr and self.t - self.rev_last >= P.rev_cool:
                keys = ('tau_k', 'redistrib', 'wealth_tax')
                alt = self.rev_alt if self.rev_alt is not None else dict(tau_k=P.rev_tau_k, redistrib=P.rev_redistrib, wealth_tax=P.rev_wealth_tax)
                self.rev_alt = {k: getattr(P, k) for k in keys}
                P.set(**alt)
                self.regime = 1 - self.regime
                self.rev_last = self.t
                self.rev_G = 0.0
                revolution = 1.0
        rec = dict(t=self.t, W=W, PN=PN, PL=PL, PI=PI, bN=bN, bL=bL, bI=bI, real_wage=W / bN, S=S, D=Dl, E=E, unemp=unemp,
                   xN=x[0], xL=x[1], xI=x[2], uN=u[0], uL=u[1], uI=u[2], KN=K_old[0], KL=K_old[1], KI=K_old[2],
                   rhoN=rho[0], rhoL=rho[1], rhoI=rho[2], tau=self.tau, M_w=H.M[0], M_c=H.M[1], deficit=deficit,
                   G=-L.stock[(gov, 'bonds')], Y=Y, resid=resid, sample_gap=sample_gap, rbar=rbar_used, infl=infl, r_policy=rD, prem=prem_used, r_loan=rL_used,
                   loans=float(Lf.sum()), dL=float(dL.sum()), bank_profit=bank_profit, div=div, tax_c=tax_c, transfer=tr,
                   inv=float(inv_cost_j.sum()), inv_limited=self.inv_limited, rent=rent, roy=roy, **dist, **land_rec,
                   prot=self.prot, prot_L=prot_L, inj_share=float(H.lab @ H.inj / H.lab.sum()),
                   dis_size=self.dis_rec.get('dis_size', 0.0), relief=self.dis_rec.get('relief', 0.0),
                   K_lost=self.dis_rec.get('K_lost', 0.0), b=self.b_last, premB=getattr(self, 'premB', 0.0),
                   r_bond=rB, levy=levy, crisis=float(crisis), defaulted=defaulted, prem_income=(rB - rD) * B_bank,
                   belief=getattr(self, 'belief', 0.0), protect=P.protect, tau_k=P.tau_k,
                   L_R=float(L_R.sum()), res_w=float(res_w.sum()), gain=float(gains.sum()), sR=float(np.mean(self.sR)),
                   gamma_front=float(np.mean(self.gamma_new)), pub_L=pub_L, **fp_rec,
                   **({} if P.lending == 'none' else dict(lend_A=self.lend_A, lend_ps=self.lend_ps, x_inv=getattr(self, 'x_inv_prev', 1.0), trust=self.trust(),
                                                          lend_loss=float(loan_loss.sum()))),
                   **({} if not P.hoard else dict(hoard=self.hoard_tot, cf=self.cf, deposits=float(np.maximum(H.M, 0.0).sum()))),
                   wage_share=float((wages_j.sum() + res_w.sum()) / Y) if Y else np.nan, kap_avg=float(np.mean(self.kap_avg)),
                   acct_use=float(bN * qc[0]), acct_lux=float(cons[1]), acct_save=float((Yd - spend).sum()),
                   acct_explore=float(P.sigma_R * cons[1] + res_w.sum()) if P.land else float(res_w.sum()),
                   cap_Y=float(bI * self.kap_avg @ self.K / Y) if Y else np.nan,
                   **({} if P.auto_share == 0 else dict(auto_taken=getattr(self, 'auto_taken', 0.0), auto_refused=getattr(self, 'auto_refused', 0.0))),
                   **({} if P.ret_age == 0 and P.child_allow == 0 and not P.vat_bi else
                      dict(ret_share=float(H.heads[ret].sum() / H.heads.sum()), pensions=float(pen_i.sum()), child_allow=float(ca_i.sum()),
                           vat=float(vat_i.sum()), pf=float(H.pf.sum()))),
                   **({} if P.psi_v == 0 else dict(n_var=self.variety(), b_share=self.b_share(),
                                                   lux_eff=qc[1] * self.variety() ** (1 / (P.sigma_v - 1)))),
                   **({} if cx is None else dict(short=cx['short'] / max(cx['dN'], 1e-12), hungry=hungry, sub_gap=sub_gap, hungry_kids=hungry_kids, hungry_ret=hungry_ret, capped=float(cx['capped']),
                                                 InvN=self.InvN or 0.0, S_G=self.S_G, subsidy=subsidy, share_pay=float(share_pay.sum()),
                                                 gov_buy=gov_buy, scrap=float(scrap.sum()), hhN=qc[0], xN_prod=x[0])),
                   **getattr(self, 'vote_rec', {}),
                   **({} if P.up_gen == 0 else dict(up_s=self.up_s, up_gX=self.up_gX, up_adv=up_adv, upgraded=upgraded, up_pay=up_pay)),
                   **({} if not P.revolt else dict(rev_G=self.rev_G, revolution=revolution, regime=float(self.regime))),
                   **({} if self.pl is None else dict(self.player_state(), pl_hungry=float(H.Yd[self.pl] < 0) if cx is None else
                                                       float(got[self.pl] < sh[self.pl] * P.q0 * (1 - 1e-6)), **getattr(self, 'vote_pl', {}))))
        self.vote_pl = {}
        self.t += 1
        return rec


def pretrain(rate, years=3000, omega=1.0, seed=0):
    """事前学習: 災害の過程だけの縮めた模型(毎年、確率 rate で災害が起きる)を years 年回し、社会の記録
    (割り引いた回数 n_eff と年数 t_eff)を返す。統合した経済は、この記録を初めの信念にして始める。"""
    rng = np.random.default_rng(seed + 1000)
    n = t = 0.0
    for _ in range(years):
        n = omega * n + (rng.random() < rate)
        t = omega * t + 1.0
    return n, t


def wquant(x, w):
    o = np.argsort(x)
    return x[o], w[o]


def gini_w(x, w):
    """重みつきのジニ係数(x は一人あたりの量、w は人数)。"""
    x, w = wquant(x, w)
    cw = np.cumsum(w); cx = np.cumsum(x * w)
    if cx[-1] <= 0:
        return 0.0
    return float(1 - np.sum(w * (cx + np.concatenate([[0.0], cx[:-1]]))) / (cw[-1] * cx[-1]))


def top_share_w(x, w, q=0.1):
    """上位 q の人数の取り分(重みつき)。"""
    x, w = wquant(x, w)
    cw = np.cumsum(w[::-1]); xs = (x * w)[::-1]
    take = np.clip((q * cw[-1] - (cw - w[::-1])) / w[::-1], 0.0, 1.0)
    return float(np.sum(xs * take) / max(np.sum(x * w), 1e-12))


def _dist(self, Yd):
    H = self.H
    pc = Yd / H.heads
    nw = H.M / H.heads
    return dict(gini_inc=gini_w(pc, H.heads), top10_inc=top_share_w(pc, H.heads), gini_dep=gini_w(nw, H.heads))


Im1.dist_stats = _dist


def simulate(T=200, shocks=(), **kw):
    e = Im1(**{k: v for k, v in kw.items()})
    for t in range(T):
        for (ts, ch) in shocks:
            if ts == t:
                e.p.set(**ch)
        e.step()
    return {k: np.array([r[k] for r in e.log]) for k in e.log[0]}, e
