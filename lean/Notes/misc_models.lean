/-
  「雑多なモデルの再検討 ── 必需品と贅沢品、縁故主義、投票、AI と弾圧」の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace MiscModels
/-! ## 第 2 章 必需品と贅沢品: 帳簿を閉じる

賃金を 1 とする。必需品の売上 EN は、賃金 LN と運転資金の利子 ic·LN とレント rent に分かれる。
贅沢品の売上 EL も同じく LL、ic·LL、利潤 profit に分かれる。働く人の数 L は LN + LL + I + G。
労働者の所得は L + T(T は再分配)、資本家の所得は rent + profit + 利子 − T。
労働者は所得をすべて使い、消費の合計は EN + EL = 労働者の所得 + 資本家の消費 Ck。
このとき、資本家の貯蓄から投資を引いた残りは、ちょうど政府の赤字 G + iB·B になる(部門の収支の和は 0)。 -/

theorem gj_books {EN EL LN LL I G ic iB B rent profit T L Mw Mk Ck : ℝ}
    (hEN : EN = (1 + ic) * LN + rent) (hEL : EL = (1 + ic) * LL + profit)
    (hL : L = LN + LL + I + G) (hMw : Mw = L + T)
    (hMk : Mk = rent + profit + ic * (LN + LL) + iB * B - T)
    (hC : EN + EL = Mw + Ck) :
    Mk - Ck - I = G + iB * B := by
  subst hEN hEL hMw hMk hL
  linarith

/-- 資本家が所得の割合 s を貯め、その割合 ι を投資に回すなら、資本家の(税引き後の)所得は
赤字 ÷ 漏れの率 (1 − ι) s に決まる(カレツキの利潤の式の形)。再分配の率 τ はこの式に出てこない。 -/
theorem gj_capitalist_income {Mk s ι D : ℝ} (h : 0 < (1 - ι) * s)
    (hbal : Mk - (1 - s) * Mk - ι * s * Mk = D) :
    Mk = D / ((1 - ι) * s) := by
  rw [eq_div_iff h.ne']
  linarith

/-- ゴーマンの集計: 二つの家計の、必要量を超えた所得のうち必需品に回す割合 b が同じなら、
二人の間で t だけ所得を移しても、必需品への支出の合計は変わらない。 -/
theorem gj_gorman (p g1 g2 b M1 M2 t : ℝ) :
    (p * g1 + b * (M1 - t - p * g1)) + (p * g2 + b * (M2 + t - p * g2))
      = (p * g1 + b * (M1 - p * g1)) + (p * g2 + b * (M2 - p * g2)) := by
  ring

/-- 割合が違えば(払う側 b1、受け取る側 b2)、必需品への支出の合計は (b2 − b1) t だけ変わる。
再分配で必需品の需要が増えるのは、受け取る側の割合が大きいときだけである。 -/
theorem gj_engel_shift (p g1 g2 b1 b2 M1 M2 t : ℝ) :
    ((p * g1 + b1 * (M1 - t - p * g1)) + (p * g2 + b2 * (M2 + t - p * g2)))
      - ((p * g1 + b1 * (M1 - p * g1)) + (p * g2 + b2 * (M2 - p * g2))) = (b2 - b1) * t := by
  ring
/-! ## 第 3 章 縁故主義: 期待の戻り、尾根、情報の値打ち

元のノートの事業は、リスク r のとき確率 (1 − r)^m で成功し、成功すると 1 + 1.5 r 倍が戻る。
m = 1 のとき、期待の戻りから 1 を引くと r (1/2 − 3r/2) になり、r < 1/3 で正、r > 1/3 で負。 -/

theorem nepo_gain_m1 (r : ℝ) : (1 - r) * (1 + 3 / 2 * r) - 1 = r * (1 / 2 - 3 / 2 * r) := by
  ring

theorem nepo_gain_pos_iff {r : ℝ} (hr : 0 < r) :
    0 < (1 - r) * (1 + 3 / 2 * r) - 1 ↔ r < 1 / 3 := by
  rw [nepo_gain_m1]
  constructor
  · intro h
    by_contra hc
    rw [not_lt] at hc
    nlinarith
  · intro h
    apply mul_pos hr
    linarith

/-- 「リスク q 以下の事業に等しく投じる」ときの、一期の得(m = 1)は
∫₀^q r (1/2 − 3r/2) dr = q²/4 − q³/2。これは q = 1/3 で最大値 1/108 をとる単峰の山で、
同じ得になる q が、山の左右に二つずつある(元のノートその 2 の「二つの枝」)。ここでは山の頂の不等式を確かめる。 -/
theorem nepo_ridge {q : ℝ} (hq : 0 ≤ q) : q ^ 2 / 4 - q ^ 3 / 2 ≤ 1 / 108 := by
  nlinarith [sq_nonneg (q - 1 / 3), mul_nonneg hq (sq_nonneg (q - 1 / 3))]

/-- 情報の値打ち: 事業 i の得 e i − 1 を知っている貸し手は、得の正の事業だけを選べる。
どんな選び方 S の得も、それを超えない。だから、外部性がなければ、正しく知る貸し手(縁故)が、
網の目の中で損をすることはない。 -/
theorem info_value {ι : Type*} [DecidableEq ι] (U S : Finset ι) (hS : S ⊆ U) (e : ι → ℝ) :
    ∑ i ∈ S, (e i - 1) ≤ ∑ i ∈ U.filter (fun i => 1 < e i), (e i - 1) := by
  calc ∑ i ∈ S, (e i - 1) ≤ ∑ i ∈ S, max (e i - 1) 0 :=
        Finset.sum_le_sum fun i _ => le_max_left _ _
    _ ≤ ∑ i ∈ U, max (e i - 1) 0 :=
        Finset.sum_le_sum_of_subset_of_nonneg hS fun i _ _ => le_max_right _ _
    _ = ∑ i ∈ U.filter (fun i => 1 < e i), (e i - 1) := by
        rw [Finset.sum_filter]
        apply Finset.sum_congr rfl
        intro i _
        split_ifs with h
        · exact max_eq_left (by linarith)
        · exact max_eq_right (by linarith)
/-! ## 第 4 章 投票の効用と相関

作者の式: 投票するときの期待値から、棄権するときの期待値と費用を引くと、
(A1 − A2)(p1 − p2) − C になる。 -/

theorem vote_utility (A1 A2 p1 p2 C : ℝ) :
    (A1 * p1 + A2 * (1 - p1)) - (A1 * p2 + A2 * (1 - p2)) - C
      = (A1 - A2) * (p1 - p2) - C := by
  ring

/-- ベータ分布 Beta(a, b) の平均は a / (a + b)。一人が投票したと知ると Beta(a + 1, b)、
棄権したと知ると Beta(a, b + 1) になる。二つの平均の差は 1 / (a + b + 1)、
つまり級内相関 ρ そのものである。自分の選択は、仲間の投票の確率を ρ だけ動かす「しるし」になる。 -/
theorem beta_update_shift {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    (a + 1) / (a + 1 + b) - a / (a + (b + 1)) = 1 / (a + b + 1) := by
  have h : a + b + 1 ≠ 0 := by linarith
  rw [show a + 1 + b = a + b + 1 by ring, show a + (b + 1) = a + b + 1 by ring]
  field_simp
  ring

/-- 級内相関 ρ の n 人の票の和の分散。一人の分散を v、二人の間の共分散を ρ v とすると、
和の分散は n v + n (n − 1) ρ v。ρ > 0 なら、分散は n² の速さで増え、票の数の揺れは人数に比例する。
独立(ρ = 0)なら n v で、揺れは √n に比例する。 -/
theorem equicorrelated_variance (n : ℕ) (v ρ : ℝ) :
    ∑ i ∈ range n, ∑ j ∈ range n, (if i = j then v else ρ * v)
      = n * v + n * (n - 1) * ρ * v := by
  have hrow : ∀ i ∈ range n,
      ∑ j ∈ range n, (if i = j then v else ρ * v) = v + (n - 1) * ρ * v := by
    intro i hi
    rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, Finset.filter_eq,
      Finset.filter_ne]
    simp only [hi, ite_true, Finset.card_singleton, nsmul_eq_mul, Nat.cast_one, one_mul]
    rw [Finset.card_erase_of_mem hi, Finset.card_range]
    rcases Nat.eq_zero_or_pos n with h0 | hpos
    · subst h0; simp at hi
    · rw [Nat.cast_sub (by omega)]; push_cast; ring
  rw [Finset.sum_congr rfl hrow, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  ring
/-! ## 第 5 章 注意の予算と、国家ロボットの規模

国家ロボットの規模を R、捕獲を防ぐ注意労働を A、捕獲の確率を Π(A) = e^{−A} とする。
捕獲を防ぐ注意 A と、統制されていない残差 R·Π(A) の監査を、同じ人間の注意から払うなら、
要る注意は A + R e^{−A} で、その最小値は 1 + log R(A = log R のとき)。
注意の量が N なら、持てる規模は R ≤ e^{N − 1} どまりになる。上限はあるが、注意の量に対して指数的に大きい。 -/

theorem attention_lower_bound {R : ℝ} (hR : 0 < R) (A : ℝ) :
    1 + Real.log R ≤ A + R * Real.exp (-A) := by
  have h1 : R * Real.exp (-A) = Real.exp (Real.log R - A) := by
    rw [Real.exp_sub, Real.exp_log hR, div_eq_mul_inv, Real.exp_neg]
  have h2 := Real.add_one_le_exp (Real.log R - A)
  rw [h1]
  linarith

theorem attention_at_optimum {R : ℝ} (hR : 0 < R) :
    Real.log R + R * Real.exp (-Real.log R) = 1 + Real.log R := by
  rw [Real.exp_neg, Real.exp_log hR, mul_inv_cancel₀ hR.ne']
  ring
end MiscModels
