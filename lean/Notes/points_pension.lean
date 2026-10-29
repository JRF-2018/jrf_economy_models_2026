/-
  「ポイントと年金と信用」(ノート P)の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace PointsPension
/-! ## 第 6 章 ID を分けても、取り分は変わらない

基金 X を、貢献 c に比例して配る(全体の貢献 T)。一つの ID の貢献 c を二つの ID に c₁ + c₂ と分けても、
二つの ID の取り分の和は、分ける前の取り分と同じ(比例の配り方は、ID の分割で得をしない)。 -/
theorem split_invariant (X T c₁ c₂ : ℝ) : X * (c₁ / T) + X * (c₂ / T) = X * ((c₁ + c₂) / T) := by ring
/-! ## 第 7 章 BI が否認されたときの返し方

作者の漸化式 p_{n+1} = (1 + R) p_n の解は、比 (1 + R) の等比数列 p_n = p_0 (1 + R)^n(閉じた式の比は R ではない)。 -/
theorem pointback_geometric (p₀ R : ℝ) (n : ℕ) : p₀ * (1 + R) ^ (n + 1) = (1 + R) * (p₀ * (1 + R) ^ n) := by ring

/-- 等比数列の和: Σ_{k<n} (1 + R)^k = ((1 + R)^n − 1) / R(R ≠ 0)。 -/
theorem pointback_sum (R : ℝ) (hR : R ≠ 0) (n : ℕ) : ∑ k ∈ range n, (1 + R) ^ k = ((1 + R) ^ n - 1) / R := by
  rw [geom_sum_eq (by intro h; apply hR; linarith)]
  congr 1; ring

/-- 負担の合計 R (X₂ − x) は、最後に残る分 x が 0 以上で R ≥ 0 なら、利率一期分 R X₂ を超えない。 -/
theorem burden_bound {R X₂ x : ℝ} (hR : 0 ≤ R) (hx : 0 ≤ x) : R * (X₂ - x) ≤ R * X₂ := by
  nlinarith
/-! ## 第 8 章 自然な借入制約と、ポンジ・ゲーム

複利で倍になるまでの期数 log 2 / log(1 + r) は、単利で倍になるまでの期数 1 / r 以下(0 < r ≤ 1)。
log の凹さ(1 と 2 を結ぶ弦の上にある)から、log(1 + r) ≥ r log 2。 -/
theorem log_one_add_ge {r : ℝ} (h0 : 0 ≤ r) (h1 : r ≤ 1) : r * Real.log 2 ≤ Real.log (1 + r) := by
  have hc := strictConcaveOn_log_Ioi.concaveOn
  have := hc.2 (show (1:ℝ) ∈ Set.Ioi 0 by norm_num) (show (2:ℝ) ∈ Set.Ioi 0 by norm_num)
    (show 0 ≤ 1 - r by linarith) h0 (by ring)
  simp only [smul_eq_mul, Real.log_one, mul_zero, zero_add] at this
  have e : (1 - r) * 1 + r * 2 = 1 + r := by ring
  rw [e] at this; linarith

theorem compound_le_simple {r : ℝ} (h0 : 0 < r) (h1 : r ≤ 1) : Real.log 2 / Real.log (1 + r) ≤ 1 / r := by
  have hl : 0 < Real.log (1 + r) := Real.log_pos (by linarith)
  rw [div_le_div_iff₀ hl h0]
  have := log_one_add_ge h0.le h1
  linarith

/-- 毎期 1 を借りて返さない借り手の借金 D_{t+1} = (1 + r) D_t + 1。r < 0 なら、借金は −1/r に落ち着く(固定点)。 -/
theorem ponzi_fixed_point {r : ℝ} (hr : r ≠ 0) : (1 + r) * (-1 / r) + 1 = -1 / r := by
  field_simp; ring

/-- r = 0 なら、借金は毎期 1 ずつ増え、際限なく増える(貸倒れの判定は働かない)。 -/
theorem ponzi_zero_rate (D : ℕ → ℝ) (h0 : D 0 = 0) (h : ∀ t, D (t + 1) = (1 + 0) * D t + 1) : ∀ t, D t = t := by
  intro t; induction t with
  | zero => simp [h0]
  | succ n ih => rw [h, ih]; push_cast; ring
end PointsPension
