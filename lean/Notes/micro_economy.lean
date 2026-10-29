/-
  「価格を最適化で決める経済」(ノート 1)の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace MicroEconomy
/-! ## 第 1 章 名目の水準は、価格の条件の中では決まらない

価格の条件「価格 = (1 + 値上げ幅) × (賃金 × 労働の係数 + 原料の価格 × 原料の係数)」は、
賃金とすべての価格を同じ k 倍しても成り立つ(一次の同次)。だから、条件だけでは名目の水準は決まらない。 -/
theorem price_condition_homogeneous {p m W γ q α k : ℝ} (h : p = (1 + m) * (W * γ + q * α)) :
    k * p = (1 + m) * ((k * W) * γ + (k * q) * α) := by
  rw [h]; ring
/-! ## 第 7 章 会計の三つの誤り

(1) 資本家の利益は、企業ごとの利益を持ち主ごとに「足す」。持ち主ごとに足した和は、企業の利益の総和に等しい
(上書きすると、二社目以降の利益が消える)。 -/
theorem owner_profit_sum {C O : Type*} [Fintype C] [Fintype O] [DecidableEq O] (owner : C → O) (π : C → ℝ) :
    ∑ o, ∑ c ∈ univ.filter (fun c => owner c = o), π c = ∑ c, π c :=
  Finset.sum_fiberwise univ owner π

/-- (3) 残業の二重計上: 企業は w·D を払うが、労働者は w·S + 2w(D − S) を受け取る(社会的負債が残業の割増を払う)。
余分は w(D − S)。 -/
theorem overtime_double_count (w S D : ℝ) : (w * S + 2 * w * (D - S)) - (w * D) = w * (D - S) := by ring
/-! ## 第 8 章 均衡の残差と、スラッファの取り換え

均衡の条件の残差の二乗和が 0 になるのは、すべての条件が満たされるとき、そのときに限る。
だから、二乗和の最小化(最小値 0)は、均衡を解くことと同じになる。 -/
theorem residual_sumsq_zero_iff {ι : Type*} (s : Finset ι) (r : ι → ℝ) :
    ∑ i ∈ s, r i ^ 2 = 0 ↔ ∀ i ∈ s, r i = 0 := by
  rw [Finset.sum_eq_zero_iff_of_nonneg (fun i _ => sq_nonneg (r i))]
  simp

/-- 一財の生産価格 p = (1 + r)(w l + p a) から、実質賃金 w/p = (1 − (1 + r) a) / ((1 + r) l)。 -/
theorem sraffa_real_wage {p w l a r : ℝ} (hp : p ≠ 0) (hl : l ≠ 0) (hr : 1 + r ≠ 0)
    (h : p = (1 + r) * (w * l + p * a)) : w / p = (1 - (1 + r) * a) / ((1 + r) * l) := by
  rw [div_eq_div_iff hp (mul_ne_zero hr hl)]
  linear_combination (-1 : ℝ) * h

/-- 賃金と利潤の取り換え: 標準の利潤率 r が高いほど、実質賃金は低い(l > 0)。 -/
theorem sraffa_tradeoff {l a r₁ r₂ : ℝ} (hl : 0 < l) (h0 : 0 < 1 + r₁) (h12 : r₁ ≤ r₂) :
    (1 - (1 + r₂) * a) / ((1 + r₂) * l) ≤ (1 - (1 + r₁) * a) / ((1 + r₁) * l) := by
  have h2 : 0 < 1 + r₂ := by linarith
  rw [div_le_div_iff₀ (mul_pos h2 hl) (mul_pos h0 hl)]
  nlinarith [mul_pos hl hl, mul_pos h0 h2]
end MicroEconomy
