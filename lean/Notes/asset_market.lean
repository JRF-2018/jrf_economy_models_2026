/-
  「土地と知財と借金」(ノート 2)の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

namespace AssetMarket
/-! ## 第 3 章 借金はなぜ際限なく増えないのか

全体の正味の資産 A の毎期の増分は、ΔA = (賃金 − 必需品の代金) × 人数 − 贅沢品の値段 × 贅沢品の需要(恒等式)。
A が落ち着く(ΔA = 0)なら、贅沢品の需要は (賃金 − 必需品) × 人数 / 値段 に決まる。返済の決まりや傾斜によらない。 -/
theorem demand_pinned {w n W pL D : ℝ} (hp : pL ≠ 0) (hA : (w - n) * W - pL * D = 0) :
    D = (w - n) * W / pL := by
  field_simp; linarith

/-- 借り入れ E(L) が全 liability L について右下がり、返済 R(L) が右上がり(真に)なら、
E(L) = R(L) となる L(落ち着き先)は高々一つ。 -/
theorem unique_crossing (E R : ℝ → ℝ) (hE : Antitone E) (hR : StrictMono R) {L₁ L₂ : ℝ}
    (h1 : E L₁ = R L₁) (h2 : E L₂ = R L₂) : L₁ = L₂ := by
  by_contra hne
  rcases lt_or_gt_of_ne hne with h | h
  · have := hR h; have := hE h.le; linarith
  · have := hR h; have := hE h.le; linarith
/-! ## 第 5 章 土地と知財: 減耗する資産の定常の量

毎期 I だけ新しく生まれ、割合 δ で減耗する資産(知財)の量 X_{t+1} = (1 − δ) X_t + I の定常の量は I / δ。 -/
theorem stock_steady {I δ : ℝ} (hδ : δ ≠ 0) : (1 - δ) * (I / δ) + I = I / δ := by
  field_simp; ring

/-- 定常の量 I / δ は、減耗が速いほど小さい(I ≥ 0)。減耗率のわずかな違いで、知財の総額が大きく変わる。 -/
theorem stock_steady_antitone {I δ₁ δ₂ : ℝ} (hI : 0 ≤ I) (h1 : 0 < δ₁) (h12 : δ₁ ≤ δ₂) : I / δ₂ ≤ I / δ₁ :=
  div_le_div_of_nonneg_left hI h1 h12

/-- 資産の値打ち = 利回りの元 ÷ (金利 + 減耗)。土地(δ = 0)は、減耗する知財(δ > 0)より値打ちが高い。 -/
theorem capitalized_value_antitone {y r δ : ℝ} (hy : 0 ≤ y) (hr : 0 < r) (hδ : 0 ≤ δ) : y / (r + δ) ≤ y / r :=
  div_le_div_of_nonneg_left hy hr (by linarith)
end AssetMarket
