/-
  「2026年の経済モデル統合」(ノート 4)の模型 im の、Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace IntegratedModel
/-! ## 第 1 章 標本の家計: 制限と持ち上げ

持ち上げ(lifting)は、標本の記入の量 m_i を、マクロの総額 M に合わせ直す。
人数 h_i に比例してずらしても(預金)、比で掛けても(土地・知財)、和は M になる。 -/

/-- 人数に比例してずらす持ち上げ: Σ (m_i + h_i (M − Σm) / Σh) = M。 -/
theorem lift_shift_sum {ι : Type*} (s : Finset ι) (m h : ι → ℝ) (M : ℝ) (hH : (∑ i ∈ s, h i) ≠ 0) :
    ∑ i ∈ s, (m i + h i * ((M - ∑ j ∈ s, m j) / ∑ j ∈ s, h j)) = M := by
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, mul_div_assoc', mul_div_cancel_left₀ _ hH]
  ring

/-- 比で掛ける持ち上げ: Σ x_i · (M / Σx) = M。 -/
theorem lift_ratio_sum {ι : Type*} (s : Finset ι) (x : ι → ℝ) (M : ℝ) (hX : (∑ i ∈ s, x i) ≠ 0) :
    ∑ i ∈ s, x i * (M / ∑ j ∈ s, x j) = M := by
  rw [← Finset.sum_mul, mul_div_cancel₀ _ hX]

/-- 人数に比例してずらしても、同じ人数の二つの記入の差(散らばり)は変わらない。 -/
theorem lift_keeps_gap (mi mj h c : ℝ) : (mi + h * c) - (mj + h * c) = mi - mj := by ring
/-! ## 第 2 章 帳簿の骨組み

一つの取引は、払う側から受け取る側へ額 a を動かす。どの部門 s についても、
取引ごとの増減(受け取りなら +a、支払いなら −a、どちらでもなければ 0)を足すと、部門の和は 0 になる。
取引をいくつ重ねても、和は 0 のまま。 -/

/-- 一つの取引の、部門 s への効き目。 -/
def entry {S : Type*} [DecidableEq S] (src dst : S) (a : ℝ) (s : S) : ℝ :=
  (if s = dst then a else 0) - (if s = src then a else 0)

theorem entry_sum_zero {S : Type*} [Fintype S] [DecidableEq S] (src dst : S) (a : ℝ) :
    ∑ s, entry src dst a s = 0 := by
  simp [entry, Finset.sum_sub_distrib]

/-- 取引の列(払う側、受け取る側、額)を重ねても、部門の和は 0。 -/
theorem ledger_sum_zero {S : Type*} [Fintype S] [DecidableEq S] (tx : List (S × S × ℝ)) :
    ∑ s, (tx.map fun x => entry x.1 x.2.1 x.2.2 s).sum = 0 := by
  induction tx with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [Finset.sum_add_distrib, ih, entry_sum_zero, add_zero]
/-! ## 第 3 章 カレツキの式の一般化(労働者も貯めるとき)

資本家は所得 Mk の割合 s を貯め、その割合 ι を投資し、残りで国債を買う。
労働者は所得 Mw の割合 s_w を貯めて国債を買う。国債の発行は政府の赤字 D に等しい。
このとき (1 − ι) s Mk + s_w Mw = D。s_w = 0 なら、資本家の所得は D / ((1 − ι) s) に決まる(ノート M の gj_capitalist_income)。
s_w > 0 なら、労働者の所得が増えるほど、資本家の所得は減る(パシネッティの問題の形)。 -/
theorem kalecki_general {Mk Mw s ι sw D : ℝ}
    (hbonds : (Mk - (1 - s) * Mk - ι * s * Mk) + sw * Mw = D) :
    (1 - ι) * s * Mk + sw * Mw = D := by
  linarith

theorem kalecki_capitalist_income {Mk Mw s ι sw D : ℝ} (h : 0 < (1 - ι) * s)
    (hbonds : (1 - ι) * s * Mk + sw * Mw = D) :
    Mk = (D - sw * Mw) / ((1 - ι) * s) := by
  rw [eq_div_iff h.ne']
  linarith
/-! ## 第 5 章 国際収支の恒等式と、利子生活者の地域

経常収支 = 貿易収支 + 第一次所得収支(外の資産の利子・配当)。経常収支が釣り合っていても、外からの利子が正なら、
貿易収支は必ず赤字になる(ノート misc_ipynb の ricardo_comparative_advantage 第 13 章と同じ命題)。 -/
theorem rentier_trade_deficit (CA TB PI : ℝ) (hid : CA = TB + PI) (hCA : CA = 0) (hPI : 0 < PI) : TB < 0 := by
  linarith

/-- 対外の純資産 NFA の利回り i が正で、経常収支が釣り合うなら、貿易の赤字の大きさは i × NFA。 -/
theorem deficit_equals_interest (TB i NFA : ℝ) (hid : 0 = TB + i * NFA) : -TB = i * NFA := by
  linarith
/-! ## 第 8 章 死蔵と準備の縛り

準備の量 R が決まっていて、預金 D のうち H が死蔵(銀行の外の現金)になると、銀行が新しく貸せる割合(信用の係数)は
cf = (R − H) / (ρ (D − H))。準備が預金より少ない(R ≤ D)限り、死蔵が増えるほど、信用の係数は下がる。 -/
theorem credit_factor_antitone {R D ρ H₁ H₂ : ℝ} (hρ : 0 < ρ) (hRD : R ≤ D) (h12 : H₁ ≤ H₂) (h2 : H₂ < D) :
    (R - H₂) / (ρ * (D - H₂)) ≤ (R - H₁) / (ρ * (D - H₁)) := by
  have d1 : 0 < ρ * (D - H₁) := mul_pos hρ (by linarith)
  have d2 : 0 < ρ * (D - H₂) := mul_pos hρ (by linarith)
  rw [div_le_div_iff₀ d2 d1]
  nlinarith [mul_nonneg (sub_nonneg.mpr h12) (sub_nonneg.mpr hRD), hρ.le]
/-! ## 第 11 章 文化から技術への波及の上限

技術の量 x が、毎年 x (1 + c n) の割合で伸び、波及の倍率 n が上限 F で頭打ちなら、伸びは指数 (1 + c F)^t で押さえられる。
上限がなく、n が x とともに増えると、この押さえがなくなる(段 8b で、どの種も発散した)。 -/
theorem capped_growth_bound (x : ℕ → ℝ) (c F : ℝ) (hc : 0 ≤ c * F)
    (step : ∀ t, 0 ≤ x t → x (t + 1) ≤ x t * (1 + c * F)) (pos : ∀ t, 0 ≤ x t) :
    ∀ t, x t ≤ x 0 * (1 + c * F) ^ t := by
  intro t
  induction t with
  | zero => simp
  | succ n ih =>
    calc x (n + 1) ≤ x n * (1 + c * F) := step n (pos n)
      _ ≤ x 0 * (1 + c * F) ^ n * (1 + c * F) := by
          apply mul_le_mul_of_nonneg_right ih; linarith
      _ = x 0 * (1 + c * F) ^ (n + 1) := by ring
/-! ## 第 16 章 群れの相関と投票率

群れの投票率 t(ρ) = 1 − exp(−A √(ρ/π) / c)(ノート M 第 4 章)。賭け金 A ≥ 0、費用 c > 0 なら、
群れの中の相関 ρ が高まるほど、投票率は上がる(呼びかけが群れを動かす道)。 -/
theorem turnout_monotone {A c ρ₁ ρ₂ : ℝ} (hA : 0 ≤ A) (hc : 0 < c) (h : ρ₁ ≤ ρ₂) :
    1 - Real.exp (-(A * Real.sqrt (ρ₁ / Real.pi) / c)) ≤ 1 - Real.exp (-(A * Real.sqrt (ρ₂ / Real.pi) / c)) := by
  have hs : Real.sqrt (ρ₁ / Real.pi) ≤ Real.sqrt (ρ₂ / Real.pi) :=
    Real.sqrt_le_sqrt (div_le_div_of_nonneg_right h Real.pi_pos.le)
  have : A * Real.sqrt (ρ₁ / Real.pi) / c ≤ A * Real.sqrt (ρ₂ / Real.pi) / c :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hs hA) hc.le
  have := Real.exp_le_exp.mpr (neg_le_neg this)
  linarith

/-- プレイヤーの切り出し: 記入 i の量 x_i を、割合 f で二つに分けても、総量は変わらない。 -/
theorem carve_keeps_total {ι : Type*} [DecidableEq ι] (s : Finset ι) (x : ι → ℝ) (i : ι) (hi : i ∈ s) (f : ℝ) :
    (∑ j ∈ s, Function.update x i (x i * (1 - f)) j) + x i * f = ∑ j ∈ s, x j := by
  rw [Finset.sum_update_of_mem hi, ← Finset.add_sum_erase s x hi, Finset.sdiff_singleton_eq_erase]
  ring
end IntegratedModel
