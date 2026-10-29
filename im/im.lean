/-
  「2026年の経済モデル統合」の模型 im の Lean による定式化(段 0)
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace IntegratedModel

/-! ## 段 0 帳簿の骨組み

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

/-- 一般のカレツキの式: 資本家は所得 Mk の割合 s を貯め、その割合 ι を投資し、残りで国債を買う。
労働者は所得 Mw の割合 s_w を貯めて国債を買う。国債の発行は政府の赤字 D に等しい。
このとき (1 − ι) s Mk + s_w Mw = D。s_w = 0 なら、資本家の所得は D / ((1 − ι) s) に決まる(ノート M の gj_capitalist_income)。
s_w > 0 なら、労働者の所得が増えるほど、資本家の所得は減る。 -/
theorem kalecki_general {Mk Mw s ι sw D : ℝ}
    (hbonds : (Mk - (1 - s) * Mk - ι * s * Mk) + sw * Mw = D) :
    (1 - ι) * s * Mk + sw * Mw = D := by
  linarith

theorem kalecki_capitalist_income {Mk Mw s ι sw D : ℝ} (h : 0 < (1 - ι) * s)
    (hbonds : (1 - ι) * s * Mk + sw * Mw = D) :
    Mk = (D - sw * Mw) / ((1 - ι) * s) := by
  rw [eq_div_iff h.ne']
  linarith

end IntegratedModel
