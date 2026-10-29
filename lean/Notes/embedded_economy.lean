/-
  「経済の中で経済が育つ」(ノート O)の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace EmbeddedEconomy
/-! ## 第 3 章 ペニーオークションの混合均衡

今の値段 p で、入札の費用 c を払って入札すると、ほかの誰も続けなければ v − p を得る。
ほかの人が続けない確率 μ が c / (v − p) なら、入札の得の期待値は 0 で、入札するかしないかが無差別になる。 -/
theorem penny_indifference {v p c : ℝ} (h : p < v) : c / (v - p) * (v - p) - c = 0 := by
  have : v - p ≠ 0 := by linarith
  field_simp; ring

/-- 無差別にする確率が 1 以下であるのは、費用が残りの値打ち以下のとき。 -/
theorem penny_prob_le_one {v p c : ℝ} (h : p < v) : c / (v - p) ≤ 1 ↔ c ≤ v - p := by
  rw [div_le_one (by linarith)]
/-! ## 第 5 章 経済の圏: 集計は保存を守る

一つの取引は、払う側に −a、受け取る側に +a を書く。払う側と受け取る側が同じ組にいれば、
その組の中の和は 0 になる(まとめた点の中の受け払いは見えなくなる。経済が「バーチャル」になる)。 -/
def entry {S : Type*} [DecidableEq S] (src dst : S) (a : ℝ) (s : S) : ℝ :=
  (if s = dst then a else 0) - (if s = src then a else 0)

theorem internal_trade_vanishes {S : Type*} [DecidableEq S] (g : Finset S) (src dst : S) (a : ℝ)
    (hs : src ∈ g) (hd : dst ∈ g) : ∑ s ∈ g, entry src dst a s = 0 := by
  simp [entry, Finset.sum_sub_distrib, Finset.sum_ite_eq', hs, hd]

/-- 段階的にまとめても、一度にまとめても、総和は同じ(集計は和を保つ)。 -/
theorem aggregate_in_stages {S G : Type*} [Fintype S] [Fintype G] [DecidableEq G] (grp : S → G) (x : S → ℝ) :
    ∑ g, ∑ s ∈ univ.filter (fun s => grp s = g), x s = ∑ s, x s :=
  Finset.sum_fiberwise univ grp x

/-- 一物一価: 組の中の値段 p_i で量 q_i を評価した和が、どの量でも一つの値段 p̄ × 量の和に等しいのは、
組の中の値段がすべて p̄ に等しいとき、そのときに限る(まとめても評価が自然である条件)。 -/
theorem one_price_iff {ι : Type*} [DecidableEq ι] (s : Finset ι) (p : ι → ℝ) (pbar : ℝ) :
    (∀ q : ι → ℝ, ∑ i ∈ s, p i * q i = pbar * ∑ i ∈ s, q i) ↔ ∀ i ∈ s, p i = pbar := by
  constructor
  · intro h i hi
    have := h (fun j => if j = i then 1 else 0)
    simpa [Finset.sum_ite_eq', hi] using this
  · intro h q
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun i hi => by rw [h i hi])

/-- 構成が一定(量が q_i = w_i Q)なら、値段が違っても、指数 Σ p_i w_i で評価はずれない(定理 A の最小の姿)。 -/
theorem constant_composition {ι : Type*} (s : Finset ι) (p w : ι → ℝ) (Q : ℝ) :
    ∑ i ∈ s, p i * (w i * Q) = (∑ i ∈ s, p i * w i) * Q := by
  rw [Finset.sum_mul]; exact Finset.sum_congr rfl (fun i _ => by ring)
/-! ## 第 8 章 ハイブリッド系としての経済: 区間ごとの縮約

縮約の定理 A(credit_simulation_v6 の CreditV6/ で証明)の結論、すなわち「一つのモードの中では、まとめる写像 agg が、
世界の 1 期の写像 F と一国の 1 期の写像 f と可換(agg ∘ F = f ∘ agg)」を仮定として置き、そこから先を証明する。 -/

section Hybrid
variable {X Y M : Type*}

/-- 区間ごとの縮約: 一つのモードの中で 1 期の写像が可換なら、何期続けても可換。 -/
theorem contraction_iterate (agg : X → Y) (F : X → X) (f : Y → Y) (h : ∀ x, agg (F x) = f (agg x)) :
    ∀ n x, agg (F^[n] x) = f^[n] (agg x) := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih => intro x; rw [Function.iterate_succ_apply', Function.iterate_succ_apply', h, ih]

/-- ハイブリッド系の一本の道: 予定表(モード、そのモードが続く期数)の列に従って流れ、区間の終わりごとにリセットする。 -/
def run (F : M → X → X) (R : X → X) : List (M × ℕ) → X → X
  | [], x => x
  | (m, n) :: rest, x => run F R rest (R ((F m)^[n] x))

/-- どのモードの流れも、リセットも、まとめる写像と可換なら、道全体が可換(縮約はずっと正しい)。
リセットが可換でないとき(一点へのまとめ)は、切り替わりのたびに本当の状態からまとめ直す必要がある。 -/
theorem contraction_hybrid (agg : X → Y) (F : M → X → X) (f : M → Y → Y) (R : X → X) (r : Y → Y)
    (hF : ∀ m x, agg (F m x) = f m (agg x)) (hR : ∀ x, agg (R x) = r (agg x)) :
    ∀ (sched : List (M × ℕ)) (x : X), agg (run F R sched x) = run f r sched (agg x) := by
  intro sched
  induction sched with
  | nil => intro x; rfl
  | cons hd rest ih =>
    intro x
    obtain ⟨m, n⟩ := hd
    simp only [run]
    rw [ih, hR, contraction_iterate agg (F m) (f m) (hF m)]

end Hybrid

/-- 評価の跳び(8.2 節): 一点にまとめた値打ちの倍率 (1 − s)/(1 + λ) + s(1 + λ) は、新しい経済の割合 s によって違う。
s = 0 と s = 1 で倍率が等しいのは、λ = 0(跳びがない)のときに限る(Lean では λ を lam と書く)。 -/
theorem jump_depends_on_share {lam : ℝ} (h : 0 < 1 + lam) :
    (1 - 0) / (1 + lam) + 0 * (1 + lam) = (1 - 1) / (1 + lam) + 1 * (1 + lam) ↔ lam = 0 := by
  constructor
  · intro e
    have hne : (1 + lam) ≠ 0 := h.ne'
    field_simp at e
    nlinarith [e]
  · intro e; subst e; norm_num

/-- 一点へのまとめは、リセットと可換にならない: 旧い経済の値打ちを a 倍、新しい経済を b 倍にするリセットで、
a ≠ b なら、まとめた値打ち o + n だけからリセットのあとの値打ち a o + b n を決める写像 g はない。 -/
theorem one_point_reset_not_commuting {a b : ℝ} (hab : a ≠ b) : ¬ ∃ g : ℝ → ℝ, ∀ o n : ℝ, a * o + b * n = g (o + n) := by
  rintro ⟨g, hg⟩
  have h1 := hg 1 0
  have h2 := hg 0 1
  simp at h1 h2
  exact hab (h1.trans h2.symm)

/-- 根の候補ごとに一点(旧い塊と新しい塊の二点)にまとめれば、リセットは可換: 塊の中の物差しが一つ(一物一価)だから。 -/
theorem two_point_reset_commutes {ι : Type*} (O N : Finset ι) (x : ι → ℝ) (a b : ℝ) :
    (∑ i ∈ O, a * x i, ∑ i ∈ N, b * x i) = (a * ∑ i ∈ O, x i, b * ∑ i ∈ N, x i) := by
  rw [Finset.mul_sum, Finset.mul_sum]
end EmbeddedEconomy
