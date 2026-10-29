/-
  「誰が一家離散するのか ── シミュレーション仏教の経済と、裾の重い運」の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Finset

namespace SimbdEconomy
/-! ## 第 2 章 裾の重い損失と、規模に対する収穫逓増

simbdp3 のレヴィの部分 θ / U²(U は 0〜1 の一様乱数)を、元手 K で切り捨てる。
U < a = √(θ/K) のとき損失は K、U ≥ a のとき θ / U²。その期待値は
∫₀ᵃ K du + ∫ₐ¹ θ / u² du = K a + θ (1/a − 1) = 2√(θK) − θ。
ここでは、積分を計算したあとの代数の部分を確かめる。 -/

theorem truncated_loss_mean {θ K : ℝ} (hθ : 0 < θ) (hK : 0 < K) :
    K * Real.sqrt (θ / K) + θ * (1 / Real.sqrt (θ / K) - 1)
      = 2 * Real.sqrt (θ * K) - θ := by
  have hθK : 0 < θ / K := div_pos hθ hK
  have hs : 0 < Real.sqrt (θ / K) := Real.sqrt_pos.mpr hθK
  have hsq : Real.sqrt (θ / K) ^ 2 = θ / K := Real.sq_sqrt hθK.le
  have hm : Real.sqrt (θ * K) = K * Real.sqrt (θ / K) := by
    rw [show θ * K = K ^ 2 * (θ / K) by field_simp]
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hK.le]
  have hK2 : K * Real.sqrt (θ / K) ^ 2 = θ := by rw [hsq]; field_simp
  rw [hm]
  field_simp
  linear_combination (-1 : ℝ) * hK2

/-- 元手あたりの損失 (2√(θK) − θ)/K は、x = θ/K とおくと 2√x − x になる。 -/
theorem loss_per_capital {θ K : ℝ} (hθ : 0 < θ) (hK : 0 < K) :
    (2 * Real.sqrt (θ * K) - θ) / K = 2 * Real.sqrt (θ / K) - θ / K := by
  have hm : Real.sqrt (θ * K) = K * Real.sqrt (θ / K) := by
    rw [show θ * K = K ^ 2 * (θ / K) by field_simp]
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hK.le]
  rw [hm]
  field_simp

/-- 2√x − x は 0 ≤ x ≤ 1 で増える。だから、元手 K が大きい(x = θ/K が小さい)ほど、元手あたりの損失は小さい。 -/
theorem two_sqrt_sub_mono {x y : ℝ} (hx : 0 ≤ x) (hxy : x ≤ y) (hy : y ≤ 1) :
    2 * Real.sqrt x - x ≤ 2 * Real.sqrt y - y := by
  have sx := Real.sq_sqrt hx
  have sy := Real.sq_sqrt (hx.trans hxy)
  have h1 : Real.sqrt x ≤ Real.sqrt y := Real.sqrt_le_sqrt hxy
  have h2 : Real.sqrt y ≤ 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_le_sqrt hy
  have h0 : 0 ≤ Real.sqrt x := Real.sqrt_nonneg x
  -- (2√y − y) − (2√x − x) = (√y − √x)(2 − √x − √y) ≥ 0
  nlinarith [mul_nonneg (sub_nonneg.mpr h1) (by linarith : (0 : ℝ) ≤ 2 - Real.sqrt x - Real.sqrt y)]

/-- 規模に対する収穫逓増: θ ≤ K₁ ≤ K₂ なら、元手 K₂ の元手あたりの損失は、元手 K₁ のそれ以下。 -/
theorem increasing_returns_to_scale {θ K₁ K₂ : ℝ} (hθ : 0 < θ) (h1 : θ ≤ K₁) (h12 : K₁ ≤ K₂) :
    (2 * Real.sqrt (θ * K₂) - θ) / K₂ ≤ (2 * Real.sqrt (θ * K₁) - θ) / K₁ := by
  have hK₁ : 0 < K₁ := hθ.trans_le h1
  have hK₂ : 0 < K₂ := hK₁.trans_le h12
  rw [loss_per_capital hθ hK₁, loss_per_capital hθ hK₂]
  apply two_sqrt_sub_mono (div_nonneg hθ.le hK₂.le)
  · exact div_le_div_of_nonneg_left hθ.le hK₁ h12
  · rw [div_le_one hK₁]; exact h1
/-! ## 第 3 章 下に壁があると、分散の大きい賭けが得になる

結果 x を下限 c で切り捨てる(生活保護・ロスカット)。平均が同じなら、ばらついた賭けの、切り捨てた後の期待値は、
確実に平均を得るときの値以上になる(イェンセンの不等式。max(·, c) は凸)。 -/

theorem floor_convexity {κ : Type*} (S : Finset κ) (p x : κ → ℝ) (hp : ∀ k ∈ S, 0 ≤ p k)
    (h1 : ∑ k ∈ S, p k = 1) (c : ℝ) :
    max (∑ k ∈ S, p k * x k) c ≤ ∑ k ∈ S, p k * max (x k) c := by
  apply max_le
  · exact sum_le_sum fun k hk => mul_le_mul_of_nonneg_left (le_max_left _ _) (hp k hk)
  · calc c = ∑ k ∈ S, p k * c := by rw [← sum_mul, h1, one_mul]
      _ ≤ ∑ k ∈ S, p k * max (x k) c :=
        sum_le_sum fun k hk => mul_le_mul_of_nonneg_left (le_max_right _ _) (hp k hk)
/-! ## 第 3 章 有限責任としての切り捨て

資産の種類 i ごとに、損益 Xᵢ を下限 cᵢ で切り捨てる(その種類の元手より多くは失わない)。
種類ごとに切り捨てた和は、まとめて切り捨てた値以上になる。
だから、損失の上限を事業ごとに分けることには、それだけで値打ちがある(分散投資の一つの根拠)。 -/

theorem limited_liability {ι : Type*} (s : Finset ι) (X c : ι → ℝ) :
    max (∑ i ∈ s, X i) (∑ i ∈ s, c i) ≤ ∑ i ∈ s, max (X i) (c i) :=
  max_le (sum_le_sum fun _ _ => le_max_left _ _) (sum_le_sum fun _ _ => le_max_right _ _)

/-- 期待値でも同じ: 結果 k が確率 p k で起きるとき、種類ごとの切り捨ての期待値は、まとめた切り捨ての期待値以上。 -/
theorem limited_liability_expect {κ ι : Type*} (S : Finset κ) (p : κ → ℝ) (hp : ∀ k ∈ S, 0 ≤ p k)
    (s : Finset ι) (X : κ → ι → ℝ) (c : ι → ℝ) :
    ∑ k ∈ S, p k * max (∑ i ∈ s, X k i) (∑ i ∈ s, c i)
      ≤ ∑ k ∈ S, p k * ∑ i ∈ s, max (X k i) (c i) :=
  sum_le_sum fun k hk => mul_le_mul_of_nonneg_left (limited_liability s (X k) c) (hp k hk)
/-! ## 第 5 章 上限と死蔵: 閾値つきの区分線形な関数

運用の上限 B を超えた富は死蔵に回る。家族 i の富を wᵢ ≥ 0 とすると、死蔵の総額は Σ max(wᵢ − B, 0)。

* 富の多い m 家族が、総額 W のうち割合 s を等しく持つなら、その死蔵は m · max(sW/m − B, 0) = max(sW − mB, 0)。
  W で割れば max(s − mB/W, 0)。つまり、死蔵の割合 = μ · max(0, Φ − θ) で、Φ = s(上位の取り分)、θ = mB/W、μ = 1。
* どの m 家族を選んでも、死蔵の総額は、その m 家族の富の和 − mB 以上(下からの評価)。 -/

theorem hoard_equal_split {m : ℝ} (hm : 0 < m) (s W B : ℝ) :
    m * max (s * W / m - B) 0 = max (s * W - m * B) 0 := by
  rw [mul_max_of_nonneg _ _ hm.le, mul_zero, mul_sub, mul_div_cancel₀ _ hm.ne']

theorem hoard_ratio {m W : ℝ} (hm : 0 < m) (hW : 0 < W) (s B : ℝ) :
    m * max (s * W / m - B) 0 / W = max (s - m * B / W) 0 := by
  rw [hoard_equal_split hm, ← max_div_div_right hW.le, zero_div, sub_div, mul_div_cancel_right₀ _ hW.ne']

/-- v6 の言葉への書き換え: 上位の m 家族の富が、利子の所得 rC の元本 C で、Φ = rC/Y(利払いの負担率)なら
C = ΦY/r。死蔵の割合 max(C/W − mB/W, 0) は、(Y/(rW)) · max(Φ − rmB/Y, 0) になる。
つまり ρ_eff = ρ + μ·max(0, Φ − θ) の形で、μ = Y/(rW)、θ = rmB/Y(上限まで運用した富が受け取る利子の、所得に対する比)。 -/
theorem hoard_as_rho_eff {r W Y : ℝ} (hr : 0 < r) (hW : 0 < W) (hY : 0 < Y) (Φ m B : ℝ) :
    max ((Φ * Y / r) / W - m * B / W) 0 = Y / (r * W) * max (Φ - r * m * B / Y) 0 := by
  have hc : 0 ≤ Y / (r * W) := by positivity
  rw [mul_max_of_nonneg _ _ hc, mul_zero]
  congr 1
  field_simp

theorem hoard_lower_bound {ι : Type*} (s T : Finset ι) (hT : T ⊆ s) (w : ι → ℝ) (B : ℝ) :
    ∑ i ∈ T, w i - (T.card : ℝ) * B ≤ ∑ i ∈ s, max (w i - B) 0 := by
  have h1 : ∑ i ∈ T, w i - (T.card : ℝ) * B = ∑ i ∈ T, (w i - B) := by
    rw [sum_sub_distrib, sum_const, nsmul_eq_mul]
  rw [h1]
  calc ∑ i ∈ T, (w i - B) ≤ ∑ i ∈ T, max (w i - B) 0 := sum_le_sum fun i _ => le_max_left _ _
    _ ≤ ∑ i ∈ s, max (w i - B) 0 :=
      sum_le_sum_of_subset_of_nonneg hT fun i _ _ => le_max_right _ _

/-- 死蔵の総額は、富の移し替えで集中が進むと増える(凸性): 富 a, b の二家族の間で、
差を広げる移し替え(和は同じ)をすると、死蔵は減らない。 -/
theorem hoard_concentration {a b d B : ℝ} (hab : b ≤ a) (hd : 0 ≤ d) :
    max (a - B) 0 + max (b - B) 0 ≤ max (a + d - B) 0 + max (b - d - B) 0 := by
  rcases le_total (a - B) 0 with h1 | h1 <;> rcases le_total (b - B) 0 with h2 | h2 <;>
    rcases le_total (a + d - B) 0 with h3 | h3 <;> rcases le_total (b - d - B) 0 with h4 | h4 <;>
    simp only [max_eq_left, max_eq_right, h1, h2, h3, h4] <;> linarith
/-! ## 第 7 章 災害と、守りと、記憶

### 守りの最適な量
守りの資本 G の費用 c·G と、災害の被害 λ·D/(1 + G)(λ は頻度、D は守りのないときの被害)の和を最小にする。
x = 1 + G とおくと、c·x + A/x ≥ 2√(cA)(A = λD)。等号は x = √(A/c)。
ここでは s² = cA, s ≥ 0 として、c·x + A/x ≥ 2s を示す。 -/

theorem protection_amgm {c A x s : ℝ} (hc : 0 < c) (hx : 0 < x) (hs : s ^ 2 = c * A) :
    2 * s ≤ c * x + A / x := by
  have key : 0 ≤ c * (c * x ^ 2 - 2 * s * x + A) := by nlinarith [sq_nonneg (c * x - s)]
  have h2 : 0 ≤ c * x ^ 2 - 2 * s * x + A := (mul_nonneg_iff_of_pos_left hc).mp key
  have e : c * x + A / x = (c * x ^ 2 + A) / x := by field_simp
  rw [e, le_div_iff₀ hx]
  nlinarith

/-- 最適な守りの量 √(λD/c) − 1 は、頻度の信念 λ とともに増える。だから、信念が区間 [λlo, λhi] なら、
ふさわしい守りの量も区間になり(ノート I の `undominated_iff` と同じ形)、悲観的な社会は多く守る。 -/
theorem protection_monotone {D c l₁ l₂ : ℝ} (hD : 0 ≤ D) (hc : 0 < c) (h : l₁ ≤ l₂) :
    Real.sqrt (l₁ * D / c) - 1 ≤ Real.sqrt (l₂ * D / c) - 1 := by
  have : l₁ * D / c ≤ l₂ * D / c :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right h hD) hc.le
  linarith [Real.sqrt_le_sqrt this]

/-! ### 記録から頻度を学ぶ(ガンマ・ポアソン)
事前の信念をガンマ分布(a, b)とし、t 年で n 回の災害を見ると、事後の平均は (a + n)/(b + t)。
これは、事前の平均 a/b と、記録の頻度 n/t を、重み b/(b + t) と t/(b + t) で混ぜたものである。
記録が長くなるほど(t が大きいほど)、記録の頻度に寄る。 -/

theorem posterior_mean_mix {a b n t : ℝ} (hb : 0 < b) (ht : 0 < t) :
    (a + n) / (b + t) = b / (b + t) * (a / b) + t / (b + t) * (n / t) := by
  have hbt : 0 < b + t := by linarith
  field_simp

/-- 忘れる社会は確信できない: 記録を毎年 ω(< 1)の割合しか残さないと、記録の実効の年数
Σ ωᵏ は、何年たっても 1/(1 − ω) を超えない。だから信念の区間は、ある幅より狭くならない
(ノート I の `crisis_collapse` が言う「区間が一点に縮む」ことは、忘れる社会では起きない)。 -/
theorem forgetting_bound {ω : ℝ} (h0 : 0 ≤ ω) (h1 : ω < 1) (N : ℕ) :
    ∑ k ∈ range N, ω ^ k ≤ 1 / (1 - ω) := by
  have hne : ω ≠ 1 := ne_of_lt h1
  rw [geom_sum_eq hne N]
  have hpos : 0 < 1 - ω := by linarith
  have hp : 0 ≤ ω ^ N := pow_nonneg h0 N
  rw [show (ω ^ N - 1) / (ω - 1) = (1 - ω ^ N) / (1 - ω) by
        rw [← neg_sub, ← neg_sub 1 ω, neg_div_neg_eq]]
  exact div_le_div_of_nonneg_right (by linarith) hpos.le
end SimbdEconomy
