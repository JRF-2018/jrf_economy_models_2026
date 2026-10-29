/-
  必需品と贅沢品の宇宙的独立関係(JRF『宗教学雑考集』、2025・2026 年の再考)の Lean による定式化
  文責: JRF / 定式化・証明: Claude Opus 5.5
  Lean 4 v4.34.1 + Mathlib
-/
import Mathlib

open Filter Topology Finset

namespace CosmicIndependence
/-! ## 第 2 章 絶滅の確率 1

毎期、少なくとも確率 ε > 0 で破局が起きるなら、n 期生き延びる確率は (1 - ε)^n 以下で、0 に収束する。 -/

theorem survival_const_tendsto_zero {ε : ℝ} (h0 : 0 < ε) (h1 : ε ≤ 1) :
    Tendsto (fun n : ℕ => (1 - ε) ^ n) atTop (𝓝 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (by linarith) (by linarith)
/-! ## 第 2.3 節 終末論法: 自己標本化の仮定(SSA)と自己指示の仮定(SIA)

仮説は二つ。「小さい人類」(観測者の総数 Ns)の事前確率 p と、「大きい人類」(総数 Nl > Ns)の事前確率 1 − p。
観測は「自分の生まれた順番が r で、r ≤ Ns」(どちらの仮説とも両立する)。

* SSA: 自分は、その世界の観測者から一様に選ばれた一人とみなす。順番 r の尤度は 1 / N。
* SIA: さらに、観測者の多い世界ほど、自分がいる確率が高いとみなす。事前確率に N を掛けてから、SSA と同じ尤度を掛ける。 -/

/-- SSA のもとでの、小さい人類の事後確率。 -/
noncomputable def ssaPost (p Ns Nl : ℝ) : ℝ := (p / Ns) / (p / Ns + (1 - p) / Nl)

/-- SIA のもとでの、小さい人類の事後確率。 -/
noncomputable def siaPost (p Ns Nl : ℝ) : ℝ :=
  (p * Ns * (1 / Ns)) / (p * Ns * (1 / Ns) + (1 - p) * Nl * (1 / Nl))

/-- 終末論法: SSA のもとでは、自分の順番を見ただけで、小さい人類(早い終末)の確率が事前より上がる。 -/
theorem ssa_doomsday_shift {p Ns Nl : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hs : 0 < Ns) (hsl : Ns < Nl) :
    p < ssaPost p Ns Nl := by
  have hl : 0 < Nl := lt_trans hs hsl
  have e : ssaPost p Ns Nl = p * Nl / (p * Nl + (1 - p) * Ns) := by
    unfold ssaPost; field_simp
  rw [e, lt_div_iff₀ (by nlinarith)]
  nlinarith [mul_pos hp0 (sub_pos.mpr hp1), mul_pos (mul_pos hp0 (sub_pos.mpr hp1)) (sub_pos.mpr hsl)]

/-- 大きい人類の総数が大きくなるほど(無限に近づくほど)、SSA のもとでは、小さい人類の事後確率は 1 に近づく。
第 3 章で、永遠に生き延びる(総数が無限になる)確率が正でも、SSA のもとでは終末論法が別の形で生き残る理由。 -/
theorem ssa_large_world_penalized {p Ns : ℝ} (hp0 : 0 < p) (hs : 0 < Ns) :
    Tendsto (fun Nl : ℝ => ssaPost p Ns Nl) atTop (𝓝 1) := by
  have h1 : Tendsto (fun Nl : ℝ => (1 - p) / Nl) atTop (𝓝 0) := by
    simpa using tendsto_const_nhds.div_atTop tendsto_id
  have h2 : Tendsto (fun Nl : ℝ => p / Ns + (1 - p) / Nl) atTop (𝓝 (p / Ns + 0)) :=
    tendsto_const_nhds.add h1
  have hne : p / Ns + 0 ≠ 0 := by rw [add_zero]; positivity
  have := (tendsto_const_nhds (x := p / Ns)).div h2 hne
  rw [add_zero, div_self (ne_of_gt (div_pos hp0 hs))] at this
  exact this

/-- 自己指示の仮定: SIA のもとでは、自分の順番を見ても、事後確率は事前確率のまま。終末論法は打ち消される。 -/
theorem sia_cancels_doomsday {p Ns Nl : ℝ} (hs : 0 < Ns) (hl : 0 < Nl) :
    siaPost p Ns Nl = p := by
  unfold siaPost
  have a : p * Ns * (1 / Ns) = p := by field_simp
  have b : (1 - p) * Nl * (1 / Nl) = 1 - p := by field_simp
  rw [a, b]; ring_nf

/-- 仮説がいくつあっても同じ: どの仮説とも両立する観測なら、SIA の事後確率は事前確率に一致する。 -/
theorem sia_cancels_general {ι : Type*} (S : Finset ι) (w N : ι → ℝ) (hN : ∀ i ∈ S, 0 < N i)
    (hw : ∑ j ∈ S, w j = 1) (i : ι) (hi : i ∈ S) :
    (w i * N i * (1 / N i)) / ∑ j ∈ S, w j * N j * (1 / N j) = w i := by
  have e : ∀ j ∈ S, w j * N j * (1 / N j) = w j := fun j hj => by
    have := (hN j hj).ne'; field_simp
  rw [Finset.sum_congr rfl e, hw, e i hi, div_one]

/-- 「僭越な哲学者」: SIA は、順番を見る前から、観測者の多い世界を強く支持する。
大きい人類の総数が大きくなるほど、その(順番を見る前の)確率は 1 に近づく。作者の多宇宙の論駁は、この性質を使っている。 -/
theorem sia_presumptuous {p Ns : ℝ} (hp1 : p < 1) (_hs : 0 < Ns) :
    Tendsto (fun Nl : ℝ => (1 - p) * Nl / (p * Ns + (1 - p) * Nl)) atTop (𝓝 1) := by
  have hq : 0 < 1 - p := sub_pos.mpr hp1
  have e : ∀ᶠ Nl : ℝ in atTop, (1 - p) * Nl / (p * Ns + (1 - p) * Nl) = 1 / (p * Ns / ((1 - p) * Nl) + 1) := by
    filter_upwards [eventually_gt_atTop 0] with Nl hNl
    have : (1 - p) * Nl ≠ 0 := by positivity
    field_simp
  rw [tendsto_congr' e]
  have h0 : Tendsto (fun Nl : ℝ => p * Ns / ((1 - p) * Nl)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop (tendsto_id.const_mul_atTop hq)
  have := (tendsto_const_nhds (x := (1 : ℝ))).div (h0.add tendsto_const_nhds) (by norm_num : (0 : ℝ) + 1 ≠ 0)
  rw [zero_add, div_one] at this
  exact this
/-! ## 第 3 章 確率 0 でも望みがある: 危険が減っていけば、永遠に生き延びる確率は正になりうる

毎期の破局の確率を a k とすると、n 期生き延びる確率は ∏ (1 - a k) で、これは 1 - ∑ a k 以上である
(ワイエルシュトラスの不等式)。だから、危険の総和が 1 より小さく抑えられれば(発明が危険を減らし続ければ)、
生き延びる確率は、どれだけ先でも正のまま残る。 -/

theorem prod_one_sub_ge (a : ℕ → ℝ) (h0 : ∀ k, 0 ≤ a k) (h1 : ∀ k, a k ≤ 1) :
    ∀ n, 1 - ∑ k ∈ range n, a k ≤ ∏ k ∈ range n, (1 - a k) := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ, prod_range_succ]
    have hS : 0 ≤ ∑ k ∈ range n, a k := sum_nonneg (fun k _ => h0 k)
    have hP : 0 ≤ 1 - a n := by linarith [h1 n]
    nlinarith [mul_le_mul_of_nonneg_right ih hP, h0 n, mul_nonneg hS (h0 n)]

theorem survival_positive_of_summable (a : ℕ → ℝ) (h0 : ∀ k, 0 ≤ a k) (h1 : ∀ k, a k ≤ 1)
    (hs : Summable a) (hlt : ∑' k, a k < 1) :
    ∀ n, 0 < 1 - ∑' k, a k ∧ 1 - ∑' k, a k ≤ ∏ k ∈ range n, (1 - a k) := by
  intro n
  refine ⟨by linarith, ?_⟩
  have := prod_one_sub_ge a h0 h1 n
  have hle : ∑ k ∈ range n, a k ≤ ∑' k, a k := hs.sum_le_tsum _ (fun k _ => h0 k)
  linarith
/-! ## 第 4 章 「0 × ∞」には値がない

確率が 0 に向かい、価値が無限に向かうとき、その積(期待値)の行き先は、向かい方しだいで何にでもなる。
とくに、「0 × ∞ の値」として決まった一つの数は存在しない。 -/

theorem zero_mul_infty_no_value :
    ¬ ∃ c : ℝ, ∀ p V : ℕ → ℝ, Tendsto p atTop (𝓝 0) → Tendsto V atTop atTop →
      Tendsto (fun n => p n * V n) atTop (𝓝 c) := by
  rintro ⟨c, hc⟩
  have hV : Tendsto (fun n : ℕ => (n : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have hp1 : Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (𝓝 0) :=
    tendsto_one_div_add_atTop_nhds_zero_nat
  have hp2 : Tendsto (fun n : ℕ => 2 * (1 / ((n : ℝ) + 1))) atTop (𝓝 0) := by
    simpa using hp1.const_mul 2
  have e1 : (fun n : ℕ => 1 / ((n : ℝ) + 1) * ((n : ℝ) + 1)) = fun _ => (1 : ℝ) := by
    funext n; field_simp
  have e2 : (fun n : ℕ => 2 * (1 / ((n : ℝ) + 1)) * ((n : ℝ) + 1)) = fun _ => (2 : ℝ) := by
    funext n; field_simp
  have l1 := hc _ _ hp1 hV
  have l2 := hc _ _ hp2 hV
  rw [e1] at l1
  rw [e2] at l2
  have : (1 : ℝ) = c := tendsto_nhds_unique tendsto_const_nhds l1
  have : (2 : ℝ) = c := tendsto_nhds_unique tendsto_const_nhds l2
  linarith

/-- 収束の速さを決める指数 r しだいで、期待値 ε^r · (1/ε) = ε^(r-1) は 0 にも、有限の値にも、無限にもなる。
ここでは r = 2(0 に行く)と r = 1(1 にとどまる)を示す。 -/
theorem expectation_depends_on_rate :
    Tendsto (fun n : ℕ => (1 / ((n : ℝ) + 1)) ^ 2 * ((n : ℝ) + 1)) atTop (𝓝 0) ∧
    Tendsto (fun n : ℕ => (1 / ((n : ℝ) + 1)) ^ 1 * ((n : ℝ) + 1)) atTop (𝓝 1) := by
  constructor
  · have e : (fun n : ℕ => (1 / ((n : ℝ) + 1)) ^ 2 * ((n : ℝ) + 1)) = fun n : ℕ => 1 / ((n : ℝ) + 1) := by
      funext n; field_simp
    rw [e]; exact tendsto_one_div_add_atTop_nhds_zero_nat
  · have e : (fun n : ℕ => (1 / ((n : ℝ) + 1)) ^ 1 * ((n : ℝ) + 1)) = fun _ => (1 : ℝ) := by
      funext n; field_simp
    rw [e]; exact tendsto_const_nhds
/-! ## 第 5 章(2) 独立性の正体: どの配分も、ある信念のもとでは最適である

贅沢品の割合を a、贅沢品の値打ちの「見込み」を θ ≥ 0 とする(θ は、贅沢品が将来もたらすものへの信念で、
事前には確かめられない)。社会の目的を、贅沢品の見込みの値打ち θ·a から、必需品を削る痛み a²/2 を引いたものとする。 -/

/-- 信念 θ のもとでの、贅沢品の割合 a の値打ち。 -/
noncomputable def F (θ a : ℝ) : ℝ := θ * a - a ^ 2 / 2

/-- 信念 θ がわかっていれば、最適な割合はただ一つ(a = θ)に決まる。 -/
theorem unique_optimum (θ b : ℝ) (hb : b ≠ θ) : F θ b < F θ θ := by
  unfold F
  have : 0 < (b - θ) ^ 2 := by positivity
  nlinarith

/-- どの割合 a ≥ 0 も、ある信念 θ ≥ 0 のもとでは最適である(どの配分も合理化できる)。 -/
theorem every_allocation_rationalizable (a : ℝ) (ha : 0 ≤ a) :
    ∃ θ, 0 ≤ θ ∧ ∀ b, F θ b ≤ F θ a :=
  ⟨a, ha, fun b => by unfold F; nlinarith [sq_nonneg (b - a)]⟩
/-! ## 第 5 章(3) ナイト的不確実性のもとでの独立性(Bewley の不完備な選好)

信念がひとつに決まらず、区間 [θlo, θhi] のどこにあるかしかわからないとする。
配分 a が「支配される」とは、別の配分 b が、区間のどの信念のもとでも、a より厳密によいことである。
支配されない配分は、ちょうど区間 [θlo, θhi] であり、そのどれを選んでも、データだけからは咎められない。
これが、「必需品の下限を超えて、贅沢品をどれだけ作るかは決められない」の、もっとも素直な数学的な意味である。 -/

def Dominated (θlo θhi a : ℝ) : Prop :=
  ∃ b, ∀ θ, θlo ≤ θ → θ ≤ θhi → F θ a < F θ b

theorem undominated_iff (θlo θhi a : ℝ) (_h : θlo ≤ θhi) :
    ¬ Dominated θlo θhi a ↔ θlo ≤ a ∧ a ≤ θhi := by
  unfold Dominated F
  constructor
  · intro hnd
    by_contra hout
    apply hnd
    rcases not_and_or.mp hout with hlo | hhi
    · push Not at hlo
      exact ⟨θlo, fun θ h1 _ => by nlinarith⟩
    · push Not at hhi
      exact ⟨θhi, fun θ _ h2 => by nlinarith⟩
  · rintro ⟨h1, h2⟩ ⟨b, hb⟩
    have := hb a h1 h2
    nlinarith [sq_nonneg (b - a)]

/-- 悲観的な社会(最悪の信念で評価する、Gilboa–Schmeidler の maxmin)は、区間の下端 θlo を選ぶ。 -/
theorem maxmin_picks_lower (θlo θhi a : ℝ) (ha : 0 ≤ a) :
    (∀ θ, θlo ≤ θ → θ ≤ θhi → F θlo a ≤ F θ a) ∧ F θlo a ≤ F θlo θlo := by
  refine ⟨fun θ h1 _ => by unfold F; nlinarith, ?_⟩
  unfold F; nlinarith [sq_nonneg (a - θlo)]

/-- 楽観的な社会(最良の信念で評価する)は、区間の上端 θhi を選ぶ。 -/
theorem maxmax_picks_upper (θlo θhi a : ℝ) (ha : 0 ≤ a) :
    (∀ θ, θlo ≤ θ → θ ≤ θhi → F θ a ≤ F θhi a) ∧ F θhi a ≤ F θhi θhi := by
  refine ⟨fun θ _ h2 => by unfold F; nlinarith, ?_⟩
  unfold F; nlinarith [sq_nonneg (a - θhi)]
/-! ## 第 5 章(1) 必需品の下限

全資源 R のうち、割合 α を贅沢品に回す。必需品の必要量 Bmin を確保するには α ≤ 1 - Bmin / R。 -/

theorem necessity_floor {R Bmin α : ℝ} (hR : 0 < R) :
    Bmin ≤ (1 - α) * R ↔ α ≤ 1 - Bmin / R := by
  rw [show α ≤ 1 - Bmin / R ↔ Bmin / R ≤ 1 - α by constructor <;> intro h <;> linarith,
    div_le_iff₀ hR]
/-- 信念に重みを付けて平均すると(θ₁ に w、θ₂ に 1 − w)、値打ちは、平均した信念 wθ₁ + (1 − w)θ₂ のもとでの値打ちに一致する。
だから、信念どうしを比べる重み(信念についての信念)があれば、区間はただ一つの信念に縮み、最適も一つに決まる。 -/
theorem mixture_collapses (w θ₁ θ₂ a : ℝ) :
    w * F θ₁ a + (1 - w) * F θ₂ a = F (w * θ₁ + (1 - w) * θ₂) a := by
  unfold F; ring

/-- 区間の中のどの配分 a も、端の信念に適当な重みを付けた平均のもとでは最適になる。
支配されない配分の区間は、「信念に付けうるすべての重み」で選ばれる最適の集まりである。 -/
theorem interval_is_all_weightings (θlo θhi a : ℝ) (h1 : θlo ≤ a) (h2 : a ≤ θhi) (hlt : θlo < θhi) :
    ∃ w, 0 ≤ w ∧ w ≤ 1 ∧ ∀ b, F (w * θlo + (1 - w) * θhi) b ≤ F (w * θlo + (1 - w) * θhi) a := by
  refine ⟨(θhi - a) / (θhi - θlo), ?_, ?_, ?_⟩
  · apply div_nonneg <;> linarith
  · rw [div_le_one (by linarith)]; linarith
  · intro b
    have hne : θhi - θlo ≠ 0 := by linarith
    have hθ : (θhi - a) / (θhi - θlo) * θlo + (1 - (θhi - a) / (θhi - θlo)) * θhi = a := by
      field_simp [hne]; ring
    rw [hθ]; unfold F; nlinarith [sq_nonneg (b - a)]
/-! ## 第 7 章 国が複数あると、独立ではなくなる

国 i の目的に、相手の国 j と配分が違うことの不利 κ (a_i - a_j)²/2 を加える(軍事や経済の競争)。
すると、国 i の最もよい配分は相手の配分に依存し、(θ + κ a_j) / (1 + κ) になる。
相手との差は、自分の信念と相手の配分の差の 1 / (1 + κ) に縮む。 -/

noncomputable def G (κ θ aj ai : ℝ) : ℝ := θ * ai - ai ^ 2 / 2 - κ * (ai - aj) ^ 2 / 2

theorem best_response (κ θ aj b : ℝ) (hκ : 0 ≤ κ) :
    G κ θ aj b ≤ G κ θ aj ((θ + κ * aj) / (1 + κ)) := by
  unfold G
  have h1 : (0 : ℝ) < 1 + κ := by linarith
  have key : G κ θ aj ((θ + κ * aj) / (1 + κ)) - G κ θ aj b
      = (1 + κ) * (b - (θ + κ * aj) / (1 + κ)) ^ 2 / 2 := by
    unfold G; field_simp; ring
  unfold G at key
  nlinarith [sq_nonneg (b - (θ + κ * aj) / (1 + κ))]

theorem best_response_gap (κ θ aj : ℝ) (hκ : 0 ≤ κ) :
    (θ + κ * aj) / (1 + κ) - aj = (θ - aj) / (1 + κ) := by
  have h1 : (1 : ℝ) + κ ≠ 0 := by linarith
  field_simp; ring
/-! ## 第 8 章 危機の近傍: 信念の区間が縮めば、配分は一点に決まる

危機が近づき、どの贅沢品が効くかがわかってくると、信念の区間 [θlo, θhi] が縮む。
区間の幅が 0 になれば、支配されない配分はただ一つになる(贅沢品から必需品への「転嫁」)。 -/

theorem crisis_collapse (θ a : ℝ) : ¬ Dominated θ θ a ↔ a = θ := by
  rw [undominated_iff θ θ a le_rfl]
  constructor
  · rintro ⟨h1, h2⟩; linarith
  · rintro rfl; exact ⟨le_rfl, le_rfl⟩
end CosmicIndependence
