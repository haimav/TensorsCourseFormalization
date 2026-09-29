/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Variational
import TensorCourse.TensorEigenvalues.BetaBound
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.GroupTheory.Perm.Fin

/-!
# Convexity of the Shifted Tensor Rayleigh Quotient (Lecture 8, `lem:shift-convex`)

For an **even-order** symmetric tensor `A`, Lecture 8 shows that the *shifted*
tensor Rayleigh quotient

  `R_{A + μ E}(x) = xᵀ A x + μ (xᵀx)^{(order)/2}`

becomes **convex on all of `ℝⁿ`** once the shift `μ` is large enough. (The
identity-on-sphere tensor `E` never needs to be constructed: on the real line
`(xᵀx)^{order/2}` is exactly the extra term, so we write it directly.)

Here the tensor has Lean-order `D + 2` (so its order is even iff `D` is even),
and `rayleigh A x` is the order-`(D+2)` full contraction `xᵀ (A ⬝ₜ x)`.

## Main definition

* `CubicTensor.shiftedRayleigh A μ x = rayleigh A x + μ (∑ᵢ xᵢ²)^{D/2+1}`.

## Main result

* `CubicTensor.convexOn_shiftedRayleigh_of_sum` — **the practical convexity
  criterion.** For symmetric `A` of even order `D+2` and

    `μ ≥ (D+1) · ∑_{i₁…i_{D+2}} |a_{i₁…i_{D+2}}|`,

  the shifted Rayleigh quotient is convex on `ℝⁿ`.

  This is the criterion actually used in practice: it is `lem:shift-convex`
  composed with the user-friendly bound `lem:beta-bound`
  (`CubicTensor.beta_le`, `β(A) ≤ (D+1)·∑|a|`), so that the abstract shift
  threshold `μ > β(A)` is replaced by the computable `μ ≥ (D+1)·∑|a|`. The
  proof does not go through the spectral radius at all: it bounds the Hessian
  of the Rayleigh term directly by `∑|a|` via an AM-GM argument, which is
  exactly the composite of the two lecture lemmas.

## Proof strategy

Convexity is proved through the classical one-dimensional second-derivative
test, restricted to every line `t ↦ f(x + t v)`:

1. `convexOn_of_convexOn_lines` — a function is convex if its restriction to
   every affine line is convex.
2. On a fixed line, `convexOn_of_deriv2_nonneg'` reduces convexity to the
   nonnegativity of the second derivative.
3. The first derivative of the Rayleigh term is computed by reusing
   `hasFDerivAt_rayleigh`; its *second* derivative is obtained by reusing
   `hasFDerivAt_rayleigh` a **second** time, on the once-contracted tensor
   `contract1 A v` (which is again symmetric). This avoids re-deriving the
   Hessian symmetry collapse by hand.
4. The second derivative of the shift term `(∑xᵢ²)^{D/2+1}` is elementary
   calculus. The domination `deriv² ≥ 0` then follows from the AM-GM bound on
   the Rayleigh Hessian by `∑|a|`.
-/

noncomputable section

open Finset BigOperators Matrix
open scoped CubicTensor ENNReal NNReal RealInnerProductSpace

namespace CubicTensor

variable {N D : ℕ}

/-- **The shifted tensor Rayleigh quotient** (Lecture 8). For an order-`(D+2)`
    tensor `A`, `shiftedRayleigh A μ x = xᵀ(A ⬝ₜ x) + μ (∑ᵢ xᵢ²)^{D/2+1}`. When
    `D` is even the exponent `D/2 + 1 = (D+2)/2` matches the lecture's
    `(xᵀx)^{order/2}`. -/
def shiftedRayleigh (A : CubicTensor ℝ (D + 2) N) (μ : ℝ) (x : Fin N → ℝ) : ℝ :=
  rayleigh A x + μ * (∑ i, x i ^ 2) ^ (D / 2 + 1)

/-! ### Step 1 — the line-restriction bridge -/

/-- **Convexity from convexity along lines.** A function on a real normed space
    is convex on `univ` if its restriction to every affine line `t ↦ f(x + t v)`
    is convex on `univ`. -/
theorem convexOn_of_convexOn_lines {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] {f : E → ℝ}
    (h : ∀ x v : E, ConvexOn ℝ Set.univ (fun t : ℝ => f (x + t • v))) :
    ConvexOn ℝ Set.univ f := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  have hline := (h x (y - x)).2 (Set.mem_univ (0 : ℝ)) (Set.mem_univ (1 : ℝ)) ha hb hab
  have e0 : x + (0 : ℝ) • (y - x) = x := by rw [zero_smul, add_zero]
  have e1 : x + (1 : ℝ) • (y - x) = y := by rw [one_smul, add_sub_cancel]
  have eb : x + (a • (0 : ℝ) + b • (1 : ℝ)) • (y - x) = a • x + b • y := by
    rw [smul_eq_mul, smul_eq_mul, mul_zero, mul_one, zero_add, smul_sub,
      show a = 1 - b from by linarith, sub_smul, one_smul]
    abel
  simp only [e0, e1, eb] at hline
  exact hline

/-! ### The once-contracted tensor `contract1 A v` -/

/-- Cubic-tensor transpose acts by precomposition with `σ.symm` (the `▸` cast in
    `Tensor.transpose` is along a constant motive, hence the identity). -/
private theorem cubicTranspose_apply {O : ℕ} (σ : Equiv.Perm (Fin O))
    (T : CubicTensor ℝ O N) (idx : Fin O → Fin N) :
    T.transpose σ idx = T (fun d => idx (σ.symm d)) := by
  simp only [Tensor.transpose_apply]
  congr 1
  funext d
  apply eq_of_heq
  exact eqRec_heq_iff_heq.mpr HEq.rfl

/-- Contract the **first** mode of an order-`(D+2)` tensor `A` against `v`,
    leaving the symmetric order-`(D+1)` tensor
    `(contract1 A v) j = ∑ᵢ vᵢ · A(i, j)`.  Its Rayleigh quotient is the
    directional slope of `R_A`, which is how it enters the second derivative. -/
def contract1 (A : CubicTensor ℝ (D + 2) N) (v : Fin N → ℝ) : CubicTensor ℝ (D + 1) N :=
  fun j => ∑ i, v i * A (Fin.cons i j)

/-- **`contract1 A v` is symmetric** whenever `A` is: contracting one mode of a
    fully symmetric tensor leaves the remaining modes symmetric. -/
theorem contract1_isSymmetric {A : CubicTensor ℝ (D + 2) N} (hA : A.IsSymmetric)
    (v : Fin N → ℝ) : (contract1 A v).IsSymmetric := by
  intro σ
  funext idx
  rw [cubicTranspose_apply]
  change (∑ i, v i * A (Fin.cons i (fun d => idx (σ.symm d)))) = ∑ i, v i * A (Fin.cons i idx)
  refine Finset.sum_congr rfl (fun i _ => ?_)
  congr 1
  -- `A (cons i (idx ∘ σ.symm)) = A (cons i idx)` via the permutation that fixes
  -- slot 0 and acts as `σ.symm` on the tail.
  set τ' : Equiv.Perm (Fin (D + 2)) := Equiv.Perm.decomposeFin.symm (0, σ.symm) with hτ'
  have hcons : (Fin.cons i (fun d => idx (σ.symm d)) : Fin (D + 2) → Fin N)
      = fun d => (Fin.cons i idx : Fin (D + 2) → Fin N) (τ' d) := by
    funext d
    refine Fin.cases ?_ ?_ d
    · simp [hτ', Equiv.Perm.decomposeFin_symm_apply_zero, Fin.cons_zero]
    · intro k
      simp [hτ', Equiv.Perm.decomposeFin_symm_apply_succ, Fin.cons_succ,
        Equiv.swap_self]
  rw [hcons]
  exact apply_perm hA τ' (Fin.cons i idx)

/-- **The Rayleigh quotient of `contract1 A v` is the directional slope of
    `R_A`.**  `rayleigh (contract1 A v) w = ∑ᵢ (A ⬝ₜ w)ᵢ · vᵢ`.  Up to the
    constant `D+2` this is the first derivative of the Rayleigh term along the
    direction `v`. -/
theorem rayleigh_contract1 (A : CubicTensor ℝ (D + 2) N) (v w : Fin N → ℝ) :
    rayleigh (contract1 A v) w = ∑ i, (A ⬝ₜ w) i * v i := by
  rw [rayleigh_eq_sum]
  simp only [contract1, apply_apply, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun idx _ => by ring))

/-! ### Step 3a — the line derivative of a Rayleigh quotient -/

/-- **First derivative of `R_T` along a line.** For symmetric `T` of order
    `E+1`, `d/dt R_T(x + t v) = (E+1) ∑ᵢ (T ⬝ₜ (x+tv))ᵢ vᵢ`.  Obtained by
    composing `hasFDerivAt_rayleigh` with the affine line `t ↦ x + t v`. -/
theorem hasDerivAt_rayleigh_line {E : ℕ} (T : CubicTensor ℝ (E + 1) N)
    (hT : T.IsSymmetric) (x v : Fin N → ℝ) (t : ℝ) :
    HasDerivAt (fun s : ℝ => rayleigh T (x + s • v))
      (((E : ℝ) + 1) * ∑ i, (T ⬝ₜ (x + t • v)) i * v i) t := by
  have hline : HasDerivAt (fun s : ℝ => x + s • v) v t := by
    simpa using ((hasDerivAt_id t).smul_const v).const_add x
  have hcomp := (hasFDerivAt_rayleigh hT (x + t • v)).comp_hasDerivAt t hline
  simpa only [Function.comp_def, ContinuousLinearMap.smul_apply, bilinFormCLM_apply,
    smul_eq_mul] using hcomp

/-! ### Step 4 — the AM-GM bound on the Rayleigh Hessian -/

/-- Reindex a sum over full multi-indices `Fin (D+2) → Fin N` as a triple sum
    over the first index `i`, the second index `j`, and the remaining `D`
    indices `rest`, via a double application of `Fin.consEquiv`.  (Local copy of
    the `private` reindexer in `BetaBound.lean`.) -/
private theorem sum_cons2 {M : Type*} [AddCommMonoid M]
    (f : (Fin (D + 2) → Fin N) → M) :
    ∑ idx : Fin (D + 2) → Fin N, f idx
      = ∑ i : Fin N, ∑ j : Fin N, ∑ rest : Fin D → Fin N,
          f (Fin.cons i (Fin.cons j rest)) := by
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) f,
      Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have step : ∀ tail : Fin (D + 1) → Fin N,
      (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) (i, tail) = Fin.cons i tail := by
    intro tail; funext d; exact Fin.consEquiv_apply _ _ _
  simp only [step]
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N))
        (fun tail => f (Fin.cons i tail)), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  have hc : (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) (j, rest) = Fin.cons j rest := by
    funext d; exact Fin.consEquiv_apply _ _ _
  rw [hc]

/-- **AM-GM bound on the Rayleigh Hessian (the fallback of `lem:shift-convex`).**
    The quadratic form `vᵀ (Hessian of R_A at w) v = (D+2)(D+1) · S`, where
    `S = ∑ⱼ (contract1 A v ⬝ₜ w)ⱼ vⱼ`, is bounded in absolute value by
    `(∑|a|) · ‖w‖^D · ‖v‖²`.  This is exactly the composite of `lem:shift-convex`
    (which controls the Hessian by `β(A)`) and `lem:beta-bound`
    (`β(A) ≤ (D+1)∑|a|`), obtained here directly, without the spectral radius,
    by pushing `|·|` through the multilinear form and using `|vᵢ||vⱼ| ≤ ‖v‖²`
    and `∏|w| ≤ ‖w‖^D`. -/
theorem abs_contract1_apply_le (A : CubicTensor ℝ (D + 2) N) (hD : Even D)
    (v w : Fin N → ℝ) :
    |∑ j, (contract1 A v ⬝ₜ w) j * v j|
      ≤ (∑ idx : Fin (D + 2) → Fin N, |A idx|)
          * (∑ i, w i ^ 2) ^ (D / 2) * (∑ i, v i ^ 2) := by
  set Q : ℝ := ∑ i, w i ^ 2 with hQdef
  set V : ℝ := ∑ i, v i ^ 2 with hVdef
  have hQnn : 0 ≤ Q := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  have hVnn : 0 ≤ V := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  -- `|vᵢ||vⱼ| ≤ V`.
  have hvsq : ∀ k, v k ^ 2 ≤ V :=
    fun k => Finset.single_le_sum (fun l _ => sq_nonneg _) (mem_univ k)
  have hvv : ∀ i j : Fin N, |v i| * |v j| ≤ V := by
    intro i j
    nlinarith [two_mul_le_add_sq |v i| |v j|, sq_abs (v i), sq_abs (v j),
      hvsq i, hvsq j, abs_nonneg (v i), abs_nonneg (v j)]
  -- `∏ |w(r d)| ≤ Q^(D/2)`.
  have hwsq : ∀ k, w k ^ 2 ≤ Q :=
    fun k => Finset.single_le_sum (fun l _ => sq_nonneg _) (mem_univ k)
  have hwle : ∀ k, |w k| ≤ Real.sqrt Q := by
    intro k; rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt (hwsq k)
  have hprod : ∀ r : Fin D → Fin N, ∏ d, |w (r d)| ≤ Q ^ (D / 2) := by
    intro r
    have h1 : ∏ d : Fin D, |w (r d)| ≤ ∏ _d : Fin D, Real.sqrt Q :=
      Finset.prod_le_prod (fun d _ => abs_nonneg _) (fun d _ => hwle (r d))
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin] at h1
    have hDD : 2 * (D / 2) = D := Nat.two_mul_div_two_of_even hD
    have h2 : (Real.sqrt Q) ^ D = Q ^ (D / 2) := by
      conv_lhs => rw [← hDD]
      rw [pow_mul, Real.sq_sqrt hQnn]
    rwa [h2] at h1
  -- Reindex `∑ⱼ∑ᵣ∑ᵢ |A(cons i(cons j r))| = ∑ idx |A idx|`.
  have hreindex : (∑ j, ∑ r : Fin D → Fin N, ∑ i, |A (Fin.cons i (Fin.cons j r))|)
      = ∑ idx : Fin (D + 2) → Fin N, |A idx| := by
    rw [sum_cons2 (fun idx => |A idx|)]
    conv_rhs => rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [Finset.sum_comm]
  -- Expand `S` into a triple sum.
  have hSexp : (∑ j, (contract1 A v ⬝ₜ w) j * v j)
      = ∑ j, ∑ r : Fin D → Fin N, ∑ i,
          v i * A (Fin.cons i (Fin.cons j r)) * (∏ d, w (r d)) * v j := by
    simp only [apply_apply, contract1, Finset.sum_mul]
  rw [hSexp]
  calc |∑ j, ∑ r : Fin D → Fin N, ∑ i,
            v i * A (Fin.cons i (Fin.cons j r)) * (∏ d, w (r d)) * v j|
      ≤ ∑ j, ∑ r : Fin D → Fin N, ∑ i,
            |v i * A (Fin.cons i (Fin.cons j r)) * (∏ d, w (r d)) * v j| := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum (fun j _ => ?_))
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum (fun r _ => ?_))
        exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, ∑ r : Fin D → Fin N, ∑ i,
            |A (Fin.cons i (Fin.cons j r))| * (V * Q ^ (D / 2)) := by
        refine Finset.sum_le_sum (fun j _ => Finset.sum_le_sum (fun r _ =>
          Finset.sum_le_sum (fun i _ => ?_)))
        have e : |v i * A (Fin.cons i (Fin.cons j r)) * (∏ d, w (r d)) * v j|
            = |A (Fin.cons i (Fin.cons j r))| * (|v i| * |v j|) * (∏ d, |w (r d)|) := by
          rw [abs_mul, abs_mul, abs_mul, Finset.abs_prod]; ring
        rw [e]
        calc |A (Fin.cons i (Fin.cons j r))| * (|v i| * |v j|) * (∏ d, |w (r d)|)
            ≤ |A (Fin.cons i (Fin.cons j r))| * V * Q ^ (D / 2) := by
              gcongr
              · exact hvv i j
              · exact hprod r
          _ = |A (Fin.cons i (Fin.cons j r))| * (V * Q ^ (D / 2)) := by ring
    _ = (∑ j, ∑ r : Fin D → Fin N, ∑ i, |A (Fin.cons i (Fin.cons j r))|)
          * (V * Q ^ (D / 2)) := by
        simp only [← Finset.sum_mul]
    _ = (∑ idx : Fin (D + 2) → Fin N, |A idx|) * Q ^ (D / 2) * V := by
        rw [hreindex]; ring

/-! ### The main theorem -/

set_option maxHeartbeats 1200000 in
-- The proof restricts to a line and runs a two-fold `HasDerivAt` computation over
-- the large multi-index set `Fin (D+2) → Fin N`; the bookkeeping needs extra heartbeats.
/-- **The shared second-derivative machinery of `lem:shift-convex`.**  The shifted
    Rayleigh quotient is convex on `ℝⁿ` whenever there is a constant `C ≥ 0` with
    `(D+1)·C ≤ μ` that lower-bounds the Rayleigh Hessian slope:

      `-(C · (∑ wᵢ²)^{D/2} · (∑ vᵢ²)) ≤ ∑ⱼ (contract1 A v ⬝ₜ w)ⱼ vⱼ`   (all `w`, `v`).

    Both concrete convexity criteria below are obtained by supplying such a `C`:
    the practical `∑|a|` bound (AM-GM) and the sharp spectral bound `β(A)/(D+1)`.
    The second derivative of the restriction to a line collapses to
    `(D+2)(D+1)·S + μ·(D+2)·Q^{D/2}·V ≥ (D+2)·Q^{D/2}·V·(μ − (D+1)·C) ≥ 0`. -/
theorem convexOn_shiftedRayleigh_of_Sbound (A : CubicTensor ℝ (D + 2) N)
    (hA : A.IsSymmetric) (hD : Even D) (μ : ℝ) (C : ℝ) (hC : 0 ≤ C)
    (hCμ : ((D : ℝ) + 1) * C ≤ μ)
    (hSbound : ∀ (w v : Fin N → ℝ),
        -(C * (∑ i, w i ^ 2) ^ (D / 2) * (∑ i, v i ^ 2))
          ≤ ∑ j, (contract1 A v ⬝ₜ w) j * v j) :
    ConvexOn ℝ Set.univ (shiftedRayleigh A μ) := by
  apply convexOn_of_convexOn_lines
  intro x v
  -- `f0` is the restriction of the shifted Rayleigh quotient to the line `x + t v`.
  set f0 : ℝ → ℝ := fun t => shiftedRayleigh A μ (x + t • v) with hf0
  -- The symmetric once-contracted tensor drives the second derivative of the Rayleigh term.
  set P : CubicTensor ℝ (D + 1) N := contract1 A v with hP
  have hPsym : P.IsSymmetric := contract1_isSymmetric hA v
  -- Coordinate line derivative: `d/dt (x + t v)ᵢ = vᵢ`.
  have hcoord : ∀ (i : Fin N) (t : ℝ),
      HasDerivAt (fun s : ℝ => (x + s • v) i) (v i) t := by
    intro i t
    have h : HasDerivAt (fun s : ℝ => x i + s * v i) (v i) t := by
      simpa using ((hasDerivAt_id t).mul_const (v i)).const_add (x i)
    have he : (fun s : ℝ => (x + s • v) i) = fun s : ℝ => x i + s * v i := by
      funext s; simp [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [he]; exact h
  -- `q t = ‖x + t v‖²` and its first/second derivatives.
  have hq_deriv : ∀ t, HasDerivAt (fun s : ℝ => ∑ i, (x + s • v) i ^ 2)
      (∑ i, 2 * (x + t • v) i * v i) t := by
    intro t
    refine HasDerivAt.fun_sum (fun i _ => ?_)
    simpa using (hcoord i t).pow 2
  have hQ1_deriv : ∀ t, HasDerivAt (fun s : ℝ => ∑ i, 2 * (x + s • v) i * v i)
      (∑ i, 2 * v i * v i) t := by
    intro t
    refine HasDerivAt.fun_sum (fun i _ => ?_)
    simpa using ((hcoord i t).const_mul (2 : ℝ)).mul_const (v i)
  -- First derivative of the Rayleigh term, packaged through `P = contract1 A v`.
  have hgR : ∀ t, HasDerivAt (fun s : ℝ => rayleigh A (x + s • v))
      (((D : ℝ) + 2) * rayleigh P (x + t • v)) t := by
    intro t
    have h := hasDerivAt_rayleigh_line A hA x v t
    rw [← rayleigh_contract1 A v (x + t • v)] at h
    have hc : ((D : ℝ) + 2) = ((D + 1 : ℕ) : ℝ) + 1 := by push_cast; ring
    rw [hP, hc]; exact h
  -- First derivative of the shift term.
  have hgN : ∀ t, HasDerivAt (fun s : ℝ => μ * (∑ i, (x + s • v) i ^ 2) ^ (D / 2 + 1))
      (μ * (((D / 2 + 1 : ℕ) : ℝ) * (∑ i, (x + t • v) i ^ 2) ^ (D / 2)
        * (∑ i, 2 * (x + t • v) i * v i))) t := by
    intro t
    have h := ((hq_deriv t).pow (D / 2 + 1)).const_mul μ
    simpa [Nat.add_sub_cancel] using h
  -- The explicit first and second derivatives of `f0`.
  set D1 : ℝ → ℝ := fun t =>
      ((D : ℝ) + 2) * rayleigh P (x + t • v)
      + μ * (((D / 2 + 1 : ℕ) : ℝ) * (∑ i, (x + t • v) i ^ 2) ^ (D / 2)
        * (∑ i, 2 * (x + t • v) i * v i)) with hD1
  set D2 : ℝ → ℝ := fun t =>
      ((D : ℝ) + 2) * (((D : ℝ) + 1) * ∑ j, (P ⬝ₜ (x + t • v)) j * v j)
      + μ * (((D / 2 + 1 : ℕ) : ℝ) *
          ((((D / 2 : ℕ) : ℝ) * (∑ i, (x + t • v) i ^ 2) ^ (D / 2 - 1)
              * (∑ i, 2 * (x + t • v) i * v i)) * (∑ i, 2 * (x + t • v) i * v i)
            + (∑ i, (x + t • v) i ^ 2) ^ (D / 2) * (∑ i, 2 * v i * v i))) with hD2
  have hg1 : ∀ t, HasDerivAt f0 (D1 t) t := by
    intro t
    rw [hf0, hD1]
    have hsum := (hgR t).add (hgN t)
    convert hsum using 1
  have hg2 : ∀ t, HasDerivAt D1 (D2 t) t := by
    intro t
    rw [hD1, hD2]
    have hRR := (hasDerivAt_rayleigh_line P hPsym x v t).const_mul ((D : ℝ) + 2)
    have hNN := (((hq_deriv t).pow (D / 2)).mul (hQ1_deriv t)).const_mul
      (((D / 2 + 1 : ℕ) : ℝ)) |>.const_mul μ
    convert hRR.add hNN using 1
    funext y
    simp only [Pi.add_apply, Pi.mul_apply, Pi.pow_apply]
    ring
  -- Convexity via the second-derivative test.
  have hderiv1 : deriv f0 = D1 := funext (fun t => (hg1 t).deriv)
  have hderiv2 : deriv D1 = D2 := funext (fun t => (hg2 t).deriv)
  refine convexOn_of_deriv2_nonneg' convex_univ
    (fun t _ => (hg1 t).differentiableAt.differentiableWithinAt) ?_ ?_
  · rw [hderiv1]
    exact fun t _ => (hg2 t).differentiableAt.differentiableWithinAt
  · intro t _
    have h2 : deriv^[2] f0 t = D2 t := by
      rw [Function.iterate_succ, Function.iterate_one, Function.comp_apply,
        hderiv1, hderiv2]
    rw [h2, hD2]
    -- Restrict to the point `w = x + t v` and use the supplied Hessian bound.
    set w := x + t • v with hw
    rw [hP]
    have hsvv : (∑ i, 2 * v i * v i) = 2 * ∑ i, v i ^ 2 := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => by ring)
    rw [hsvv]
    have hSlb0 := hSbound w v
    set V := ∑ i, v i ^ 2 with hV
    set P2 := (∑ i, w i ^ 2) ^ (D / 2) with hP2
    set SS := ∑ j, (contract1 A v ⬝ₜ w) j * v j with hSS
    set DROP := (((D / 2 : ℕ) : ℝ) * (∑ i, w i ^ 2) ^ (D / 2 - 1)
        * (∑ i, 2 * w i * v i)) * (∑ i, 2 * w i * v i) with hDROP
    -- Sign facts.
    have hVnn : 0 ≤ V := by rw [hV]; positivity
    have hP2nn : 0 ≤ P2 := by rw [hP2]; positivity
    have hμ0 : 0 ≤ μ := le_trans (mul_nonneg (by positivity) hC) hCμ
    have hSlb : -(C * P2 * V) ≤ SS := hSlb0
    have hdrop : 0 ≤ DROP := by
      have hEq : DROP = ((D / 2 : ℕ) : ℝ) * (∑ i, w i ^ 2) ^ (D / 2 - 1)
          * (∑ i, 2 * w i * v i) ^ 2 := by rw [hDROP]; ring
      rw [hEq]; positivity
    -- The cast identity `2·(D/2+1) = D+2` (needs `Even D`).
    have hcast : (2 : ℝ) * ((D / 2 + 1 : ℕ) : ℝ) = (D : ℝ) + 2 := by
      have hDD : 2 * (D / 2) = D := Nat.two_mul_div_two_of_even hD
      have hstep : ((D / 2 + 1 : ℕ) : ℝ) = ((D / 2 : ℕ) : ℝ) + 1 := by push_cast; ring
      have h3 : (2 : ℝ) * ((D / 2 : ℕ) : ℝ) = (D : ℝ) := by exact_mod_cast hDD
      rw [hstep]; linear_combination h3
    -- Lower bound the shift term by `μ·(D+2)·P2·V`.
    have h2c : ((D / 2 + 1 : ℕ) : ℝ) * (P2 * (2 * V)) = ((D : ℝ) + 2) * (P2 * V) := by
      rw [show ((D / 2 + 1 : ℕ) : ℝ) * (P2 * (2 * V))
        = (2 * ((D / 2 + 1 : ℕ) : ℝ)) * (P2 * V) from by ring, hcast]
    have hstep1 : μ * (((D : ℝ) + 2) * (P2 * V))
        ≤ μ * (((D / 2 + 1 : ℕ) : ℝ) * (DROP + P2 * (2 * V))) := by
      apply mul_le_mul_of_nonneg_left _ hμ0
      rw [← h2c]
      apply mul_le_mul_of_nonneg_left _ (Nat.cast_nonneg _)
      linarith [hdrop]
    -- Lower bound the Rayleigh term by `-(D+2)(D+1)·C·P2·V`.
    have hstep2 : -(((D : ℝ) + 2) * (((D : ℝ) + 1) * (C * P2 * V)))
        ≤ ((D : ℝ) + 2) * (((D : ℝ) + 1) * SS) := by
      calc -(((D : ℝ) + 2) * (((D : ℝ) + 1) * (C * P2 * V)))
          = ((D : ℝ) + 2) * (((D : ℝ) + 1) * (-(C * P2 * V))) := by ring
        _ ≤ ((D : ℝ) + 2) * (((D : ℝ) + 1) * SS) :=
            mul_le_mul_of_nonneg_left
              (mul_le_mul_of_nonneg_left hSlb (by positivity)) (by positivity)
    -- The combined lower bound is nonnegative.
    have hpos : 0 ≤ -(((D : ℝ) + 2) * (((D : ℝ) + 1) * (C * P2 * V)))
        + μ * (((D : ℝ) + 2) * (P2 * V)) := by
      have hEq : -(((D : ℝ) + 2) * (((D : ℝ) + 1) * (C * P2 * V)))
          + μ * (((D : ℝ) + 2) * (P2 * V))
          = ((D : ℝ) + 2) * P2 * V * (μ - ((D : ℝ) + 1) * C) := by ring
      rw [hEq]
      exact mul_nonneg (by positivity) (by linarith [hCμ])
    linarith [hstep1, hstep2, hpos]

/-- **The practical convexity criterion (Lecture 8, `lem:shift-convex` composed
    with `lem:beta-bound`).**  For a symmetric tensor `A` of even order `D+2`,
    if the shift satisfies `μ ≥ (D+1) · ∑|a|` then the shifted tensor Rayleigh
    quotient is convex on `ℝⁿ`.

    This is the *acceptable fallback* of the two forms of `lem:shift-convex`:
    the faithful statement asks `μ > β(A)`, but `β(A) ≤ (D+1) ∑|a|`
    (`CubicTensor.beta_le`), so the hypothesis here is strictly stronger and the
    conclusion follows.  The proof bounds the Rayleigh Hessian directly by
    `∑|a|` (AM-GM), never invoking the spectral radius. -/
theorem convexOn_shiftedRayleigh_of_sum (A : CubicTensor ℝ (D + 2) N)
    (hA : A.IsSymmetric) (hD : Even D) (μ : ℝ)
    (hμ : ((D : ℝ) + 1) * (∑ idx : Fin (D + 2) → Fin N, |A idx|) ≤ μ) :
    ConvexOn ℝ Set.univ (shiftedRayleigh A μ) := by
  refine convexOn_shiftedRayleigh_of_Sbound A hA hD μ
    (∑ idx : Fin (D + 2) → Fin N, |A idx|) (by positivity) hμ (fun w v => ?_)
  have habs := abs_contract1_apply_le A hD v w
  exact le_trans (neg_le_neg habs) (neg_abs_le _)

/-! ### The sharp spectral criterion `μ > β(A)` -/

/-- **Diagonalization lower bound for the Rayleigh quotient of a Hermitian
    matrix.**  If every eigenvalue of `M` has absolute value at most `ρ`, then the
    quadratic form is bounded below by `−ρ · ‖v‖²`.  Proved by expanding `v` in the
    orthonormal eigenbasis: `vᵀ M v = ∑ᵢ λᵢ cᵢ²` and `‖v‖² = ∑ᵢ cᵢ²` with
    `cᵢ = ⟪bᵢ, v⟫`, then `∑ λᵢ cᵢ² ≥ ∑ (−ρ) cᵢ² = −ρ ‖v‖²`. -/
theorem neg_le_dotProduct_mulVec (M : Matrix (Fin N) (Fin N) ℝ) (hM : M.IsHermitian)
    (ρ : ℝ) (hρ : ∀ i, |hM.eigenvalues i| ≤ ρ) (v : Fin N → ℝ) :
    -(ρ * (∑ i, v i ^ 2)) ≤ v ⬝ᵥ (M *ᵥ v) := by
  classical
  -- Unitary matrix of eigenvectors `U` and coordinates `z = star U ⬝ᵥ v`.
  set U : Matrix (Fin N) (Fin N) ℝ := (hM.eigenvectorUnitary : Matrix (Fin N) (Fin N) ℝ) with hU
  set z : Fin N → ℝ := (star U) *ᵥ v with hz
  have hmem := (hM.eigenvectorUnitary).2
  have hUUs : U * star U = 1 := by rw [hU]; exact Matrix.mem_unitaryGroup_iff.mp hmem
  -- `v ᵥ* U = z` since over `ℝ` we have `star U = Uᵀ`.
  have hvzU : v ᵥ* U = z := by
    rw [hz, Matrix.star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial,
      Matrix.mulVec_transpose]
  -- Spectral decomposition `M = U · diag(λ) · star U`.
  have hMspec : M = U * diagonal hM.eigenvalues * star U := by
    have hst := hM.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply,
      show (RCLike.ofReal ∘ hM.eigenvalues) = hM.eigenvalues from by
        funext i; simp [RCLike.ofReal_real_eq_id]] at hst
    rw [hU]; exact hst
  -- Diagonalize the quadratic form: `vᵀ M v = ∑ᵢ λᵢ zᵢ²`.
  have hSval : v ⬝ᵥ (M *ᵥ v) = ∑ i, hM.eigenvalues i * (z i) ^ 2 := by
    conv_lhs => rw [hMspec]
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, ← hz,
      Matrix.dotProduct_mulVec, hvzU]
    simp only [dotProduct]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Matrix.mulVec_diagonal]
    ring
  -- Orthogonality: `‖z‖² = ‖v‖²`.
  have hzz : z ⬝ᵥ z = v ⬝ᵥ v := by
    have hthis : z ⬝ᵥ z = (v ᵥ* U) ⬝ᵥ (star U *ᵥ v) := by rw [hvzU, ← hz]
    rw [hthis, ← Matrix.dotProduct_mulVec, Matrix.mulVec_mulVec, hUUs, Matrix.one_mulVec]
  have hVc : (∑ i, v i ^ 2) = ∑ i, (z i) ^ 2 := by
    have hvv : ∀ a : Fin N → ℝ, (∑ i, (a i) ^ 2) = a ⬝ᵥ a := fun a => by
      simp [dotProduct, pow_two]
    rw [hvv v, hvv z, hzz]
  -- Termwise: `−ρ · zᵢ² ≤ λᵢ · zᵢ²` since `λᵢ ≥ −ρ`.
  have key : ∀ i, -ρ * (z i) ^ 2 ≤ hM.eigenvalues i * (z i) ^ 2 := fun i =>
    mul_le_mul_of_nonneg_right (abs_le.mp (hρ i)).1 (sq_nonneg _)
  rw [hSval, hVc, ← neg_mul, Finset.mul_sum]
  exact Finset.sum_le_sum (fun i _ => key i)

/-- **The Rayleigh Hessian slope as a matrix quadratic form.**  The directional
    slope `∑ⱼ (contract1 A v ⬝ₜ w)ⱼ vⱼ` is exactly `vᵀ (applyMatrix A w) v`. -/
theorem sum_contract1_apply_eq_dotProduct (A : CubicTensor ℝ (D + 2) N) (v w : Fin N → ℝ) :
    (∑ j, (contract1 A v ⬝ₜ w) j * v j) = v ⬝ᵥ (applyMatrix A w *ᵥ v) := by
  have hL : (∑ j, (contract1 A v ⬝ₜ w) j * v j)
      = ∑ j, ∑ rest : Fin D → Fin N, ∑ i,
          v i * A (Fin.cons i (Fin.cons j rest)) * (∏ d, w (rest d)) * v j := by
    simp only [apply_apply, contract1, Finset.sum_mul]
  have hR : v ⬝ᵥ (applyMatrix A w *ᵥ v)
      = ∑ j, ∑ rest : Fin D → Fin N, ∑ i,
          v i * A (Fin.cons i (Fin.cons j rest)) * (∏ d, w (rest d)) * v j := by
    simp only [dotProduct, Matrix.mulVec, applyMatrix, Finset.mul_sum, Finset.sum_mul]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl (fun rest _ => Finset.sum_congr rfl (fun i _ => by ring))
  rw [hL, hR]

/-- **Homogeneity of the contracted matrix.**  `applyMatrix A (c • w) = cᴰ • applyMatrix A w`. -/
theorem applyMatrix_smul_v (A : CubicTensor ℝ (D + 2) N) (c : ℝ) (w : Fin N → ℝ) :
    applyMatrix A (c • w) = c ^ D • applyMatrix A w := by
  ext i j
  simp only [applyMatrix, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  have hprod : (∏ d, (c • w) (rest d)) = c ^ D * ∏ d, w (rest d) := by
    simp only [Pi.smul_apply, smul_eq_mul]
    rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [hprod]; ring

/-- **Each eigenvalue of a Hermitian matrix is bounded by its spectral radius.** -/
theorem abs_eigenvalue_le_spectralRadius {n : Type*} [Fintype n] [DecidableEq n]
    (M : Matrix n n ℝ) (hM : M.IsHermitian) (hfin : spectralRadius ℝ M ≠ ⊤) (i : n) :
    |hM.eigenvalues i| ≤ (spectralRadius ℝ M).toReal := by
  have hmem : hM.eigenvalues i ∈ spectrum ℝ M := hM.eigenvalues_mem_spectrum_real i
  have hle : (‖hM.eigenvalues i‖₊ : ℝ≥0∞) ≤ spectralRadius ℝ M :=
    le_iSup₂ (f := fun k (_ : k ∈ spectrum ℝ M) => (‖k‖₊ : ℝ≥0∞)) (hM.eigenvalues i) hmem
  have h2 := ENNReal.toReal_mono hfin hle
  rwa [ENNReal.coe_toReal, coe_nnnorm, Real.norm_eq_abs] at h2

/-- **Normalisation for the homogeneity argument.**  For `N > 0` and any `w`, there is a
    unit vector `p` (on the sphere `∑ pᵢ² = 1`) with
    `applyMatrix A w = (∑ wᵢ²)^{D/2} • applyMatrix A p`.  If `w ≠ 0` take `p = w/‖w‖`;
    if `w = 0` any standard basis vector works (the scalar `(∑wᵢ²)^{D/2}` then vanishes
    for `D ≥ 2` and is `1` for the constant `D = 0` case). -/
theorem exists_unit_smul_applyMatrix (A : CubicTensor ℝ (D + 2) N) (hD : Even D) (hN : 0 < N)
    (w : Fin N → ℝ) :
    ∃ p : Fin N → ℝ, (∑ i, p i ^ 2 = 1) ∧
      applyMatrix A w = (∑ i, w i ^ 2) ^ (D / 2) • applyMatrix A p := by
  set Q : ℝ := ∑ i, w i ^ 2 with hQ
  have hQnn : 0 ≤ Q := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  -- A fixed unit vector for the degenerate case.
  set p₀ : Fin N → ℝ := Pi.single (⟨0, hN⟩ : Fin N) 1 with hp₀
  have hp₀unit : (∑ i, p₀ i ^ 2) = 1 := by
    rw [Finset.sum_eq_single (⟨0, hN⟩ : Fin N)
      (fun b _ hb => by simp [hp₀, Pi.single_eq_of_ne hb])
      (fun h => absurd (Finset.mem_univ _) h)]
    simp [hp₀]
  rcases eq_or_lt_of_le hQnn with hQ0 | hQpos
  · -- `Q = 0`, so `w = 0`.
    have hw0 : w = 0 := by
      funext i
      have hsum0 : Q = 0 := hQ0.symm
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun l _ => sq_nonneg (w l))).mp hsum0 i
        (Finset.mem_univ i)
      exact pow_eq_zero_iff (by norm_num) |>.mp this
    refine ⟨p₀, hp₀unit, ?_⟩
    have hzz : (0 : ℝ) ^ D = (0 : ℝ) ^ (D / 2) := by
      rcases Nat.eq_zero_or_pos D with hD0 | hDpos
      · simp [hD0]
      · obtain ⟨k, hk⟩ := hD
        rw [zero_pow (by omega), zero_pow (by omega)]
    rw [hw0, show (0 : Fin N → ℝ) = (0 : ℝ) • p₀ from by rw [zero_smul],
      applyMatrix_smul_v, hzz, ← hQ0]
  · -- `Q > 0`: normalise `w`.
    refine ⟨(Real.sqrt Q)⁻¹ • w, ?_, ?_⟩
    · simp only [Pi.smul_apply, smul_eq_mul, mul_pow]
      rw [← Finset.mul_sum, ← hQ, inv_pow, Real.sq_sqrt hQnn]
      exact inv_mul_cancel₀ hQpos.ne'
    · rw [applyMatrix_smul_v, smul_smul]
      have hsqrtD : (Real.sqrt Q) ^ D = Q ^ (D / 2) := by
        have hDD : 2 * (D / 2) = D := Nat.two_mul_div_two_of_even hD
        conv_lhs => rw [← hDD]
        rw [pow_mul, Real.sq_sqrt hQnn]
      rw [inv_pow, hsqrtD, mul_inv_cancel₀ (pow_ne_zero _ hQpos.ne'), one_smul]

/-- **The sharp form of `lem:shift-convex`.**  For a symmetric tensor `A` of even
    order `D+2`, if the shift strictly exceeds the convergence threshold
    `β(A)` then the shifted tensor Rayleigh quotient is convex on `ℝⁿ`.  This is
    the faithful statement of the lecture lemma; the spectral radius enters
    through `β(A)`. -/
theorem convexOn_shiftedRayleigh (A : CubicTensor ℝ (D + 2) N) (hA : A.IsSymmetric)
    (hD : Even D) (μ : ℝ) (hμ : beta A < ENNReal.ofReal μ) :
    ConvexOn ℝ Set.univ (shiftedRayleigh A μ) := by
  have hbeta_top : beta A ≠ ⊤ := (hμ.trans ENNReal.ofReal_lt_top).ne
  have hD1 : (0 : ℝ) < (D : ℝ) + 1 := by positivity
  have hbeta_toReal_le : (beta A).toReal ≤ μ := by
    have := (ENNReal.toReal_lt_of_lt_ofReal hμ).le
    exact this
  refine convexOn_shiftedRayleigh_of_Sbound A hA hD μ
    ((beta A).toReal / ((D : ℝ) + 1)) (div_nonneg ENNReal.toReal_nonneg hD1.le) ?_ ?_
  · rw [mul_div_cancel₀ _ hD1.ne']
    exact hbeta_toReal_le
  · intro w v
    -- Rewrite the slope `S` as a matrix quadratic form `vᵀ (applyMatrix A w) v`.
    rw [sum_contract1_apply_eq_dotProduct A v w]
    rcases Nat.eq_zero_or_pos N with hN0 | hN
    · subst hN0; simp
    -- Choose a unit vector `p` with `applyMatrix A w = Q • applyMatrix A p`.
    obtain ⟨p, hpunit, hpeq⟩ := exists_unit_smul_applyMatrix A hD hN w
    set Q : ℝ := (∑ i, w i ^ 2) ^ (D / 2) with hQ
    set V : ℝ := ∑ i, v i ^ 2 with hV
    set Mp : Matrix (Fin N) (Fin N) ℝ := applyMatrix A p with hMp
    have hQnn : 0 ≤ Q := by rw [hQ]; positivity
    have hVnn : 0 ≤ V := by rw [hV]; positivity
    have hMpH : Mp.IsHermitian := applyMatrix_isHermitian A hA p
    -- The supremum defining `β(A)`.
    set Sβ : ℝ≥0∞ := ⨆ x : {x : Fin N → ℝ // ∑ i, x i ^ 2 = 1},
        spectralRadius ℝ (applyMatrix A x.1) with hSβ
    have hbeta_eq : beta A = ((D : ℝ≥0∞) + 1) * Sβ := by rw [hSβ]; rfl
    have hDℝ0 : ((D : ℝ≥0∞) + 1) ≠ 0 := by positivity
    have hSβ_top : Sβ ≠ ⊤ :=
      (ENNReal.lt_top_of_mul_ne_top_right (hbeta_eq ▸ hbeta_top) hDℝ0).ne
    -- `spectralRadius (applyMatrix A p) ≤ Sβ`, hence finite.
    have hMp_le : spectralRadius ℝ Mp ≤ Sβ := by
      rw [hSβ]
      exact le_iSup (fun x : {x : Fin N → ℝ // ∑ i, x i ^ 2 = 1} =>
        spectralRadius ℝ (applyMatrix A x.1)) ⟨p, hpunit⟩
    have hMp_top : spectralRadius ℝ Mp ≠ ⊤ :=
      (hMp_le.trans_lt (lt_top_iff_ne_top.mpr hSβ_top)).ne
    -- `ρp := ρ(applyMatrix A p) ≤ β(A).toReal / (D+1)`.
    set ρp : ℝ := (spectralRadius ℝ Mp).toReal with hρp
    have hρp_le : ρp ≤ (beta A).toReal / ((D : ℝ) + 1) := by
      have h1 : ρp ≤ Sβ.toReal := ENNReal.toReal_mono hSβ_top hMp_le
      have h2 : Sβ.toReal = (beta A).toReal / ((D : ℝ) + 1) := by
        rw [hbeta_eq, ENNReal.toReal_mul,
          show ((D : ℝ≥0∞) + 1).toReal = (D : ℝ) + 1 from by
            rw [ENNReal.toReal_add (by simp) (by simp), ENNReal.toReal_natCast,
              ENNReal.toReal_one],
          mul_comm, mul_div_assoc, div_self hD1.ne', mul_one]
      rwa [h2] at h1
    -- Diagonalization lower bound on the Rayleigh quotient of `applyMatrix A p`.
    have hDiag := neg_le_dotProduct_mulVec Mp hMpH ρp
      (fun i => abs_eigenvalue_le_spectralRadius Mp hMpH hMp_top i) v
    -- `vᵀ (applyMatrix A w) v = Q · vᵀ (applyMatrix A p) v`.
    have hSscale : v ⬝ᵥ (applyMatrix A w *ᵥ v) = Q * (v ⬝ᵥ (Mp *ᵥ v)) := by
      rw [hpeq, Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]
    rw [hSscale]
    have hscaled : Q * (-(ρp * V)) ≤ Q * (v ⬝ᵥ (Mp *ᵥ v)) :=
      mul_le_mul_of_nonneg_left hDiag hQnn
    have hlb : -((beta A).toReal / ((D : ℝ) + 1) * Q * V) ≤ Q * (-(ρp * V)) := by
      nlinarith [mul_nonneg (mul_nonneg hQnn hVnn) (sub_nonneg.mpr hρp_le)]
    linarith [hscaled, hlb]

end CubicTensor
