/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Eigenpair
import TensorCourse.Tensor.Transpose
import Mathlib.Analysis.Calculus.FDeriv.Pow
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.FDeriv.Congr
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# The ℓp-Variational Characterization of Tensor Eigenvectors (Lecture 8)

This file formalises the variational (Lagrange-multiplier) characterisation of
`ℓp`-eigenvectors from Lecture 8. Working over the reals, for a **symmetric**
cubic order-`(D+1)` tensor `A ∈ ℝ^{N×⋯×N}`, the `ℓp`-eigenvectors of `A` are
exactly the *critical points of the tensor Rayleigh quotient*
`R_A(x) = xᵀ (A ⬝ₜ x)` subject to the normalisation `∑ᵢ xᵢᵖ = 1`.

The Lagrange-multiplier condition is expressed through the Fréchet derivative
`fderiv ℝ` on the finite-dimensional space `Fin N → ℝ`: `x` is a *constrained
critical point* if the constraint holds and
`fderiv (R_A) x = μ • fderiv (∑ᵢ ·ᵢᵖ) x` for some multiplier `μ`.

Because a cubic tensor of Lean-order `D+1` models the lecture's order-`D`
tensor, the paper's gradient formula `∇R_A = D · (A x)` becomes
`∇R_A = (D+1) · (A ⬝ₜ x)` here, and `∇(∑ᵢ ·ᵢᵖ) = p · x^{[p-1]}`.

## Main definitions

* `CubicTensor.bilinFormCLM w` — the continuous linear functional
  `v ↦ ∑ᵢ wᵢ vᵢ`; both gradients are of this form.
* `CubicTensor.rayleighConstraint p x = ∑ᵢ xᵢᵖ` — the (unrooted) `ℓp`
  normalisation used as the constraint.
* `CubicTensor.IsConstrainedCriticalPoint p A x` — the Lagrange-multiplier
  predicate.

## Main results

* `CubicTensor.hasFDerivAt_rayleighConstraint` — `∇(∑ᵢ ·ᵢᵖ) = p • x^{[p-1]}`.
* `CubicTensor.hasFDerivAt_rayleigh` — `∇R_A = (D+1) • (A ⬝ₜ x)` for symmetric `A`
  (the tedious symmetry-collapse of the derivative of the multilinear form).
* `CubicTensor.isLpEigenpair_iff_isConstrainedCriticalPoint` — the variational
  characterisation: for even `p ≥ 2` and symmetric `A`, `x` carries an
  `ℓp`-eigenpair iff it is a constrained critical point.
-/

noncomputable section

open Finset BigOperators
open scoped CubicTensor

namespace CubicTensor

variable {N D : ℕ}
variable {A : CubicTensor ℝ (D + 1) N}

/-! ### The gradient functional `bilinFormCLM` -/

/-- The `i`-th coordinate functional on `Fin N → ℝ`, as a continuous linear map.
    Bundling it as a `def` with a concrete Pi family avoids the higher-order
    metavariable issues that arise when `ContinuousLinearMap.proj`'s implicit
    family has to be inferred from the codomain during instance search. -/
def coord (i : Fin N) : (Fin N → ℝ) →L[ℝ] ℝ := ContinuousLinearMap.proj i

@[simp]
theorem coord_apply (i : Fin N) (v : Fin N → ℝ) : coord i v = v i := rfl

/-- The coordinate functional is its own Fréchet derivative (it is linear). -/
theorem coord_hasFDerivAt (i : Fin N) (x : Fin N → ℝ) :
    HasFDerivAt (fun y : Fin N → ℝ => y i) (coord i) x := (coord i).hasFDerivAt

/-- The continuous linear functional `v ↦ ∑ᵢ wᵢ vᵢ` on `Fin N → ℝ`.
    Both the Rayleigh-quotient gradient and the constraint gradient are of this
    form, so the Lagrange condition reduces to an equality of such functionals. -/
def bilinFormCLM (w : Fin N → ℝ) : (Fin N → ℝ) →L[ℝ] ℝ :=
  ∑ i : Fin N, w i • coord i

@[simp]
theorem bilinFormCLM_apply (w v : Fin N → ℝ) :
    bilinFormCLM w v = ∑ i, w i * v i := by
  simp only [bilinFormCLM, ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
    coord_apply, smul_eq_mul]

/-- `bilinFormCLM w` picks out the `j`-th coordinate `wⱼ` on the standard basis
    vector `Pi.single j 1`. This is used to extract the pointwise eigenvector
    equation from the equality of gradients. -/
theorem bilinFormCLM_single (w : Fin N → ℝ) (j : Fin N) :
    bilinFormCLM w (Pi.single j 1) = w j := by
  rw [bilinFormCLM_apply,
    Finset.sum_eq_single j (fun i _ hij => by rw [Pi.single_eq_of_ne hij, mul_zero])
      (fun h => absurd (Finset.mem_univ j) h)]
  rw [Pi.single_eq_same, mul_one]

/-- `bilinFormCLM` is linear in its coefficient vector:
    `bilinFormCLM (c • w) = c • bilinFormCLM w`. -/
theorem bilinFormCLM_smul (c : ℝ) (w : Fin N → ℝ) :
    bilinFormCLM (c • w) = c • bilinFormCLM w := by
  ext v
  simp only [bilinFormCLM_apply, ContinuousLinearMap.smul_apply, Pi.smul_apply, smul_eq_mul,
    Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-! ### Symmetry helpers

We re-prove locally the fact (from `Tensor/Diagonal.lean`, where it is `private`)
that on a *cubic* tensor the transposition cast is trivial. -/

/-- On a cubic tensor, `(T.transpose σ) idx = T (idx ∘ σ.symm)`; the `▸` cast in
    `Tensor.transpose` is along a constant motive, hence the identity. -/
private theorem cubic_transpose_apply {D N : ℕ} (σ : Equiv.Perm (Fin D))
    (T : CubicTensor ℝ D N) (idx : (d : Fin D) → Fin N) :
    T.transpose σ idx = T (fun d => idx (σ.symm d)) := by
  simp only [Tensor.transpose_apply]
  congr 1
  funext d
  apply eq_of_heq
  exact eqRec_heq_iff_heq.mpr HEq.rfl

/-- Invariance of a symmetric tensor under precomposition of its multi-index by a
    permutation: `A (idx ∘ σ) = A idx`. -/
theorem apply_perm (hA : A.IsSymmetric) (σ : Equiv.Perm (Fin (D + 1)))
    (idx : Fin (D + 1) → Fin N) :
    A (fun k => idx (σ k)) = A idx := by
  have h : A.transpose σ.symm idx = A idx := by rw [hA σ.symm]
  rw [cubic_transpose_apply] at h
  simpa only [Equiv.symm_symm] using h

/-- **Insertion vs. cons.** For symmetric `A`, inserting the value `a` at an
    arbitrary slot `d` gives the same tensor entry as consing it at the front:
    `A (insertNth d a rest) = A (cons a rest)`. This is the only place the full
    permutation symmetry of `A` is used in the Rayleigh-gradient computation. -/
theorem apply_insertNth (hA : A.IsSymmetric) (d : Fin (D + 1)) (a : Fin N)
    (rest : Fin D → Fin N) :
    A (Fin.insertNth d a rest) = A (Fin.cons a rest) := by
  have hρd : ((finSuccEquiv' d).trans (finSuccEquiv' (0 : Fin (D + 1))).symm) d = 0 := by
    change (finSuccEquiv' (0 : Fin (D + 1))).symm (finSuccEquiv' d d) = 0
    rw [finSuccEquiv'_at, finSuccEquiv'_symm_none]
  have hρs : ∀ k : Fin D,
      ((finSuccEquiv' d).trans (finSuccEquiv' (0 : Fin (D + 1))).symm) (d.succAbove k)
        = Fin.succ k := by
    intro k
    change (finSuccEquiv' (0 : Fin (D + 1))).symm (finSuccEquiv' d (d.succAbove k)) = Fin.succ k
    rw [finSuccEquiv'_succAbove, finSuccEquiv'_symm_some, Fin.succAbove_zero]
  have hfun : ∀ e, (Fin.insertNth d a rest : Fin (D + 1) → Fin N) e
      = (Fin.cons a rest : Fin (D + 1) → Fin N)
        (((finSuccEquiv' d).trans (finSuccEquiv' (0 : Fin (D + 1))).symm) e) := by
    rw [Fin.forall_iff_succAbove d]
    refine ⟨?_, ?_⟩
    · rw [Fin.insertNth_apply_same, hρd, Fin.cons_zero]
    · intro k
      rw [Fin.insertNth_apply_succAbove, hρs, Fin.cons_succ]
  calc A (Fin.insertNth d a rest)
      = A (fun e => (Fin.cons a rest : Fin (D + 1) → Fin N)
          (((finSuccEquiv' d).trans (finSuccEquiv' (0 : Fin (D + 1))).symm) e)) := by
        congr 1; funext e; exact hfun e
    _ = A (Fin.cons a rest) :=
        apply_perm hA ((finSuccEquiv' d).trans (finSuccEquiv' (0 : Fin (D + 1))).symm)
          (Fin.cons a rest)

/-! ### The gradient of the constraint `∑ᵢ xᵢᵖ` -/

/-- The (unrooted) `ℓp` normalisation used as the constraint: `∑ᵢ xᵢᵖ`. On the
    admissible set `∑ᵢ xᵢᵖ = 1` this equals `‖x‖ₚᵖ` because `p` is even. -/
def rayleighConstraint (p : ℕ) (x : Fin N → ℝ) : ℝ := ∑ i, x i ^ p

/-- **Gradient of the constraint.** `∇(∑ᵢ ·ᵢᵖ)(x) = p • x^{[p-1]}`, expressed as
    the functional `p • bilinFormCLM (fun i => xᵢ^{p-1})`. -/
theorem hasFDerivAt_rayleighConstraint (p : ℕ) (x : Fin N → ℝ) :
    HasFDerivAt (rayleighConstraint p)
      ((p : ℝ) • bilinFormCLM (fun i => x i ^ (p - 1))) x := by
  have hterm : ∀ i ∈ (Finset.univ : Finset (Fin N)),
      HasFDerivAt (fun y : Fin N → ℝ => y i ^ p)
        ((p • x i ^ (p - 1)) • coord i) x :=
    fun i _ => (coord_hasFDerivAt i x).pow p
  have hsum := HasFDerivAt.fun_sum hterm
  refine hsum.congr_fderiv ?_
  ext v
  simp only [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
    coord_apply, bilinFormCLM_apply, smul_eq_mul, nsmul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  ring

/-! ### The gradient of the Rayleigh quotient -/

/-- **Bridging product reindex.** A product over `univ.erase d` re-expressed as a
    product over `Fin D` via `d.succAbove`. -/
theorem prod_erase_eq_prod_succAbove (g : Fin (D + 1) → ℝ) (d : Fin (D + 1)) :
    ∏ d' ∈ Finset.univ.erase d, g d' = ∏ k : Fin D, g (d.succAbove k) := by
  symm
  refine Finset.prod_bij (fun k _ => d.succAbove k) ?_ ?_ ?_ ?_
  · intro k _
    exact Finset.mem_erase.mpr ⟨d.succAbove_ne k, Finset.mem_univ _⟩
  · intro k₁ _ k₂ _ h
    exact Fin.succAbove_right_injective h
  · intro d' hd'
    obtain ⟨k, hk⟩ := Fin.exists_succAbove_eq (Finset.mem_erase.mp hd').1
    exact ⟨k, Finset.mem_univ _, hk⟩
  · intro k _
    rfl

/-- **Symmetry collapse.** For a fixed slot `d`, the `d`-th "leave-one-out"
    contraction of the derivative of the multilinear form collapses (using the
    symmetry of `A`) to the single contraction `∑ⱼ (A ⬝ₜ x)ⱼ vⱼ`. Summing over the
    `D+1` slots is what produces the factor `D+1` in the Rayleigh gradient. -/
theorem rayleigh_deriv_collapse (hA : A.IsSymmetric) (x v : Fin N → ℝ)
    (d : Fin (D + 1)) :
    ∑ idx : Fin (D + 1) → Fin N,
        A idx * (∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d)
      = ∑ j : Fin N, (A ⬝ₜ x) j * v j := by
  rw [← Equiv.sum_comp (Fin.insertNthEquiv (fun _ : Fin (D + 1) => Fin N) d)
        (fun idx => A idx * (∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d)),
      Fintype.sum_prod_type]
  simp only [apply_apply, Finset.sum_mul]
  refine Finset.sum_congr rfl (fun a _ => Finset.sum_congr rfl (fun rest _ => ?_))
  change A (Fin.insertNth d a rest : Fin (D + 1) → Fin N)
        * (∏ d' ∈ Finset.univ.erase d, x ((Fin.insertNth d a rest : Fin (D + 1) → Fin N) d'))
        * v ((Fin.insertNth d a rest : Fin (D + 1) → Fin N) d)
      = A (Fin.cons a rest) * (∏ dd : Fin D, x (rest dd)) * v a
  rw [Fin.insertNth_apply_same, apply_insertNth hA, prod_erase_eq_prod_succAbove]
  simp only [Fin.insertNth_apply_succAbove]

set_option maxHeartbeats 800000 in
-- The symmetry-collapse of the degree-`(D+1)` multilinear derivative ranges over the
-- large index set `Fin (D+1) → Fin N`; the reindexing/defeq bookkeeping needs extra heartbeats.
/-- **Gradient of the Rayleigh quotient (Lecture 8).** For symmetric `A`,
    `∇R_A(x) = (D+1) • (A ⬝ₜ x)`, expressed as `(D+1) • bilinFormCLM (A ⬝ₜ x)`.
    (In the lecture's order-`D` numbering this is the formula `∇R_A = D · A x`.) -/
theorem hasFDerivAt_rayleigh (hA : A.IsSymmetric) (x : Fin N → ℝ) :
    HasFDerivAt (rayleigh A) (((D : ℝ) + 1) • bilinFormCLM (A ⬝ₜ x)) x := by
  -- Raw derivative of each monomial `∏_d y (idx d)`.
  have hprod : ∀ idx : Fin (D + 1) → Fin N,
      HasFDerivAt (fun y : Fin N → ℝ => ∏ d : Fin (D + 1), y (idx d))
        (∑ d : Fin (D + 1),
          (∏ d' ∈ Finset.univ.erase d, x (idx d')) • coord (idx d)) x := by
    intro idx
    exact HasFDerivAt.finset_prod (𝔸' := ℝ) (u := (Finset.univ : Finset (Fin (D + 1))))
      (g := fun d y => y (idx d)) (g' := fun d => coord (idx d))
      (fun d _ => coord_hasFDerivAt (idx d) x)
  -- Multiply by the constant `A idx` and sum over all multi-indices.
  have hsum := HasFDerivAt.fun_sum (u := (Finset.univ : Finset (Fin (D + 1) → Fin N)))
    (fun idx _ => (hprod idx).const_mul (A idx))
  have hfun : (fun y : Fin N → ℝ => ∑ idx : Fin (D + 1) → Fin N,
      A idx * ∏ d : Fin (D + 1), y (idx d)) = rayleigh A := by
    funext y; exact (rayleigh_eq_sum A y).symm
  rw [← hfun]
  refine hsum.congr_fderiv ?_
  ext v
  simp only [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
    coord_apply, smul_eq_mul, bilinFormCLM_apply]
  have collapse : ∀ d : Fin (D + 1),
      ∑ idx : Fin (D + 1) → Fin N,
          A idx * ((∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d))
        = ∑ j, (A ⬝ₜ x) j * v j := by
    intro d
    rw [← rayleigh_deriv_collapse hA x v d]
    exact Finset.sum_congr rfl (fun idx _ => (mul_assoc _ _ _).symm)
  calc ∑ idx : Fin (D + 1) → Fin N, A idx * ∑ d : Fin (D + 1),
          (∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d)
      = ∑ idx : Fin (D + 1) → Fin N, ∑ d : Fin (D + 1),
          A idx * ((∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d)) := by
        exact Finset.sum_congr rfl (fun idx _ => Finset.mul_sum _ _ (A idx))
    _ = ∑ d : Fin (D + 1), ∑ idx : Fin (D + 1) → Fin N,
          A idx * ((∏ d' ∈ Finset.univ.erase d, x (idx d')) * v (idx d)) := Finset.sum_comm
    _ = ∑ _d : Fin (D + 1), ∑ j, (A ⬝ₜ x) j * v j :=
        Finset.sum_congr rfl (fun d _ => collapse d)
    _ = ((D : ℝ) + 1) * ∑ j, (A ⬝ₜ x) j * v j := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        push_cast; ring

/-! ### The variational characterisation -/

/-- **Constrained critical point (Lagrange condition).** `x` is a constrained
    critical point of the tensor Rayleigh quotient for the `ℓp` normalisation if
    `∑ᵢ xᵢᵖ = 1` and the gradient of `R_A` is a multiple of the gradient of the
    constraint at `x`. -/
def IsConstrainedCriticalPoint (p : ℕ) (A : CubicTensor ℝ (D + 1) N)
    (x : Fin N → ℝ) : Prop :=
  (∑ i, x i ^ p = 1) ∧
    ∃ μ : ℝ, fderiv ℝ (rayleigh A) x = μ • fderiv ℝ (rayleighConstraint p) x

/-- **The ℓp-variational characterisation (Lecture 8).** For an even `p ≥ 2` and a
    symmetric cubic tensor `A`, the `ℓp`-eigenvectors of `A` are exactly the
    critical points of the tensor Rayleigh quotient subject to `∑ᵢ xᵢᵖ = 1`. -/
theorem isLpEigenpair_iff_isConstrainedCriticalPoint
    (hA : A.IsSymmetric) (hp : Even p) (hp2 : 2 ≤ p) (x : Fin N → ℝ) :
    (∃ lam : ℝ, IsLpEigenpair p A lam x) ↔ IsConstrainedCriticalPoint p A x := by
  have hpow : ∀ i, ‖x i‖ ^ p = x i ^ p := fun i => by
    rw [Real.norm_eq_abs]; exact hp.pow_abs (x i)
  have hp0 : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hD1 : ((D : ℝ) + 1) ≠ 0 := by positivity
  constructor
  · -- eigenpair ⇒ constrained critical point
    rintro ⟨lam, heq, hnorm⟩
    refine ⟨?_, ((D : ℝ) + 1) * lam / (p : ℝ), ?_⟩
    · rw [← hnorm]; exact Finset.sum_congr rfl (fun i _ => (hpow i).symm)
    · have hscalar : ((D : ℝ) + 1) * lam = ((D : ℝ) + 1) * lam / (p : ℝ) * (p : ℝ) := by
        field_simp
      rw [(hasFDerivAt_rayleigh hA x).fderiv, (hasFDerivAt_rayleighConstraint p x).fderiv, heq,
        bilinFormCLM_smul, smul_smul, smul_smul, ← hscalar]
  · -- constrained critical point ⇒ eigenpair
    rintro ⟨hcon, μ, hderiv⟩
    rw [(hasFDerivAt_rayleigh hA x).fderiv, (hasFDerivAt_rayleighConstraint p x).fderiv,
      smul_smul] at hderiv
    refine ⟨μ * (p : ℝ) / ((D : ℝ) + 1), ?_, ?_⟩
    · funext j
      have hj := DFunLike.congr_fun hderiv (Pi.single j 1)
      simp only [ContinuousLinearMap.smul_apply, bilinFormCLM_single, smul_eq_mul] at hj
      rw [Pi.smul_apply, smul_eq_mul]
      field_simp
      linear_combination hj
    · rw [← hcon]; exact Finset.sum_congr rfl (fun i _ => hpow i)

end CubicTensor
