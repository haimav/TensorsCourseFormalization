/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Rayleigh
import TensorCourse.Tensor.FrobeniusNorm

/-!
# Best Symmetric Rank-1 Approximation (Lecture 8)

Lecture 8 studies the best symmetric rank-1 approximation of a tensor `X`,
`min_{γ,x} ‖X - γ · x ◯ ⋯ ◯ x‖_F` with `‖x‖₂ = 1`, and reduces it to the
tensor Rayleigh quotient. This file formalises the two theorems of that
reduction:

* **Optimal `γ` (thm:rank1-gamma).** For a fixed unit vector `x`, the coefficient
  minimising `‖X - γ · u‖_F` (where `u = x ◯ ⋯ ◯ x`) is `γ⋆ = ⟪u, X⟫`, which is
  the tensor Rayleigh quotient `R_X(x)` (they coincide over `ℝ`).
* **Dual equivalence (thm:rank1-dual).** At that optimal `γ`, the residual satisfies
  `‖X - γ⋆ u‖_F² = ‖X‖_F² - ‖γ⋆‖²`, so *minimising* the rank-1 approximation error
  is equivalent to *maximising* `‖R_X(x)‖` — the largest-magnitude Z-eigenvalue.

Both are pure orthogonal-projection facts: `u` is a unit tensor (`⟪u,u⟫ = 1`), and
the residual `X - ⟪u,X⟫ u` is orthogonal to `u`, so the results follow from the
Pythagorean identity `Tensor.frobNormSq_add_of_orthogonal`. No symmetry of `X`
is needed. The rank-1 tensor is the pointwise product `u(idx) = ∏_d x(idx d)`,
i.e. `x ◯ x ◯ ⋯ ◯ x`.

## Main definitions

* `CubicTensor.symRankOne x` — the symmetric rank-1 tensor `x ◯ ⋯ ◯ x`.

## Main results

* `CubicTensor.frobNormSq_sub_symRankOne_min` — optimality of `γ⋆ = ⟪u, X⟫`.
* `CubicTensor.frobNormSq_residual_eq` — the dual identity `‖X - γ⋆ u‖² = ‖X‖² - ‖γ⋆‖²`.
-/

noncomputable section

open Finset BigOperators Tensor

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ} {x : Fin N → 𝕜}

/-- The **symmetric rank-1 tensor** `x ◯ x ◯ ⋯ ◯ x` (`D+1` copies of `x`),
    as the pointwise product `u(idx) = ∏_d x(idx d)`. -/
def symRankOne (x : Fin N → 𝕜) : CubicTensor 𝕜 (D + 1) N :=
  fun idx => ∏ d : Fin (D + 1), x (idx d)

theorem symRankOne_apply (x : Fin N → 𝕜) (idx : Fin (D + 1) → Fin N) :
    (symRankOne x : CubicTensor 𝕜 (D + 1) N) idx = ∏ d : Fin (D + 1), x (idx d) := rfl

/-- The squared Frobenius norm of `x ◯ ⋯ ◯ x` is `(∑ᵢ ‖xᵢ‖²)^{D+1} = ‖x‖₂^{2(D+1)}`. -/
theorem frobNormSq_symRankOne (x : Fin N → 𝕜) :
    frobNormSq (symRankOne x : CubicTensor 𝕜 (D + 1) N) = (∑ i, ‖x i‖ ^ 2) ^ (D + 1) := by
  rw [frobNormSq_def]
  have h : ∀ idx : Fin (D + 1) → Fin N,
      ‖(symRankOne x : CubicTensor 𝕜 (D + 1) N) idx‖ ^ 2 = ∏ d, ‖x (idx d)‖ ^ 2 := by
    intro idx; rw [symRankOne_apply, norm_prod, ← Finset.prod_pow]
  rw [Finset.sum_congr rfl (fun idx _ => h idx),
    show (∑ i, ‖x i‖ ^ 2) ^ (D + 1) = ∏ _d : Fin (D + 1), (∑ i, ‖x i‖ ^ 2) from by
      rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin],
    Finset.prod_univ_sum, Fintype.piFinset_univ]

/-- For a **unit** vector `x`, the rank-1 tensor `u = x ◯ ⋯ ◯ x` is a unit tensor:
    `⟪u, u⟫ = 1`. -/
theorem dotProd_symRankOne_self_of_unit (hx : ∑ i, ‖x i‖ ^ 2 = 1) :
    ⟪symRankOne x, (symRankOne x : CubicTensor 𝕜 (D + 1) N)⟫ₜ = (1 : 𝕜) := by
  rw [dotProd_self_eq_ofReal]
  have hsum : (∑ idx, ‖(symRankOne x : CubicTensor 𝕜 (D + 1) N) idx‖ ^ 2) = (1 : ℝ) := by
    rw [← frobNormSq_def, frobNormSq_symRankOne, hx, one_pow]
  rw [hsum, RCLike.ofReal_one]

/-- The residual `X - ⟪u, X⟫ u` is orthogonal to `u` (tested on the left):
    `⟪u, X - ⟪u,X⟫ u⟫ = 0`. -/
theorem dotProd_symRankOne_residual (hx : ∑ i, ‖x i‖ ^ 2 = 1)
    (X : CubicTensor 𝕜 (D + 1) N) :
    ⟪symRankOne x, X - (⟪symRankOne x, X⟫ₜ) • symRankOne x⟫ₜ = 0 := by
  rw [sub_eq_add_neg, ← neg_smul, dotProd_add_right, dotProd_smul_right,
    dotProd_symRankOne_self_of_unit hx]
  ring

/-- The residual is orthogonal to `u` (tested on the right): `⟪X - ⟪u,X⟫ u, u⟫ = 0`. -/
theorem dotProd_residual_symRankOne (hx : ∑ i, ‖x i‖ ^ 2 = 1)
    (X : CubicTensor 𝕜 (D + 1) N) :
    ⟪X - (⟪symRankOne x, X⟫ₜ) • symRankOne x, symRankOne x⟫ₜ = 0 := by
  rw [dotProd_conj_symm _ (symRankOne x), dotProd_symRankOne_residual hx X, map_zero]

/-- **Optimal rank-1 coefficient (Lecture 8, thm:rank1-gamma).** For a fixed unit
    vector `x`, over all scalars `γ` the coefficient `γ⋆ = ⟪u, X⟫` (the tensor
    Rayleigh quotient `R_X(x)`) minimises the rank-1 approximation error
    `‖X - γ · u‖_F`, where `u = x ◯ ⋯ ◯ x`. -/
theorem frobNormSq_sub_symRankOne_min (hx : ∑ i, ‖x i‖ ^ 2 = 1)
    (X : CubicTensor 𝕜 (D + 1) N) (γ : 𝕜) :
    frobNormSq (X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
      ≤ frobNormSq (X - γ • symRankOne x) := by
  -- `X - γ u = (X - γ⋆ u) + (γ⋆ - γ) u`, with the two summands orthogonal.
  have hsplit : X - γ • symRankOne x
      = (X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
          + ((⟪symRankOne x, X⟫ₜ) - γ) • symRankOne x := by
    rw [sub_smul]; abel
  have horth : ⟪X - (⟪symRankOne x, X⟫ₜ) • symRankOne x,
      ((⟪symRankOne x, X⟫ₜ) - γ) • symRankOne x⟫ₜ = 0 := by
    rw [dotProd_smul_right, dotProd_residual_symRankOne hx X, mul_zero]
  rw [hsplit, frobNormSq_add_of_orthogonal horth]
  have hnn : 0 ≤ frobNormSq (((⟪symRankOne x, X⟫ₜ) - γ)
      • (symRankOne x : CubicTensor 𝕜 (D + 1) N)) :=
    frobNormSq_nonneg _
  linarith

/-- **Dual equivalence (Lecture 8, thm:rank1-dual).** At the optimal coefficient
    `γ⋆ = ⟪u, X⟫`, the residual energy is `‖X - γ⋆ u‖_F² = ‖X‖_F² - ‖γ⋆‖²`. Hence
    minimising the rank-1 error is the same as maximising `‖R_X(x)‖`, i.e. finding
    the largest-magnitude Z-eigenvalue. -/
theorem frobNormSq_residual_eq (hx : ∑ i, ‖x i‖ ^ 2 = 1)
    (X : CubicTensor 𝕜 (D + 1) N) :
    frobNormSq (X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
      = frobNormSq X - ‖⟪symRankOne x, X⟫ₜ‖ ^ 2 := by
  -- `‖γ⋆ u‖² = ‖γ⋆‖² ‖u‖² = ‖γ⋆‖²`.
  have hcu : frobNormSq ((⟪symRankOne x, X⟫ₜ) • (symRankOne x : CubicTensor 𝕜 (D + 1) N))
      = ‖⟪symRankOne x, X⟫ₜ‖ ^ 2 := by
    rw [frobNormSq_def]
    have h : ∀ idx, ‖((⟪symRankOne x, X⟫ₜ) • (symRankOne x : CubicTensor 𝕜 (D + 1) N)) idx‖ ^ 2
        = ‖⟪symRankOne x, X⟫ₜ‖ ^ 2 * ‖(symRankOne x : CubicTensor 𝕜 (D + 1) N) idx‖ ^ 2 := by
      intro idx; rw [Tensor.smul_apply, norm_mul, mul_pow]
    rw [Finset.sum_congr rfl (fun idx _ => h idx), ← Finset.mul_sum,
      ← frobNormSq_def, frobNormSq_symRankOne, hx, one_pow, mul_one]
  -- Pythagoras on `X = (X - γ⋆ u) + γ⋆ u`.
  have horthX : ⟪X - (⟪symRankOne x, X⟫ₜ) • symRankOne x,
      (⟪symRankOne x, X⟫ₜ) • symRankOne x⟫ₜ = 0 := by
    rw [dotProd_smul_right, dotProd_residual_symRankOne hx X, mul_zero]
  have heq : (X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
      + (⟪symRankOne x, X⟫ₜ) • symRankOne x = X := by abel
  have hpyth : frobNormSq X
      = frobNormSq (X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
        + frobNormSq ((⟪symRankOne x, X⟫ₜ) • (symRankOne x : CubicTensor 𝕜 (D + 1) N)) :=
    calc frobNormSq X
        = frobNormSq ((X - (⟪symRankOne x, X⟫ₜ) • symRankOne x)
            + (⟪symRankOne x, X⟫ₜ) • symRankOne x) := by rw [heq]
      _ = _ := frobNormSq_add_of_orthogonal horthX
  rw [hcu] at hpyth
  linarith

end CubicTensor
