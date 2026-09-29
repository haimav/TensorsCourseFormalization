/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import TensorCourse.Tensor.OuterProd

/-!
# CP Decomposition of Outer Products

This file shows that the outer product of two tensors with CP decompositions
admits a CP decomposition whose rank is the product of the individual ranks.

## Main definitions

* `CPDecomp.outerProdCP` — combines two CP decompositions into one for the outer product
  (with scoped notation `◯ₜ`)

## Main results

* `CPDecomp.outerProd_tensor` — the combined decomposition reconstructs the outer product:
  `(cp₁ ◯ₜ cp₂).tensor = cp₁.tensor ◯ₜ cp₂.tensor`
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D₁ D₂ : ℕ}
variable {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}

/-- Combine two CP decompositions into one for the outer product.
    Given decompositions of ranks `R₁` and `R₂`, this produces a decomposition
    of rank `R₁ * R₂`. The `(r₁, r₂)`-th term uses `cp₁.factors r₁` for the
    first `D₁` modes and `cp₂.factors r₂` for the last `D₂` modes. -/
def CPDecomp.outerProdCP
    (cp₁ : CPDecomp 𝕜 D₁ dims₁ R₁) (cp₂ : CPDecomp 𝕜 D₂ dims₂ R₂) :
    CPDecomp 𝕜 (D₁ + D₂) (combinedDims dims₁ dims₂) (R₁ * R₂) where
  factors := fun r d =>
    let pair := (finProdFinEquiv.symm r : Fin R₁ × Fin R₂)
    Fin.addCases
      (fun d₁ => fun i => cp₁.factors pair.1 d₁ (i.cast (combinedDims_castAdd d₁)))
      (fun d₂ => fun i => cp₂.factors pair.2 d₂ (i.cast (combinedDims_natAdd d₂)))
      d

/-- Scoped notation for CP decomposition outer product, matching the `◯ₜ` notation
    for tensor outer product. Lean dispatches based on the argument types. -/
scoped infixl:70 " ◯ₜ " => CPDecomp.outerProdCP

/-- The combined CP decomposition reconstructs the outer product tensor.

    The key algebraic identity is:
    `∑ (r₁,r₂), ∏ d, f(r₁,r₂,d) = (∑ r₁, ∏ d<D₁, g(r₁,d)) * (∑ r₂, ∏ d<D₂, h(r₂,d))`
    when the factors split according to `Fin.addCases`.

    This requires showing that:
    1. `∏ d : Fin (D₁+D₂), ... = (∏ d₁ : Fin D₁, ...) * (∏ d₂ : Fin D₂, ...)`
       via `Finset.prod_Fin_add`-style lemmas
    2. `∑ r : Fin (R₁*R₂), ... = ∑ r₁, ∑ r₂, ...` via `finProdFinEquiv`
    3. The sum of products distributes as `(∑ a) * (∑ b) = ∑ a, ∑ b, a * b`
       via `Finset.sum_mul_sum`
-/
theorem CPDecomp.outerProd_tensor
    (cp₁ : CPDecomp 𝕜 D₁ dims₁ R₁) (cp₂ : CPDecomp 𝕜 D₂ dims₂ R₂) :
    (cp₁ ◯ₜ cp₂).tensor = cp₁.tensor ◯ₜ cp₂.tensor := by
  ext idx
  simp only [CPDecomp.tensor, outerProd_apply]
  -- Reindex the sum from Fin (R₁ * R₂) to Fin R₁ × Fin R₂.
  rw [(Equiv.sum_comp finProdFinEquiv _).symm]
  -- Split the product over Fin (D₁ + D₂) into products over Fin D₁ and Fin D₂.
  simp_rw [Fin.prod_univ_add]
  -- Unfold outerProdCP and simplify Fin.addCases.
  simp_rw [CPDecomp.outerProdCP, Fin.addCases_left, Fin.addCases_right,
           Equiv.symm_apply_apply]
  -- Decompose the sum over pairs into a double sum.
  rw [Fintype.sum_prod_type]
  -- Factor as a product of sums (the left product depends only on r₁,
  -- and the right product depends only on r₂).
  simp_rw [← Finset.mul_sum]
  rw [← Finset.sum_mul]
  -- Resolve remaining cast equalities via splitLeft/splitRight definitions.
  simp only [splitLeft, splitRight]

end Tensor
