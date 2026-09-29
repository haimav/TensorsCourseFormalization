/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.Brackets
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Diagonal
import Mathlib.Data.Fin.VecNotation
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# σ-Form CP Decomposition and Indeterminacies

This file introduces the *weighted* CP reconstruction (σ-form), the
σ-form bracket macro `⟦σ ;; A, B, …⟧`, scaling/permutation
indeterminacies, and the order-2 identity linking σ-form CP to a
matrix factorisation `U Σ Vᵀ` (the SVD-link identity).

## Main definitions

* `CPDecomp.tensorWeighted` — weighted reconstruction
  `T(idx) = ∑ r, σ r * ∏ d, factors r d (idx d)`.
* Scoped notation `⟦σ ;; A, B, …⟧` (for `D ≥ 2`) — a `Tensor`
  built by feeding the factor matrices through the existing
  `cpBracketCases%` helper and reconstructing with weights.
* `CPDecomp.scaleFactor` — multiply one mode's factor by `σ`.
* `CPDecomp.scaleFactors` — multiply each mode's factor by `α d`.
* `CPDecomp.permuteRanks` — reindex the rank summation by a
  permutation `π : Equiv.Perm (Fin R)`.

## Main results

* `CPDecomp.tensorWeighted_eq_tensor_absorb_zero` — when `0 < D`,
  the σ-form coincides with the plain CP decomposition obtained by
  absorbing the weights into the mode-0 factor.
* `CPDecomp.tensor_scaleAll` — scaling indeterminacy: if
  `∏ d, α d = 1`, scaling factors mode-by-mode preserves the tensor.
* `CPDecomp.tensor_permute` — permutation indeterminacy: reindexing
  the rank-`R` summation does not change the tensor.
* `CPDecomp.tensor_scale_permute` — combined scaling-and-permutation
  indeterminacy.
* `cpWeighted_two_eq_matrix_mul` — order-2 identity linking
  `⟦σ ;; U, V⟧` to `(U * diagonal σ * Vᵀ) i j`. This is the algebraic
  content of the SVD ↔ CP correspondence for matrices.

## Disambiguation

The plain bracket `⟦A, B, …⟧` (defined in
`CPDecomposition/Brackets.lean`) elaborates to a `CPDecomp`. The
σ-form bracket `⟦σ ;; A, B, …⟧` introduced here elaborates to a
`Tensor` (the result of `tensorWeighted`). The `;;` separator
disambiguates the two macros at the parse level.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}

/-! ### Weighted reconstruction (`tensorWeighted`) -/

/-- Reconstruct a tensor from a CP decomposition with explicit weights.

    `tensorWeighted σ cp idx = ∑ r, σ r * ∏ d, cp.factors r d (idx d)`.

    Equivalent to `(cp.scaleFactor 0 σ).tensor` when `0 < D`; see
    `tensorWeighted_eq_tensor_absorb_zero`. -/
def CPDecomp.tensorWeighted (σ : Fin R → 𝕜) (cp : CPDecomp 𝕜 D dims R) :
    Tensor 𝕜 D dims :=
  fun idx => ∑ r : Fin R, σ r * ∏ d : Fin D, cp.factors r d (idx d)

theorem CPDecomp.tensorWeighted_apply
    (σ : Fin R → 𝕜) (cp : CPDecomp 𝕜 D dims R)
    (idx : (d : Fin D) → Fin (dims d)) :
    cp.tensorWeighted σ idx = ∑ r : Fin R, σ r * ∏ d : Fin D, cp.factors r d (idx d) :=
  rfl

/-! ### Bridge to plain CP via `scaleFactor` -/

/-- Multiply the mode-`d` factor of a CP decomposition by `σ r`
    (depending on the rank index). All other factors are left
    unchanged. Used as the bridge between σ-form CP and plain CP. -/
def CPDecomp.scaleFactor (cp : CPDecomp 𝕜 D dims R) (d : Fin D)
    (σ : Fin R → 𝕜) : CPDecomp 𝕜 D dims R where
  factors r d' i :=
    if h : d' = d then σ r * cp.factors r d (h ▸ i) else cp.factors r d' i

/-- Absorbing the weights into mode `0` reduces the σ-form to plain
    CP. Requires `0 < D`. -/
theorem CPDecomp.tensorWeighted_eq_tensor_absorb_zero
    (σ : Fin R → 𝕜) (cp : CPDecomp 𝕜 D dims R) (h : 0 < D) :
    cp.tensorWeighted σ = (cp.scaleFactor ⟨0, h⟩ σ).tensor := by
  ext idx
  simp only [CPDecomp.tensorWeighted, CPDecomp.tensor]
  apply Finset.sum_congr rfl
  intro r _
  -- Split the product on the LHS at the mode-0 factor.
  rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
        (fun d => cp.factors r d (idx d)) (Finset.mem_univ ⟨0, h⟩)]
  -- Split the product on the RHS at the mode-0 factor.
  rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
        (fun d => (cp.scaleFactor ⟨0, h⟩ σ).factors r d (idx d))
        (Finset.mem_univ ⟨0, h⟩)]
  -- Match factor-at-mode-0 (gets σ) and factors-at-other-modes
  -- (left untouched) on each side.
  have h0 : (cp.scaleFactor ⟨0, h⟩ σ).factors r ⟨0, h⟩ (idx ⟨0, h⟩)
              = σ r * cp.factors r ⟨0, h⟩ (idx ⟨0, h⟩) := by
    simp [CPDecomp.scaleFactor]
  rw [h0]
  have hother : ∀ d ∈ (Finset.univ : Finset (Fin D)).erase ⟨0, h⟩,
      (cp.scaleFactor ⟨0, h⟩ σ).factors r d (idx d) = cp.factors r d (idx d) := by
    intro d hd
    have hne : d ≠ ⟨0, h⟩ := (Finset.mem_erase.mp hd).1
    simp [CPDecomp.scaleFactor, hne]
  rw [Finset.prod_congr rfl hother]
  ring

/-! ### σ-form bracket macro `⟦σ ;; A, B, …⟧`

The macro mirrors the existing `⟦A, B, …⟧` macro from
`CPDecomposition/Brackets.lean`, reusing its `cpBracketCases%`
helper. The result of `⟦σ ;; A, B, …⟧` is a `Tensor 𝕜 D ![…] R`,
not a `CPDecomp`. The `;;` separator disambiguates it from the
plain bracket. -/

/-- σ-form CP bracket. `⟦σ ;; A₁, A₂, …, A_D⟧ : Tensor 𝕜 D ![I₁,…,I_D]`,
    requiring `D ≥ 2`. Internally expands to
    `CPDecomp.tensorWeighted σ ⟦A₁, A₂, …, A_D⟧`. -/
scoped syntax (name := cpBracketsW) "⟦" term ";;" term,+ "⟧" : term

macro_rules
  | `(⟦$σ:term ;; $a:term, $bs:term,*⟧) =>
      `(CPDecomp.tensorWeighted $σ
          (({ factors := fun r => cpBracketCases% [$a, $bs,*] at r } :
              CPDecomp _ _ _ _)))

/-! #### Beta-style reduction lemmas for the σ-form bracket -/

section ArityW2

variable {I₁ I₂ R : ℕ}
  (σ : Fin R → 𝕜)
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)

/-- The σ-form CP tensor `⟦σ ;; A, B⟧` evaluated pointwise. -/
theorem bracketW_tensor_apply₂
    (idx : (d : Fin 2) → Fin (![I₁, I₂] d)) :
    (⟦σ ;; A, B⟧ : Tensor 𝕜 2 ![I₁, I₂]) idx =
      ∑ r : Fin R, σ r * (A (idx 0) r * B (idx 1) r) := by
  simp only [CPDecomp.tensorWeighted]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 2)) = {0, 1} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_singleton]
  rfl

end ArityW2

section ArityW3

variable {I₁ I₂ I₃ R : ℕ}
  (σ : Fin R → 𝕜)
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)
  (C : Matrix (Fin I₃) (Fin R) 𝕜)

/-- The σ-form CP tensor `⟦σ ;; A, B, C⟧` evaluated pointwise. -/
theorem bracketW_tensor_apply₃
    (idx : (d : Fin 3) → Fin (![I₁, I₂, I₃] d)) :
    (⟦σ ;; A, B, C⟧ : Tensor 𝕜 3 ![I₁, I₂, I₃]) idx =
      ∑ r : Fin R, σ r * (A (idx 0) r * B (idx 1) r * C (idx 2) r) := by
  simp only [CPDecomp.tensorWeighted]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 3)) = {0, 1, 2} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
    Finset.prod_singleton]
  change σ r * (A (idx 0) r * (B (idx 1) r * C (idx 2) r)) = _
  ring

end ArityW3

section ArityW4

variable {I₁ I₂ I₃ I₄ R : ℕ}
  (σ : Fin R → 𝕜)
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)
  (C : Matrix (Fin I₃) (Fin R) 𝕜) (D' : Matrix (Fin I₄) (Fin R) 𝕜)

/-- The σ-form CP tensor `⟦σ ;; A, B, C, D⟧` evaluated pointwise. -/
theorem bracketW_tensor_apply₄
    (idx : (d : Fin 4) → Fin (![I₁, I₂, I₃, I₄] d)) :
    (⟦σ ;; A, B, C, D'⟧ : Tensor 𝕜 4 ![I₁, I₂, I₃, I₄]) idx =
      ∑ r : Fin R,
        σ r * (A (idx 0) r * B (idx 1) r * C (idx 2) r * D' (idx 3) r) := by
  simp only [CPDecomp.tensorWeighted]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 4)) = {0, 1, 2, 3} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
    Finset.prod_insert (by decide), Finset.prod_singleton]
  change σ r * (A (idx 0) r * (B (idx 1) r * (C (idx 2) r * D' (idx 3) r))) = _
  ring

end ArityW4

/-! ### Indeterminacies: `scaleFactors` and `permuteRanks` -/

/-- Multiply each mode's factor by `α d` (independent of the rank
    index). Combined with `∏ d, α d = 1`, this is the scaling
    indeterminacy. -/
def CPDecomp.scaleFactors (cp : CPDecomp 𝕜 D dims R) (α : Fin D → 𝕜) :
    CPDecomp 𝕜 D dims R where
  factors r d i := α d * cp.factors r d i

/-- Reindex the rank summation by a permutation `π`. The
    reconstructed tensor is unchanged; see `tensor_permute`. -/
def CPDecomp.permuteRanks (cp : CPDecomp 𝕜 D dims R)
    (π : Equiv.Perm (Fin R)) : CPDecomp 𝕜 D dims R where
  factors r d i := cp.factors (π r) d i

/-- **Scaling indeterminacy.** If `∏ d, α d = 1`, then scaling each
    mode's factor by the corresponding `α d` does not change the
    reconstructed tensor. -/
theorem CPDecomp.tensor_scaleAll
    (cp : CPDecomp 𝕜 D dims R) (α : Fin D → 𝕜) (h : ∏ d, α d = 1) :
    (cp.scaleFactors α).tensor = cp.tensor := by
  ext idx
  simp only [CPDecomp.tensor, CPDecomp.scaleFactors]
  apply Finset.sum_congr rfl
  intro r _
  rw [Finset.prod_mul_distrib, h, one_mul]

/-- **Permutation indeterminacy.** Reindexing the rank-`R` summation
    by a permutation `π : Equiv.Perm (Fin R)` leaves the
    reconstructed tensor unchanged. -/
theorem CPDecomp.tensor_permute
    (cp : CPDecomp 𝕜 D dims R) (π : Equiv.Perm (Fin R)) :
    (cp.permuteRanks π).tensor = cp.tensor := by
  ext idx
  simp only [CPDecomp.tensor, CPDecomp.permuteRanks]
  exact (Equiv.sum_comp π (fun r => ∏ d : Fin D, cp.factors r d (idx d)))

/-- **Joint scaling-and-permutation indeterminacy.** -/
theorem CPDecomp.tensor_scale_permute
    (cp : CPDecomp 𝕜 D dims R) (α : Fin D → 𝕜) (h : ∏ d, α d = 1)
    (π : Equiv.Perm (Fin R)) :
    ((cp.scaleFactors α).permuteRanks π).tensor = cp.tensor := by
  rw [CPDecomp.tensor_permute, CPDecomp.tensor_scaleAll _ _ h]

/-! ### Order-2 SVD-link identity

For matrices `U`, `V` with shared column count `R` and a weight
vector `σ : Fin R → 𝕜`, the σ-form CP `⟦σ ;; U, V⟧` evaluates to
`(U * diagonal σ * Vᵀ) i j`. This is the algebraic content of the
SVD ↔ CP correspondence for matrices. -/

/-- **Order-2 σ-form CP equals the matrix product `U * diagonal σ * Vᵀ`.** -/
theorem cpWeighted_two_eq_matrix_mul
    {I J R : ℕ}
    (σ : Fin R → 𝕜)
    (U : Matrix (Fin I) (Fin R) 𝕜) (V : Matrix (Fin J) (Fin R) 𝕜)
    (i : Fin I) (j : Fin J) :
    (⟦σ ;; U, V⟧ : Tensor 𝕜 2 ![I, J])
      (fun d => match d with | ⟨0, _⟩ => i | ⟨1, _⟩ => j)
    = (U * Matrix.diagonal σ * Vᵀ) i j := by
  -- LHS via the per-arity bracket reduction.
  rw [bracketW_tensor_apply₂]
  -- RHS: expand the matrix product, apply mul_diagonal to fold one diagonal,
  -- then `mul_apply` for the remaining product.
  rw [Matrix.mul_apply]
  simp only [Matrix.mul_diagonal, Matrix.transpose_apply]
  -- Both sides are now ∑ r, σ r * (U i r * V j r) up to commutativity.
  apply Finset.sum_congr rfl
  intro r _
  ring

end Tensor
