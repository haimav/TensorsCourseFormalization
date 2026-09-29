/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.OuterProd
import Mathlib.Data.Nat.Lattice

/-!
# Sub-multiplicativity of CP Rank

This file proves that the CP rank of the outer product of two tensors is
at most the product of their CP ranks.

## Main results

* `cpRank_outerProd_le_of_decomps` — given explicit decompositions, bound the outer product rank
* `cpRank_outerProd_le` — `cpRank (A ◯ₜ B) ≤ cpRank A * cpRank B`
  (for positive-order tensors; CP-existence discharged via `exists_cpDecomp`)
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D₁ D₂ : ℕ}
variable {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}

open scoped Tensor

/-- If there exist CP decompositions of ranks `R₁` and `R₂` for `A` and `B`,
    then there is a CP decomposition of rank `R₁ * R₂` for `A ◯ₜ B`. -/
theorem cpRank_outerProd_le_of_decomps
    {A : Tensor 𝕜 D₁ dims₁} {B : Tensor 𝕜 D₂ dims₂}
    {R₁ R₂ : ℕ}
    (cp₁ : CPDecomp 𝕜 D₁ dims₁ R₁) (cp₂ : CPDecomp 𝕜 D₂ dims₂ R₂)
    (h₁ : cp₁.tensor = A) (h₂ : cp₂.tensor = B) :
    cpRank (A ◯ₜ B) ≤ R₁ * R₂ := by
  apply cpRank_le_of_cpDecomp (cp₁ ◯ₜ cp₂)
  rw [CPDecomp.outerProd_tensor]
  rw [h₁, h₂]

/-- The CP rank of the outer product is at most the product of the CP ranks,
    for tensors of positive order.

    `cpRank` is defined via `sInf` with `sInf ∅ = 0` for `ℕ`, so the bound
    could fail only for a tensor with *no* CP decomposition — possible solely
    at order 0. For order ≥ 1 every tensor admits a CP decomposition
    (`exists_cpDecomp`), so we take `0 < D₁`, `0 < D₂` and discharge the
    existence internally. The fully general statement (any orders) is
    `cpRank_outerProd_le_of_decomps`, which takes the decompositions
    explicitly. -/
theorem cpRank_outerProd_le (A : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂)
    (hD₁ : 0 < D₁) (hD₂ : 0 < D₂) :
    cpRank (A ◯ₜ B) ≤ cpRank A * cpRank B := by
  -- Positive order ⇒ both tensors admit a CP decomposition.
  have hA := exists_cpDecomp hD₁ A
  have hB := exists_cpDecomp hD₂ B
  -- Extract optimal decompositions at ranks cpRank A and cpRank B
  have hA' : Set.Nonempty { r : ℕ | ∃ cp : CPDecomp 𝕜 D₁ dims₁ r, cp.tensor = A } :=
    let ⟨r, cp, h⟩ := hA; ⟨r, cp, h⟩
  have hB' : Set.Nonempty { r : ℕ | ∃ cp : CPDecomp 𝕜 D₂ dims₂ r, cp.tensor = B } :=
    let ⟨r, cp, h⟩ := hB; ⟨r, cp, h⟩
  obtain ⟨cpA, hcpA⟩ := Nat.sInf_mem hA'
  obtain ⟨cpB, hcpB⟩ := Nat.sInf_mem hB'
  exact cpRank_outerProd_le_of_decomps cpA cpB hcpA hcpB

end Tensor
