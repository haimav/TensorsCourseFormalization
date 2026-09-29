/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs

/-!
# Tensor Reindexing (Reshape)

Reindexing transforms a tensor by relabeling its multi-indices via an
equivalence, without changing any values. This captures mode permutation,
mode merging (reshape), and mode splitting.

## Main definitions

* `Tensor.reindex` — reindex a tensor along an equivalence of multi-index sets
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D₁ D₂ : ℕ}
variable {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}

/-- Reindex a tensor by an equivalence between multi-index sets.
    `Tensor.reindex e T` has the same values as `T` but with relabeled indices:
    `(reindex e T) idx = T (e.symm idx)`. -/
def reindex
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (T : Tensor 𝕜 D₁ dims₁) : Tensor 𝕜 D₂ dims₂ :=
  fun idx => T (e.symm idx)

@[simp]
theorem reindex_apply
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (T : Tensor 𝕜 D₁ dims₁)
    (idx : (d : Fin D₂) → Fin (dims₂ d)) :
    reindex e T idx = T (e.symm idx) := rfl

/-- Reindexing by the identity equivalence is the identity. -/
@[simp]
theorem reindex_id (T : Tensor 𝕜 D₁ dims₁) :
    reindex (Equiv.refl _) T = T := by
  ext idx; simp [reindex]

/-- Reindexing composes contravariantly. -/
theorem reindex_comp {D₃ : ℕ} {dims₃ : Fin D₃ → ℕ}
    (e₁ : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (e₂ : ((d : Fin D₂) → Fin (dims₂ d)) ≃ ((d : Fin D₃) → Fin (dims₃ d)))
    (T : Tensor 𝕜 D₁ dims₁) :
    reindex e₂ (reindex e₁ T) = reindex (e₁.trans e₂) T := by
  ext idx; simp [reindex]

/-- Reindexing is additive. -/
theorem reindex_add
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (T₁ T₂ : Tensor 𝕜 D₁ dims₁) :
    reindex e (T₁ + T₂) = reindex e T₁ + reindex e T₂ := by
  ext idx; simp [reindex]

/-- Reindexing commutes with scalar multiplication. -/
theorem reindex_smul
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (c : 𝕜) (T : Tensor 𝕜 D₁ dims₁) :
    reindex e (c • T) = c • reindex e T := by
  ext idx; simp [reindex]

/-- Reindexing is injective (and in fact bijective). -/
theorem reindex_injective
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d))) :
    Function.Injective (reindex (𝕜 := 𝕜) e) := by
  intro T₁ T₂ h
  ext idx
  have h' := congr_fun h (e idx)
  simpa [reindex] using h'

end Tensor
