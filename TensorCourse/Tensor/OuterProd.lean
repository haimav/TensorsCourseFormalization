/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs

/-!
# Outer Product of Tensors

The outer product of an order-`D₁` tensor and an order-`D₂` tensor is an
order-`(D₁ + D₂)` tensor.

## Main definitions

* `Tensor.combinedDims` — concatenation of two dimension functions via `Fin.addCases`
* `Tensor.splitLeft` / `Tensor.splitRight` — project a multi-index over combined dims
* `Tensor.outerProd` — the outer product of two tensors
-/

noncomputable section

open Finset

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D₁ D₂ : ℕ}
variable {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}

/-- Combined dimension function for the outer product: the first `D₁` modes have
    dimensions from `dims₁`, the last `D₂` modes from `dims₂`. -/
def combinedDims (dims₁ : Fin D₁ → ℕ) (dims₂ : Fin D₂ → ℕ) : Fin (D₁ + D₂) → ℕ :=
  Fin.addCases dims₁ dims₂

@[simp]
theorem combinedDims_castAdd (d : Fin D₁) :
    combinedDims dims₁ dims₂ (Fin.castAdd D₂ d) = dims₁ d := by
  simp [combinedDims, Fin.addCases_left]

@[simp]
theorem combinedDims_natAdd (d : Fin D₂) :
    combinedDims dims₁ dims₂ (Fin.natAdd D₁ d) = dims₂ d := by
  simp [combinedDims, Fin.addCases_right]

/-- Extract the first `D₁` components of a multi-index over combined dims. -/
def splitLeft
    (idx : (d : Fin (D₁ + D₂)) → Fin (combinedDims dims₁ dims₂ d)) :
    (d : Fin D₁) → Fin (dims₁ d) :=
  fun d => (idx (Fin.castAdd D₂ d)).cast (combinedDims_castAdd d)

/-- Extract the last `D₂` components of a multi-index over combined dims. -/
def splitRight
    (idx : (d : Fin (D₁ + D₂)) → Fin (combinedDims dims₁ dims₂ d)) :
    (d : Fin D₂) → Fin (dims₂ d) :=
  fun d => (idx (Fin.natAdd D₁ d)).cast (combinedDims_natAdd d)

/-- The outer product of an order-`D₁` tensor `A` and an order-`D₂` tensor `B`
    is an order-`(D₁ + D₂)` tensor whose entry at multi-index `idx` is
    `A(idx_left) * B(idx_right)`. -/
def outerProd
    (A : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂) :
    Tensor 𝕜 (D₁ + D₂) (combinedDims dims₁ dims₂) :=
  fun idx => A (splitLeft idx) * B (splitRight idx)

scoped infixl:70 " ◯ₜ " => outerProd

@[simp]
theorem outerProd_apply (A : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂)
    (idx : (d : Fin (D₁ + D₂)) → Fin (combinedDims dims₁ dims₂ d)) :
    (A ◯ₜ B) idx = A (splitLeft idx) * B (splitRight idx) := rfl

/-- Outer product is linear in the left argument. -/
theorem outerProd_add_left (A₁ A₂ : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂) :
    (A₁ + A₂) ◯ₜ B = A₁ ◯ₜ B + A₂ ◯ₜ B := by
  ext idx
  simp [add_mul]

/-- Outer product is linear in the right argument. -/
theorem outerProd_add_right (A : Tensor 𝕜 D₁ dims₁) (B₁ B₂ : Tensor 𝕜 D₂ dims₂) :
    A ◯ₜ (B₁ + B₂) = A ◯ₜ B₁ + A ◯ₜ B₂ := by
  ext idx
  simp [mul_add]

/-- Scalar multiplication pulls out of the left factor. -/
theorem outerProd_smul_left (c : 𝕜) (A : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂) :
    (c • A) ◯ₜ B = c • (A ◯ₜ B) := by
  ext idx
  simp [mul_assoc]

/-- Scalar multiplication pulls out of the right factor. -/
theorem outerProd_smul_right (c : 𝕜) (A : Tensor 𝕜 D₁ dims₁) (B : Tensor 𝕜 D₂ dims₂) :
    A ◯ₜ (c • B) = c • (A ◯ₜ B) := by
  ext idx
  simp [mul_left_comm]

/-- Outer product with the zero tensor on the left is zero. -/
@[simp]
theorem zero_outerProd (B : Tensor 𝕜 D₂ dims₂) :
    (0 : Tensor 𝕜 D₁ dims₁) ◯ₜ B = 0 := by
  ext idx
  simp

/-- Outer product with the zero tensor on the right is zero. -/
@[simp]
theorem outerProd_zero (A : Tensor 𝕜 D₁ dims₁) :
    A ◯ₜ (0 : Tensor 𝕜 D₂ dims₂) = 0 := by
  ext idx
  simp

end Tensor
