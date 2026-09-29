/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import Mathlib.Data.Fin.Tuple.Basic

/-!
# Squeezing a Singleton Mode of a Tensor

The `squeeze` operation drops a mode whose dimension is `1`: an order-`(D+1)`
tensor with `dims n = 1` becomes an order-`D` tensor with the `n`-th mode
removed. The result has shape `dims ∘ n.succAbove`.

## Main definitions

* `Tensor.squeeze` — drop a singleton mode from a tensor.

## Main results

* `Tensor.squeeze_apply` — pointwise formula for squeeze.
* `Tensor.squeeze_add`, `Tensor.squeeze_smul` — linearity.
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin (D + 1) → ℕ}

/-- Reconstruct a multi-index of length `D + 1` from a multi-index of length
    `D` (for the modes other than `n`) and the singleton entry at mode `n`,
    using `dims n = 1` to fill in the singleton coordinate. -/
private def squeezeFill (n : Fin (D + 1)) (h : dims n = 1)
    (idx : (i : Fin D) → Fin (dims (n.succAbove i))) :
    (d : Fin (D + 1)) → Fin (dims d) :=
  Fin.insertNth n (Fin.cast h.symm 0) idx

/-- Drop a mode whose dimension is `1`. The resulting tensor has order `D`
    and shape `dims ∘ n.succAbove`. -/
def squeeze (n : Fin (D + 1)) (h : dims n = 1) (T : Tensor 𝕜 (D + 1) dims) :
    Tensor 𝕜 D (fun i => dims (n.succAbove i)) :=
  fun idx => T (squeezeFill n h idx)

/-- Pointwise formula: `squeeze n h T idx = T (insertNth n 0 idx)`, where
    `0 : Fin (dims n) = Fin 1` is the unique element. -/
@[simp]
theorem squeeze_apply (n : Fin (D + 1)) (h : dims n = 1) (T : Tensor 𝕜 (D + 1) dims)
    (idx : (i : Fin D) → Fin (dims (n.succAbove i))) :
    squeeze n h T idx = T (Fin.insertNth n (Fin.cast h.symm 0) idx) := rfl

/-- Squeeze is additive in the tensor argument. -/
theorem squeeze_add (n : Fin (D + 1)) (h : dims n = 1)
    (T₁ T₂ : Tensor 𝕜 (D + 1) dims) :
    squeeze n h (T₁ + T₂) = squeeze n h T₁ + squeeze n h T₂ := by
  ext idx
  rfl

/-- Squeeze commutes with scalar multiplication. -/
theorem squeeze_smul (n : Fin (D + 1)) (h : dims n = 1)
    (c : 𝕜) (T : Tensor 𝕜 (D + 1) dims) :
    squeeze n h (c • T) = c • squeeze n h T := by
  ext idx
  rfl

end Tensor
