/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# General Order-D Tensors

This file defines tensors of arbitrary order over `RCLike` scalars.

A tensor of order `D` with shape `dims : Fin D → ℕ` is a function from
multi-indices to scalars. A multi-index is a dependent function
`(d : Fin D) → Fin (dims d)`.

## Main definitions

* `Tensor 𝕜 D dims` — the type of order-D tensors
* `Tensor.prodExcept n dims` — product of `dims` over all modes except `n`

The algebraic instances (`AddCommGroup`, `Module`, etc.) are inherited
from the `Pi` instances automatically via `abbrev`.
-/

noncomputable section

/-- A tensor of order `D` with shape `dims` over scalars `𝕜`.
    An element is a function from multi-indices `(d : Fin D) → Fin (dims d)` to `𝕜`. -/
abbrev Tensor (𝕜 : Type*) [RCLike 𝕜] (D : ℕ) (dims : Fin D → ℕ) :=
  ((d : Fin D) → Fin (dims d)) → 𝕜

/-- Product of `dims` over all modes except `n`. The column dimension of the
    mode-`n` matricization of a tensor with shape `dims`. -/
abbrev Tensor.prodExcept {D : ℕ} (n : Fin D) (dims : Fin D → ℕ) : ℕ :=
  ∏ d ∈ Finset.univ.erase n, dims d

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

@[simp]
theorem zero_apply (idx : (d : Fin D) → Fin (dims d)) :
    (0 : Tensor 𝕜 D dims) idx = 0 := rfl

@[simp]
theorem add_apply (T₁ T₂ : Tensor 𝕜 D dims) (idx : (d : Fin D) → Fin (dims d)) :
    (T₁ + T₂) idx = T₁ idx + T₂ idx := rfl

@[simp]
theorem smul_apply (c : 𝕜) (T : Tensor 𝕜 D dims) (idx : (d : Fin D) → Fin (dims d)) :
    (c • T) idx = c * T idx := rfl

@[simp]
theorem neg_apply (T : Tensor 𝕜 D dims) (idx : (d : Fin D) → Fin (dims d)) :
    (-T) idx = -(T idx) := rfl

/-- Two tensors are equal iff they agree on all multi-indices. -/
@[ext]
theorem ext {T₁ T₂ : Tensor 𝕜 D dims}
    (h : ∀ idx, T₁ idx = T₂ idx) : T₁ = T₂ :=
  funext h

end Tensor
