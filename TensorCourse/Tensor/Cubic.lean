/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs

/-!
# Cubic Tensors

A *cubic* (or *cubical*) tensor is one in which every mode has the same
dimension `N`. Several operations — diagonal tensors, symmetry, tensor
powers — are intrinsically only well-defined in this setting, because
permuting modes must yield a tensor of the same type.

## Main definitions

* `CubicTensor 𝕜 D N` — an `abbrev` for `Tensor 𝕜 D (fun _ : Fin D => N)`.

## Zero-cost design

Because `CubicTensor` is an `abbrev` (not a `def`), every lemma about
`Tensor 𝕜 D (fun _ => N)` — including all `Pi`-inherited typeclass
structure (`AddCommGroup`, `Module`, `SMul`, …) — applies automatically
to `CubicTensor 𝕜 D N` without any unfolding or coercion.
-/

noncomputable section

/-- A *cubic* tensor: all `D` modes have the same dimension `N`.

    Pure type alias — inherits everything from `Tensor 𝕜 D (fun _ => N)`. -/
abbrev CubicTensor (𝕜 : Type*) [RCLike 𝕜] (D N : ℕ) : Type _ :=
  Tensor 𝕜 D (fun _ : Fin D => N)

end

