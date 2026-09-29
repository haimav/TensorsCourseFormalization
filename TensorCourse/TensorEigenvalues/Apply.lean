/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Cubic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# Applying a Tensor to a Vector (Lecture 8)

This file formalises the *tensor apply* operation from Lecture 8. Given a
regular (cubic) order-`(D+1)` tensor `A ∈ 𝕜^{N×⋯×N}` and a vector
`x ∈ 𝕜^N`, applying `A` to `x` — written `A ⬝ₜ x` in this development and
`𝒜 x` in the lecture — contracts the *last* `D` modes of `A` with `x`,
producing a vector in `𝕜^N`:

  `(A ⬝ₜ x) i = ∑_{i₂,…,i_{D+1}} A(i, i₂, …, i_{D+1}) · x_{i₂} ⋯ x_{i_{D+1}}`.

In the lecture this is defined as the chain of vector `n`-mode products
`A ×̄₂ x ×̄₃ x ⋯ ×̄_D x`; the pointwise sum above (which the lecture also
gives, via horizontal slices) is the closed form we take as the definition.
It is the standard `A xᴰ⁻¹` object of tensor-eigenvalue theory. The order is
written `D+1` so that a cubic tensor always has at least one mode and the
result is genuinely a vector; the homogeneity degree is then `D` (one less
than the order, matching the lecture's "homogeneous of degree `D-1`").

## Main definitions

* `CubicTensor.apply A x` — tensor apply; scoped notation `A ⬝ₜ x`.

## Main results

* `CubicTensor.apply_apply` — the pointwise defining formula.
* `CubicTensor.apply_smul_v` — homogeneity in `x`: `A ⬝ₜ (c • x) = cᴰ • (A ⬝ₜ x)`
  (Lecture 8, the "homogeneous of degree `D-1`" fact).
* `CubicTensor.apply_neg_v` — the sign specialisation `A ⬝ₜ (-x) = (-1)ᴰ • (A ⬝ₜ x)`.
* `CubicTensor.apply_add_T`, `CubicTensor.apply_smul_T` — linearity in the
  tensor argument.
-/

noncomputable section

open Finset BigOperators

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}

/-- **Applying a tensor to a vector.** For a cubic order-`(D+1)` tensor `A`
    and a vector `x : Fin N → 𝕜`, `A.apply x` contracts the last `D` modes
    of `A` with `x`:

      `(A.apply x) i = ∑ rest, A (Fin.cons i rest) * ∏ d, x (rest d)`,

    where `rest : Fin D → Fin N` ranges over the remaining `D` mode indices
    and `Fin.cons i rest : Fin (D+1) → Fin N` prepends the free index `i`.
    This is the lecture's `A ×̄₂ x ×̄₃ x ⋯ ×̄_D x`, written in closed form. -/
def apply (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) : Fin N → 𝕜 :=
  fun i => ∑ rest : Fin D → Fin N, A (Fin.cons i rest) * ∏ d : Fin D, x (rest d)

@[inherit_doc]
scoped infixl:74 " ⬝ₜ " => CubicTensor.apply

/-- The pointwise defining formula for tensor apply. Deliberately not `@[simp]`
    (it expands into a large sum); rewrite with it when the sum is wanted. -/
theorem apply_apply (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) (i : Fin N) :
    (A ⬝ₜ x) i = ∑ rest : Fin D → Fin N, A (Fin.cons i rest) * ∏ d : Fin D, x (rest d) :=
  rfl

/-- **Homogeneity of tensor apply (Lecture 8).** Tensor apply is homogeneous
    of degree `D` in the vector argument (i.e. degree `order − 1`):
    `A ⬝ₜ (c • x) = cᴰ • (A ⬝ₜ x)`. Taking `c = α ≠ 0` recovers the lecture's
    `𝒜(α x) = α^{D-1} 𝒜 x` for an order-`D` tensor. -/
theorem apply_smul_v (A : CubicTensor 𝕜 (D + 1) N) (c : 𝕜) (x : Fin N → 𝕜) :
    A ⬝ₜ (c • x) = c ^ D • (A ⬝ₜ x) := by
  funext i
  simp only [apply_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  have hprod : ∏ d : Fin D, (c * x (rest d)) = c ^ D * ∏ d : Fin D, x (rest d) := by
    rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [hprod]; ring

/-- Sign specialisation of homogeneity: `A ⬝ₜ (-x) = (-1)ᴰ • (A ⬝ₜ x)`.
    This drives the eigenpair sign symmetry: for even order `D+1` (i.e. odd
    `D`) it flips the eigenvalue, and for odd order (even `D`) it does not. -/
theorem apply_neg_v (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    A ⬝ₜ (-x) = (-1 : 𝕜) ^ D • (A ⬝ₜ x) := by
  rw [← neg_one_smul 𝕜 x, apply_smul_v]

/-- Tensor apply is additive in the tensor argument. -/
theorem apply_add_T (A B : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    (A + B) ⬝ₜ x = A ⬝ₜ x + B ⬝ₜ x := by
  funext i
  simp only [apply_apply, Pi.add_apply, add_mul, Finset.sum_add_distrib]

/-- Scalar multiplication of the tensor pulls out of tensor apply. -/
theorem apply_smul_T (c : 𝕜) (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    (c • A) ⬝ₜ x = c • (A ⬝ₜ x) := by
  funext i
  simp only [apply_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum, mul_assoc]

/-- Tensor apply of the zero tensor is the zero vector. -/
@[simp]
theorem zero_apply_vec (x : Fin N → 𝕜) :
    (0 : CubicTensor 𝕜 (D + 1) N) ⬝ₜ x = 0 := by
  funext i
  simp [apply_apply]

end CubicTensor
