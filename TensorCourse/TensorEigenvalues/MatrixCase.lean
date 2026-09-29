/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Eigenpair
import Mathlib.Data.Matrix.Mul

/-!
# Tensor Eigenpairs Collapse to Matrix Eigenpairs at Order 2 (Lecture 8)

Lecture 8 remarks that "both Z-eigenvalues and H-eigenvalues collapse to
regular matrix eigenvalues when `D = 2`". This file makes that precise. An
order-`2` cubic tensor `A ∈ 𝕜^{N×N}` *is* a matrix `toMatrix A`, and tensor
apply `A ⬝ₜ x` is exactly the matrix–vector product `(toMatrix A) *ᵥ x`.
Consequently the tensor eigen-equation `A ⬝ₜ x = λ • x` is the ordinary matrix
eigen-equation `M x = λ x`, and both the Z- and H-eigenpair conditions reduce
to it.

(The order is written `1 + 1` so that it unifies with the `D + 1` shape of
`CubicTensor.apply`; `1 + 1 = 2`.)

## Main definitions

* `CubicTensor.toMatrix A` — the `N × N` matrix underlying an order-2 tensor.

## Main results

* `CubicTensor.apply_eq_mulVec` — `A ⬝ₜ x = (toMatrix A) *ᵥ x`.
* `CubicTensor.isZEigenpair_iff_matrix` — a Z-eigenpair of an order-2 tensor
  is a unit-norm matrix eigenpair of `toMatrix A`.
* `CubicTensor.isHEigenpair_iff_matrix` — an H-eigenpair of an order-2 tensor
  is a (not-necessarily-normalised) matrix eigenpair of `toMatrix A`.
-/

noncomputable section

open Finset BigOperators Matrix

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {N : ℕ}

/-- The `N × N` matrix underlying an order-2 cubic tensor:
    `(toMatrix A) i j = A (i, j)`. -/
def toMatrix (A : CubicTensor 𝕜 (1 + 1) N) : Matrix (Fin N) (Fin N) 𝕜 :=
  fun i j => A (Fin.cons i (fun _ : Fin 1 => j))

/-- **Tensor apply is matrix–vector multiplication at order 2.** For an
    order-2 cubic tensor, `A ⬝ₜ x = (toMatrix A) *ᵥ x`. -/
theorem apply_eq_mulVec (A : CubicTensor 𝕜 (1 + 1) N) (x : Fin N → 𝕜) :
    A ⬝ₜ x = (toMatrix A) *ᵥ x := by
  funext i
  rw [apply_apply,
    ← Equiv.sum_comp (Equiv.funUnique (Fin 1) (Fin N)).symm
      (fun rest => A (Fin.cons i rest) * ∏ d : Fin 1, x (rest d))]
  simp only [Matrix.mulVec, dotProduct, toMatrix, Equiv.funUnique_symm_apply,
    Fin.prod_univ_one]
  rfl

/-- **Z-eigenpairs of an order-2 tensor are unit matrix eigenpairs.** The
    tensor Z-eigen-equation `A ⬝ₜ x = λ • x` with `‖x‖₂ = 1` is exactly the
    matrix eigen-equation `(toMatrix A) x = λ x` with `‖x‖₂ = 1`. -/
theorem isZEigenpair_iff_matrix (A : CubicTensor 𝕜 (1 + 1) N) (lam : 𝕜)
    (x : Fin N → 𝕜) :
    IsZEigenpair A lam x ↔
      (toMatrix A) *ᵥ x = lam • x ∧ ∑ i, ‖x i‖ ^ 2 = 1 := by
  unfold IsZEigenpair
  rw [apply_eq_mulVec]

/-- **H-eigenpairs of an order-2 tensor are matrix eigenpairs.** At order 2 the
    elementwise power is `x^{[1]} = x`, so the H-eigen-equation is the ordinary
    matrix eigen-equation `(toMatrix A) x = λ x` (no normalisation). -/
theorem isHEigenpair_iff_matrix (A : CubicTensor 𝕜 (1 + 1) N) (lam : 𝕜)
    (x : Fin N → 𝕜) :
    IsHEigenpair A lam x ↔ (toMatrix A) *ᵥ x = lam • x := by
  unfold IsHEigenpair
  rw [apply_eq_mulVec, show (fun i => x i ^ 1) = x from by funext i; rw [pow_one]]

end CubicTensor
