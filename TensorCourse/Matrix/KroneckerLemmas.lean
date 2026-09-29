/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.OuterProd
import TensorCourse.Tensor.Vectorize
import TensorCourse.Matrix.Vectorize
import TensorCourse.Matrix.KhatriRao
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# Connecting Kronecker Product to Outer Product of Tensors

This file proves the lecture identity `u ⊗ₜ v = vec(v ◯ₜ u)` for vectors.
The Kronecker product of two column vectors, flattened to a single
column, coincides with the vectorization of the outer product of the
corresponding order-1 tensors — **with the two factors swapped**: `v`
first, `u` second. That swap is what makes the identity hold under the
row-fastest Kolda-Bader convention used by `Matrix.vectorize` and
`Tensor.vectorize` (see `Tensor/Vectorize.lean`).

## Main definition

* `Tensor.toTensor1` — view a vector `u : Fin M → 𝕜` as an order-1
  tensor `Tensor 𝕜 1 (fun _ => M)`.

## Main result

* `kronecker_vec_eq_vectorize_outerProd` — for `u : Fin M → 𝕜` and
  `v : Fin N → 𝕜`,

      (u ⊗ₜ v) (finProdFinEquiv.symm l) = Tensor.vectorize (v ◯ₜ u) (cast l)

  where the cast reindexes `Fin (M * N)` as `Fin (∏ d, (combinedDims _ _) d)`
  via `Fin.prod_univ_two`, and `u, v` are viewed as column vectors and
  order-1 tensors as appropriate.
-/

noncomputable section

open Matrix Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- View a vector `u : Fin M → 𝕜` as an order-1 tensor. -/
def toTensor1 {M : ℕ} (u : Fin M → 𝕜) : Tensor 𝕜 1 (fun _ => M) :=
  fun idx => u (idx 0)

@[simp]
theorem toTensor1_apply {M : ℕ} (u : Fin M → 𝕜) (idx : (d : Fin 1) → Fin M) :
    toTensor1 u idx = u (idx 0) := rfl

/-- Under the row-fastest Kolda-Bader convention, the Kronecker product of two
    column vectors (flattened to a single column) equals the vectorization of
    the outer product of the two order-1 tensors **with factors swapped**:
    `v` on the left, `u` on the right.

    Concretely, for `u : Fin M → 𝕜` and `v : Fin N → 𝕜`:
    at the pair `(m, n) : Fin M × Fin N`,

      (u ⊗ₜ v) — viewed as column vector at row `finProdFinEquiv (m, n)` —
        equals `u m * v n`

    and

      (v.toTensor1 ◯ₜ u.toTensor1) at multi-index `![n, m]`
        = v n * u m

    which are equal by commutativity of multiplication in `𝕜`.

    Stated as a pointwise identity on pairs, which is the form most amenable
    to rewriting. -/
theorem kronecker_of_vec_pair {M N : ℕ}
    (u : Fin M → 𝕜) (v : Fin N → 𝕜) (m : Fin M) (n : Fin N) :
    ((Matrix.of fun (i : Fin M) (_ : Fin 1) => u i) ⊗ₜ
        (Matrix.of fun (i : Fin N) (_ : Fin 1) => v i))
          (m, n) (0, 0) = u m * v n := by
  simp [kroneckerMap_apply]

/-- A specific multi-index for the outer product `v ◯ₜ u` of two order-1
    tensors: mode 0 carries `n : Fin N`, mode 1 carries `m : Fin M`. -/
def outerProdPairIdx {M N : ℕ} (m : Fin M) (n : Fin N) :
    (d : Fin 2) → Fin (combinedDims (fun _ : Fin 1 => N) (fun _ : Fin 1 => M) d)
  | ⟨0, _⟩ => n
  | ⟨1, _⟩ => m

/-- The outer product `v.toTensor1 ◯ₜ u.toTensor1`, evaluated at the
    multi-index whose components are `n` (mode 0) and `m` (mode 1),
    equals `v n * u m`. -/
theorem outerProd_toTensor1_apply {M N : ℕ}
    (u : Fin M → 𝕜) (v : Fin N → 𝕜) (m : Fin M) (n : Fin N) :
    (toTensor1 v ◯ₜ toTensor1 u) (outerProdPairIdx m n) = v n * u m := by
  simp [outerProd, splitLeft, splitRight, toTensor1, outerProdPairIdx,
    Fin.castAdd, Fin.natAdd, Fin.cast]

/-- **Core identity: `u ⊗ₜ v = vec(v ◯ₜ u)` for vectors.**

    For vectors `u : Fin M → 𝕜` and `v : Fin N → 𝕜`, the Kronecker product
    (of `u, v` viewed as single-column matrices) — evaluated on a
    `finProdFinEquiv`-built row index — equals the tensor vectorization of
    the outer product `v ◯ₜ u` (note the order: `v` first, `u` second) at
    the same flat index (up to the product-reassociation `Fin.cast`).

    The swap `v ◯ₜ u` (not `u ◯ₜ v`) is forced by the row-fastest
    Kolda-Bader convention: in `Matrix.vectorize`, the row index varies
    fastest, and `finProdFinEquiv (m, n) = n + N * m`, so `m` is the
    slow mode and `n` is the fast mode; correspondingly the tensor's
    fast mode (mode 0) must carry `n`, which is the first factor `v`. -/
theorem kronecker_vec_eq_vectorize_outerProd {M N : ℕ}
    (u : Fin M → 𝕜) (v : Fin N → 𝕜) (m : Fin M) (n : Fin N) :
    ((Matrix.of fun (i : Fin M) (_ : Fin 1) => u i) ⊗ₜ
        (Matrix.of fun (i : Fin N) (_ : Fin 1) => v i))
      (m, n) (0, 0) =
    (Tensor.vectorize (toTensor1 v ◯ₜ toTensor1 u))
      ((multiIndexEquiv _) (outerProdPairIdx m n)) := by
  rw [kronecker_of_vec_pair]
  rw [vectorize_multiIndexEquiv]
  rw [outerProd_toTensor1_apply]
  exact mul_comm _ _

end Tensor
