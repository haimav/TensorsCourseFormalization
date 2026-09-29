/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Matrix Multiplication Tensor

This file defines the order-3 tensor `matMulTensor 𝕜 M` encoding the bilinear
map of `M × M` matrix multiplication. Each mode has dimension `M * M`, and a
mode index is decomposed into a pair of indices in `Fin M × Fin M` via
Mathlib's `finProdFinEquiv`.

Under the column-major `Matrix.vectorize` convention
(`Matrix.vectorize A (i + M * j) = A i j`, so
`Matrix.vectorize A l = A (l % M) (l / M)`), the row
index of a flat matrix index lives in the `.2` slot of
`finProdFinEquiv.symm` and the column index in the `.1` slot. Writing
`((k_A, i), (j, k_B), (j', i'))` for the doubly-unpacked multi-index, the
entry is `1` when `i = i'`, `j = j'`, and `k_A = k_B`, and `0` otherwise.
Intuitively: mode 0 indexes the entries of the first matrix `A` (rows × inner),
mode 1 indexes the entries of the second matrix `B` (inner × cols), and mode 2
indexes the entries of the product `A * B` (rows × cols). See
`EncodesMatMul.lean` for a precise statement of this encoding.

## Main definitions

* `matMulDims M` — the constant dimension function `fun _ => M * M` for the
  three modes of the matrix multiplication tensor
* `matMulTensor 𝕜 M` — the matrix multiplication tensor itself
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- Dimension function for the matrix multiplication tensor: all three modes
    have dimension `M * M`. We use a constant function rather than the
    `![M*M, M*M, M*M]` matrix notation to avoid unfolding issues. -/
def matMulDims (M : ℕ) : Fin 3 → ℕ := fun _ => M * M

@[simp]
theorem matMulDims_apply (M : ℕ) (d : Fin 3) : matMulDims M d = M * M := rfl

/-- The matrix multiplication tensor of size `M`: an order-3 tensor with shape
    `(M * M, M * M, M * M)`. Decomposing each mode index as a pair in
    `Fin M × Fin M` via `finProdFinEquiv.symm` (which yields `(l / M, l % M)`),
    with the column-major convention `vec A (i + M * j) = A i j` the row
    index of a flat matrix index is the second component (`= l % M`) and the
    column index is the first component (`= l / M`). Writing
    `p0 = (k_A, i)`, `p1 = (j, k_B)`, `p2 = (j', i')`, the entry is `1` iff
    `i = i'`, `j = j'`, and `k_A = k_B`. -/
def matMulTensor (𝕜 : Type*) [RCLike 𝕜] (M : ℕ) : Tensor 𝕜 3 (matMulDims M) :=
  fun idx =>
    let p0 := finProdFinEquiv.symm (idx 0)
    let p1 := finProdFinEquiv.symm (idx 1)
    let p2 := finProdFinEquiv.symm (idx 2)
    -- Under the column-major `vec`, `.2` is the matrix row and `.1` is the
    -- matrix column. `p0 = (k_A, i)`, `p1 = (j, k_B)`, `p2 = (j', i')`.
    -- Conditions: `i = i'` ↦ `p0.2 = p2.2`, `j = j'` ↦ `p1.1 = p2.1`,
    -- shared inner dim `k_A = k_B` ↦ `p0.1 = p1.2`.
    if p0.2 = p2.2 ∧ p1.1 = p2.1 ∧ p0.1 = p1.2 then (1 : 𝕜) else 0

/-- Evaluation lemma for `matMulTensor`: unfolds the definition to a conditional
    expression on the decomposed indices. -/
@[simp]
theorem matMulTensor_apply (M : ℕ) (idx : (d : Fin 3) → Fin (matMulDims M d)) :
    matMulTensor 𝕜 M idx =
      (if (finProdFinEquiv.symm (idx 0)).2 = (finProdFinEquiv.symm (idx 2)).2 ∧
          (finProdFinEquiv.symm (idx 1)).1 = (finProdFinEquiv.symm (idx 2)).1 ∧
          (finProdFinEquiv.symm (idx 0)).1 = (finProdFinEquiv.symm (idx 1)).2
        then (1 : 𝕜) else 0) := rfl

end Tensor
