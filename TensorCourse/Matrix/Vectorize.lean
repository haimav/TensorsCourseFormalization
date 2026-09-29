/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Matrix Vectorization

This file defines a vectorization of a (rectangular) matrix into a vector,
following the Kolda-Bader column-major convention used throughout the
lecture notes: the row index varies fastest, the column index varies
slowest. Concretely, the entry `A i j` of a matrix
`A : Matrix (Fin M) (Fin N) 𝕜` lands at flat index `i + M * j`, i.e.
`Matrix.vectorize A (i + M * j) = A i j`.

## Main definitions

* `Matrix.vectorize` — flatten a matrix `A : Matrix (Fin M) (Fin N) 𝕜` into
  a vector `Fin (M * N) → 𝕜` in **column-major order** (row varies fastest):
  `Matrix.vectorize A l = A (l % M) (l / M)`.
  Implementation: cast `l : Fin (M * N)` to `Fin (N * M)` via `Nat.mul_comm`
  and take `finProdFinEquiv.symm`, which gives the pair `(l / M, l % M)` in
  `Fin N × Fin M`. The row index is then the second component and the column
  index is the first component.
-/

noncomputable section

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- Vectorization of a matrix in **column-major order** (Kolda-Bader,
    row-fastest): `A i j` lands at flat index `i + M * j`, so
    `Matrix.vectorize A l = A (l % M) (l / M)`.

    Implementation: cast `l : Fin (M * N)` to `Fin (N * M)` via
    `Nat.mul_comm`, then apply `finProdFinEquiv.symm : Fin (N * M) ≃ Fin N × Fin M`
    (which yields `(l / M, l % M)`). The row index `l % M : Fin M` comes
    from the second component of the pair; the column index `l / M : Fin N`
    from the first. -/
def vectorize {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜) : Fin (M * N) → 𝕜 :=
  fun l =>
    A (finProdFinEquiv.symm (l.cast (Nat.mul_comm M N))).2
      (finProdFinEquiv.symm (l.cast (Nat.mul_comm M N))).1

omit [RCLike 𝕜] in
@[simp]
theorem vectorize_apply {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜) (l : Fin (M * N)) :
    vectorize A l =
      A (finProdFinEquiv.symm (l.cast (Nat.mul_comm M N))).2
        (finProdFinEquiv.symm (l.cast (Nat.mul_comm M N))).1 := rfl

end Matrix
