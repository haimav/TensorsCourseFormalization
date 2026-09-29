/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.LinearIndependent.Defs
import Mathlib.LinearAlgebra.Matrix.Diagonal
import Mathlib.GroupTheory.Perm.Basic

/-!
# Moore-Penrose pseudoinverse and eigendecomposition uniqueness (axioms)

This file axiomatizes the matrix-theoretic facts that underpin the proof
of the simultaneous-diagonalization uniqueness theorem
(`Tensor.simDiag_uniqueness` in
`TensorCourse/CPDecomposition/SimultaneousDiagonalization.lean`).

Mathlib does not currently expose the Moore-Penrose pseudoinverse at the
matrix level in a form usable for our proof, nor does it expose the
"eigendecomposition is unique up to permutation when the eigenvalues
are distinct" statement at the matrix level. We axiomatize the textbook
facts we need; future work could derive them from a future Mathlib API
(e.g. via `Matrix.LinearMap.range`/`Matrix.adjoint` once those gain a
"left inverse for full-column-rank" lemma).

## What is axiomatized

* `Matrix.pinv` — the Moore-Penrose pseudoinverse. Opaque, with
  properties given by the axioms below.
* `Matrix.pinv_mul_self_of_lin_indep` — for full-column-rank matrices,
  the pseudoinverse is a left inverse.
* `Matrix.mul_pinv_self_of_full_row_lin_indep` — for full-row-rank
  matrices, the pseudoinverse is a right inverse.
* `Matrix.pinv_mul_full_rank` — reverse-order pseudoinverse for
  full-rank factor decompositions of equal rank.
* `Matrix.diagonal_pinv_of_nonzero` — the pseudoinverse of an
  invertible diagonal matrix is the diagonal of pointwise inverses.
* `Matrix.pinv_transpose` — the pseudoinverse commutes with transpose.
* `Matrix.eigendecomp_unique_of_distinct` — eigendecomposition
  uniqueness with distinct eigenvalues.

The naming convention `pinv_mul_full_rank` follows the lecture's
"reverse-order law" terminology.
-/

noncomputable section

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- Moore-Penrose pseudoinverse. Opaque — properties given by axioms below. -/
opaque pinv {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜) :
    Matrix (Fin N) (Fin M) 𝕜

/-- For full-column-rank matrices, the pseudoinverse is a left inverse. -/
axiom pinv_mul_self_of_lin_indep
    {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜)
    (h : LinearIndependent 𝕜 (fun s : Fin N => fun i : Fin M => A i s)) :
    A.pinv * A = 1

/-- For full-row-rank matrices, the pseudoinverse is a right inverse. -/
axiom mul_pinv_self_of_full_row_lin_indep
    {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜)
    (h : LinearIndependent 𝕜 (fun s : Fin M => fun j : Fin N => A s j)) :
    A * A.pinv = 1

/-- Reverse-order pseudoinverse for full-rank factor decompositions of
    equal rank. The lecture explicitly leaves this as a "technical lemma
    we just state". Requires `X : M × r` with full column rank and
    `Y : r × N` with full row rank, both rank `r`. -/
axiom pinv_mul_full_rank
    {M r N : ℕ} (X : Matrix (Fin M) (Fin r) 𝕜) (Y : Matrix (Fin r) (Fin N) 𝕜)
    (hX : LinearIndependent 𝕜 (fun s : Fin r => fun i : Fin M => X i s))
    (hY : LinearIndependent 𝕜 (fun s : Fin r => fun j : Fin N => Y s j)) :
    (X * Y).pinv = Y.pinv * X.pinv

/-- Pseudoinverse of an invertible diagonal matrix is the diagonal of
    pointwise inverses. -/
axiom diagonal_pinv_of_nonzero
    {r : ℕ} (d : Fin r → 𝕜) (h : ∀ s : Fin r, d s ≠ 0) :
    (Matrix.diagonal d).pinv = Matrix.diagonal (fun s => (d s)⁻¹)

/-- The pseudoinverse commutes with transpose. -/
axiom pinv_transpose {M N : ℕ} (A : Matrix (Fin M) (Fin N) 𝕜) :
    (Aᵀ).pinv = (A.pinv)ᵀ

/-- **Eigendecomposition uniqueness with distinct eigenvalues.** If a
    matrix `M` admits two diagonalizing factorizations `U Λ U†` and
    `U' Λ' U'†` with `U, U'` having linearly independent columns and
    `Λ` having pairwise distinct entries, then the columns of `U` and
    `U'` differ by a permutation `π` and per-column scaling, with
    eigenvalues matched accordingly. -/
axiom eigendecomp_unique_of_distinct
    {M r : ℕ}
    (U U' : Matrix (Fin M) (Fin r) 𝕜)
    (Λ Λ' : Fin r → 𝕜)
    (hU : LinearIndependent 𝕜 (fun s : Fin r => fun i : Fin M => U i s))
    (hU' : LinearIndependent 𝕜 (fun s : Fin r => fun i : Fin M => U' i s))
    (hΛ : Function.Injective Λ)
    (heq : U * Matrix.diagonal Λ * U.pinv =
      U' * Matrix.diagonal Λ' * U'.pinv) :
    ∃ π : Equiv.Perm (Fin r),
      (∀ i, Λ i = Λ' (π i)) ∧
      (∀ i, ∃ c : 𝕜, c ≠ 0 ∧ ∀ j, U' j (π i) = c * U j i)

end Matrix

end
