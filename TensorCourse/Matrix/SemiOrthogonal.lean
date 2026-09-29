/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Kronecker
import TensorCourse.Matrix.KhatriRao

/-!
# Semi-orthogonal (orthonormal-columns) matrices

A matrix `U : Matrix (Fin I) (Fin R) 𝕜` has *orthonormal columns* when
`Uᴴ * U = 1`. This is the lecture's "orthogonal factor matrix" condition
on the Tucker / HOSVD factors. Over `ℝ`, `ᴴ = ᵀ`, so the predicate
specialises to the lecture's `Uᵀ U = I`.

We use the conjugate transpose `ᴴ` (not the plain transpose `ᵀ`)
throughout, consistent with the tensor inner product `⟪·,·⟫ₜ`, which
conjugates its first argument: the adjoint of mode-`n` multiplication is
`ᴴ` and the orthogonal projector is `U Uᴴ`, both of which require
`Uᴴ U = 1` to be idempotent / norm-preserving over `ℂ`.

## Main definitions

* `Matrix.HasOrthonormalCols U` — the proposition `Uᴴ * U = 1`.

## Main results

* `Matrix.projector_idem` — `U Uᴴ` is idempotent: `(U Uᴴ)(U Uᴴ) = U Uᴴ`.
* `Matrix.projector_conjTranspose` — `U Uᴴ` is self-adjoint:
  `(U Uᴴ)ᴴ = U Uᴴ`.
* `Matrix.kronecker_orthonormalCols` — the Kronecker product of two
  orthonormal-column matrices has orthonormal columns.
-/

noncomputable section

open scoped Matrix

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- A matrix has *orthonormal columns* iff `Uᴴ * U = 1`. The lecture's
    "orthogonal factor matrix". Over `ℝ`, `ᴴ = ᵀ`, matching the lecture. -/
def HasOrthonormalCols {I R : ℕ} (U : Matrix (Fin I) (Fin R) 𝕜) : Prop :=
  Uᴴ * U = 1

/-- `HasOrthonormalCols U` unfolds to `Uᴴ * U = 1` (definitional). -/
theorem hasOrthonormalCols_iff {I R : ℕ} (U : Matrix (Fin I) (Fin R) 𝕜) :
    HasOrthonormalCols U ↔ Uᴴ * U = 1 := Iff.rfl

/-- The projector `U Uᴴ` associated to an orthonormal-column matrix is
    idempotent: `(U Uᴴ) * (U Uᴴ) = U Uᴴ`. -/
theorem projector_idem {I R : ℕ} (U : Matrix (Fin I) (Fin R) 𝕜)
    (hU : HasOrthonormalCols U) :
    (U * Uᴴ) * (U * Uᴴ) = U * Uᴴ := by
  have hU' : Uᴴ * U = 1 := hU
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ U Uᴴ, hU', Matrix.one_mul]

/-- The projector `U Uᴴ` is self-adjoint as a matrix: `(U Uᴴ)ᴴ = U Uᴴ`. -/
theorem projector_conjTranspose {I R : ℕ} (U : Matrix (Fin I) (Fin R) 𝕜) :
    (U * Uᴴ)ᴴ = U * Uᴴ := by
  rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]

/-- The Kronecker product of two orthonormal-column matrices has
    orthonormal columns: if `Uᴴ U = 1` and `Vᴴ V = 1` then
    `(U ⊗ₜ V)ᴴ (U ⊗ₜ V) = 1`. -/
theorem kronecker_orthonormalCols {I R I' R' : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜) (V : Matrix (Fin I') (Fin R') 𝕜)
    (hU : HasOrthonormalCols U) (hV : HasOrthonormalCols V) :
    (U ⊗ₜ V)ᴴ * (U ⊗ₜ V) = 1 := by
  have hU' : Uᴴ * U = 1 := hU
  have hV' : Vᴴ * V = 1 := hV
  rw [Matrix.conjTranspose_kronecker, ← Matrix.mul_kronecker_mul, hU', hV',
    Matrix.one_kronecker_one]

end Matrix
