/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TuckerDecomposition.Matricized
import TensorCourse.Tensor.FrobeniusNorm
import TensorCourse.Tensor.Vectorize
import Mathlib.Data.Matrix.Mul

/-!
# Tucker Decomposition: the Optimal Core Theorem

Given fixed factor matrices `U d` with orthonormal columns
(`(U d)ᴴ (U d) = 1`), the **optimal core** minimising the Frobenius
reconstruction error `‖X − ⟦S; U⟧‖` over all cores `S` is the projected
core `Ŝ = X ×-chain Uᴴ = tuckerTensor X (fun d => (U d)ᴴ)`. This is the
lecture's `thm:optimal-core`.

Unlike the lecture, which writes `vec(Ŝ) = (U⁽ᴰ⁾⊗⋯⊗U⁽¹⁾)⁺ vec(X)` using the
pseudoinverse identity `Uᵀ = U⁺` for orthonormal-column `U`, we use the
direct identity `KᴴK = 1` (`kroneckerAll_orthonormalCols`, with
`K = kroneckerAll U`) together with a least-squares Pythagoras. So **the
proof imports no pseudoinverse axiom** and is correct over `ℂ` (where `Uᴴ`,
not `Uᵀ`, is the adjoint).

## Main results

* `Tensor.optimal_core` — the optimal-core minimisation theorem.

## Proof outline

Vectorising (`frobNormSq_vectorize`, `vectorize_tuckerTensor`) turns the
claim into a vector least-squares inequality `‖b − K(Kᴴ b)‖² ≤ ‖b − K z'‖²`
with `b = vec X`, `z' = vec Z`, `K = kroneckerAll U`, and `KᴴK = 1`. The
inequality is the vector Pythagoras: writing `b − K z' = (b − K Kᴴ b) + K(Kᴴ
b − z')`, the cross term vanishes because `⟪v, K w⟫ = ⟪Kᴴ v, w⟫` and
`KᴴK = 1`, so the squared norm splits and dropping the nonnegative second
piece gives the bound.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### A self-contained vector least-squares Pythagoras

We work with the Euclidean structure on `Fin M → 𝕜` directly (sum of
squared entry norms), to match `frobNormSq_vectorize`'s output. -/

section Vector

variable {M N : ℕ}

/-- Squared Euclidean norm of a vector `v : Fin M → 𝕜`. -/
private def vNormSq (v : Fin M → 𝕜) : ℝ := ∑ i, ‖v i‖ ^ 2

/-- Euclidean inner product of vectors (conjugate-linear in the first
    argument), mirroring the tensor inner product `⟪·,·⟫ₜ`. -/
private def vInner (v w : Fin M → 𝕜) : 𝕜 := ∑ i, starRingEnd 𝕜 (v i) * w i

private theorem vNormSq_eq_re_vInner (v : Fin M → 𝕜) :
    vNormSq v = RCLike.re (vInner v v) := by
  rw [vNormSq, vInner, map_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [RCLike.conj_mul, ← RCLike.ofReal_pow, RCLike.ofReal_re]

private theorem vInner_add_right (u v w : Fin M → 𝕜) :
    vInner u (v + w) = vInner u v + vInner u w := by
  simp only [vInner, Pi.add_apply, mul_add, Finset.sum_add_distrib]

private theorem vInner_add_left (u v w : Fin M → 𝕜) :
    vInner (u + v) w = vInner u w + vInner v w := by
  simp only [vInner, Pi.add_apply, map_add, add_mul, Finset.sum_add_distrib]

private theorem vInner_conj_symm (v w : Fin M → 𝕜) :
    vInner v w = starRingEnd 𝕜 (vInner w v) := by
  simp only [vInner, map_sum, map_mul, starRingEnd_self_apply]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  ring

/-- **Vector Pythagoras.** If `v ⟂ w` (`vInner v w = 0`), squared norms add. -/
private theorem vNormSq_add_of_orthogonal {v w : Fin M → 𝕜}
    (h : vInner v w = 0) :
    vNormSq (v + w) = vNormSq v + vNormSq w := by
  rw [vNormSq_eq_re_vInner, vNormSq_eq_re_vInner, vNormSq_eq_re_vInner]
  rw [vInner_add_left, vInner_add_right, vInner_add_right]
  have hwv : vInner w v = 0 := by rw [vInner_conj_symm w v, h, map_zero]
  rw [h, hwv, add_zero, zero_add, map_add]

/-- **Vector adjoint identity:** `⟪u, K w⟫ = ⟪Kᴴ u, w⟫`. Moving the matrix
    across the inner product conjugate-transposes it (a double-sum swap). -/
private theorem vInner_mulVec (K : Matrix (Fin M) (Fin N) 𝕜)
    (u : Fin M → 𝕜) (w : Fin N → 𝕜) :
    vInner u (K *ᵥ w) = vInner (Kᴴ *ᵥ u) w := by
  -- Expand both `mulVec`s pointwise to explicit double sums and swap the order.
  simp only [vInner, Matrix.mulVec, dotProduct, Matrix.conjTranspose_apply,
    map_sum, map_mul, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  refine Finset.sum_congr rfl (fun i _ => ?_)
  -- `conj (star (K i j)) = K i j` (double conjugation).
  rw [← starRingEnd_apply, starRingEnd_self_apply]
  ring

/-- **Vector least-squares optimality.** For `K` with orthonormal columns
    (`Kᴴ K = 1`), the projected solution `K (Kᴴ b)` is the least-squares
    minimiser: `‖b − K(Kᴴ b)‖² ≤ ‖b − K z'‖²` for every `z'`.

    Proof: `b − K z' = (b − K Kᴴ b) + K(Kᴴ b − z')`; the two summands are
    orthogonal (`vInner_mulVec` + `KᴴK = 1`), so by `vNormSq_add_of_orthogonal`
    the squared norm splits, and the second piece is nonnegative. -/
private theorem vNormSq_sub_proj_le (K : Matrix (Fin M) (Fin N) 𝕜)
    (hK : Kᴴ * K = 1) (b : Fin M → 𝕜) (z' : Fin N → 𝕜) :
    vNormSq (b - K *ᵥ (Kᴴ *ᵥ b)) ≤ vNormSq (b - K *ᵥ z') := by
  -- Decompose `b − K z' = r + s` with `r = b − K Kᴴ b`, `s = K (Kᴴ b − z')`.
  set r : Fin M → 𝕜 := b - K *ᵥ (Kᴴ *ᵥ b) with hr
  set s : Fin M → 𝕜 := K *ᵥ (Kᴴ *ᵥ b - z') with hs
  have hsplit : b - K *ᵥ z' = r + s := by
    rw [hr, hs, Matrix.mulVec_sub]
    abel
  -- Orthogonality `vInner r s = 0`.
  have horth : vInner r s = 0 := by
    rw [hs, vInner_mulVec, hr]
    -- `Kᴴ (b − K Kᴴ b) = Kᴴ b − (KᴴK)(Kᴴ b) = Kᴴ b − Kᴴ b = 0`.
    rw [Matrix.mulVec_sub, Matrix.mulVec_mulVec, hK, Matrix.one_mulVec, sub_self]
    -- `vInner 0 _ = 0`.
    simp [vInner]
  rw [hsplit, vNormSq_add_of_orthogonal horth]
  have hs_nonneg : 0 ≤ vNormSq s := Finset.sum_nonneg fun _ _ => sq_nonneg _
  linarith

end Vector

/-! ### The optimal-core theorem -/

/-- `vectorize` commutes with subtraction. -/
private theorem vectorize_sub {D : ℕ} {dims : Fin D → ℕ}
    (T₁ T₂ : Tensor 𝕜 D dims) :
    vectorize (T₁ - T₂) = vectorize T₁ - vectorize T₂ := by
  funext i; rfl

/-- **`thm:optimal-core` (no pseudoinverse axiom).** With factor matrices `U`
    having orthonormal columns, the projected core
    `Ŝ = tuckerTensor X (fun d => (U d)ᴴ)` minimises the squared Frobenius
    reconstruction error over all cores `Z`:

      ‖X − ⟦Ŝ; U⟧‖²  ≤  ‖X − ⟦Z; U⟧‖².

    `[NeZero D]` is **not** required: `vectorize_tuckerTensor` holds at all
    orders (including `D = 0`, where every tensor is a scalar and the bound is
    trivial). -/
theorem optimal_core {D : ℕ} {dims R : Fin D → ℕ}
    (X : Tensor 𝕜 D dims)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (R d)) 𝕜)
    (hU : ∀ d, (U d)ᴴ * (U d) = 1)
    (Z : Tensor 𝕜 D R) :
    (X - tuckerTensor (tuckerTensor X (fun d => (U d)ᴴ)) U).frobNormSq
      ≤ (X - tuckerTensor Z U).frobNormSq := by
  set K : Matrix (Fin (∏ d, dims d)) (Fin (∏ d, R d)) 𝕜 := kroneckerAll U with hKdef
  -- `Kᴴ K = 1`.
  have hKortho : Kᴴ * K = 1 := kroneckerAll_orthonormalCols U hU
  -- Vectorize both errors to the Euclidean norm of the vectorizations.
  rw [frobNormSq_vectorize, frobNormSq_vectorize]
  rw [vectorize_sub, vectorize_sub]
  -- `vec ⟦Z; U⟧ = K *ᵥ vec Z`.
  rw [vectorize_tuckerTensor Z U]
  -- `vec ⟦Ŝ; U⟧ = K *ᵥ vec Ŝ`, and `vec Ŝ = Kᴴ *ᵥ vec X`.
  rw [vectorize_tuckerTensor (tuckerTensor X (fun d => (U d)ᴴ)) U]
  rw [vectorize_tuckerTensor X (fun d => (U d)ᴴ)]
  -- `kroneckerAll (fun d => (U d)ᴴ) = Kᴴ`.
  rw [show kroneckerAll (fun d => (U d)ᴴ) = Kᴴ from by
    rw [hKdef]; exact kroneckerAll_conjTranspose U]
  -- Now the goal is definitionally the vector least-squares inequality (the
  -- sums of squared entry norms are `vNormSq` of the entrywise differences).
  exact vNormSq_sub_proj_le K hKortho (vectorize X) (vectorize Z)

end Tensor

end
