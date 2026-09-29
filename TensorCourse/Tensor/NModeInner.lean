/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.NModeProd
import TensorCourse.Tensor.Cast
import TensorCourse.Tensor.InnerProd
import TensorCourse.Tensor.FrobeniusNorm
import TensorCourse.Matrix.SemiOrthogonal

/-!
# n-Mode Product and the Inner Product (adjoint identity)

The n-mode product interacts with the tensor inner product `⟪·,·⟫ₜ`
through an **adjoint identity** (the lecture's "Theoretical Exercise
Set 1"): moving an n-mode factor across the inner product replaces it by
its conjugate transpose,

    ⟪Z, Y ×ₙ A⟫ₜ = ⟪Z ×ₙ Aᴴ, Y⟫ₜ

(modulo a `Function.update_idem`/`update_eq_self` cast bridging the type of
`Z ×ₙ Aᴴ` back to the shape of `Y`). As an immediate corollary, mode-`n`
multiplication by a matrix with orthonormal columns preserves the
Frobenius norm.

## Main results

* `Tensor.dotProd_nModeProd_right` — the n-mode adjoint identity.
* `Tensor.frobNormSq_nModeProd_of_orthonormal` — norm preservation under a
  semi-orthogonal factor (`Matrix.HasOrthonormalCols`).
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-! ### The n-mode adjoint identity -/

/-- The composite cast `Function.update (Function.update dims n J) n (dims n) = dims`,
    obtained by collapsing the doubled update (`update_idem`) and then
    `update_eq_self`. -/
theorem updateUpdate_eq (n : Fin D) (J : ℕ) :
    Function.update (Function.update dims n J) n (dims n) = dims :=
  (Function.update_idem J (dims n) dims).trans (Function.update_eq_self n dims)

/-- `Aᴴ`, reindexed so its columns are typed as `Fin ((update dims n J) n)`
    (definitionally `Fin J`), making it a valid mode-`n` factor for a tensor
    of shape `update dims n J`. -/
def adjFactor {J : ℕ} (n : Fin D) (A : Matrix (Fin J) (Fin (dims n)) 𝕜) :
    Matrix (Fin (dims n)) (Fin ((Function.update dims n J) n)) 𝕜 :=
  Aᴴ.submatrix id (fun j => j.cast (Function.update_self n J dims))

/-- Transport `Z ×ₙ Aᴴ` back to the shape `dims` of `Y` (collapsing the
    doubled `Function.update`). -/
def nModeProdAdjCast {J : ℕ} (n : Fin D)
    (A : Matrix (Fin J) (Fin (dims n)) 𝕜)
    (Z : Tensor 𝕜 D (Function.update dims n J)) : Tensor 𝕜 D dims :=
  cast (congrArg (Tensor 𝕜 D) (updateUpdate_eq n J)) (Z.nModeProd n (adjFactor n A))

/-- **n-mode adjoint identity** (lecture's "Theoretical Exercise Set 1").
    Moving an n-mode factor across the inner product conjugate-transposes
    it. The cast `nModeProdAdjCast` transports `Z ×ₙ Aᴴ`'s type back to the
    shape `dims` of `Y` (collapsing the doubled `Function.update`). -/
theorem dotProd_nModeProd_right {J : ℕ} (n : Fin D)
    (A : Matrix (Fin J) (Fin (dims n)) 𝕜)
    (Y : Tensor 𝕜 D dims) (Z : Tensor 𝕜 D (Function.update dims n J)) :
    ⟪Z, Y.nModeProd n A⟫ₜ = ⟪nModeProdAdjCast n A Z, Y⟫ₜ := by
  rw [dotProd_def, dotProd_def]
  -- Expand the LHS: `Y ×ₙ A` and pull `conj (Z outIdx)` into the inner sum.
  conv_lhs =>
    rw [show (fun outIdx => starRingEnd 𝕜 (Z outIdx) * (Y.nModeProd n A) outIdx) =
          (fun outIdx => ∑ iₙ : Fin (dims n),
            starRingEnd 𝕜 (Z outIdx) *
              (A ((outIdx n).cast (Function.update_self n J dims)) iₙ *
                Y (fun d => if h : d = n then h ▸ iₙ
                  else (outIdx d).cast (by rw [Function.update_of_ne h]))))
        from funext fun outIdx => by rw [nModeProd_apply, Finset.mul_sum]]
  -- Expand the RHS analogously, rewriting `Aᴴ`'s entry as `star (A · ·)`.
  -- The contraction variable `jₙ` lives in `Fin ((update dims n J) n)` (the
  -- input mode of `Z`); its cast to `Fin J` is `A`'s row index.
  conv_rhs =>
    rw [show (fun idx => starRingEnd 𝕜 (nModeProdAdjCast n A Z idx) * Y idx) =
          (fun idx => ∑ jₙ : Fin ((Function.update dims n J) n),
            A (jₙ.cast (Function.update_self n J dims)) (idx n) *
              (starRingEnd 𝕜
                  (Z (fun d => if h : d = n then h ▸ jₙ
                    else (idx d).cast (by rw [Function.update_of_ne h]))) * Y idx))
        from funext fun idx => by
          rw [nModeProdAdjCast, cast_tensor_apply (h := updateUpdate_eq n J),
            nModeProd_apply, map_sum, Finset.sum_mul]
          refine Finset.sum_congr rfl fun jₙ _ => ?_
          rw [map_mul]
          simp only [adjFactor, Matrix.submatrix_apply, id_eq,
            Matrix.conjTranspose_apply, starRingEnd_apply, star_star]
          rw [mul_assoc]
          -- Match the `A`-factor and the `conj(Z) * Y` factor separately.
          refine congrArg₂ (· * ·) ?_ ?_
          · -- `A`-entry: row and column indices agree (equal `.val`).
            refine congrArg₂ A ?_ ?_ <;> (apply Fin.ext; simp [Fin.val_cast])
          · -- `conj (Z arg₁) * Y idx`: only the `Z`-argument needs matching.
            refine congrArg₂ (· * ·) ?_ rfl
            refine congrArg (starRingEnd 𝕜) ?_
            refine congrArg Z ?_
            funext d
            apply Fin.ext
            by_cases h : d = n
            · subst h; simp
            · simp only [h, ↓reduceDIte, Fin.val_cast]]
  -- Both sides are now double sums; reindex via the mode-`n` splitting
  -- bijection `(outIdx, iₙ) ↦ (idx := update outIdx n iₙ, jₙ := iₙ)`.
  rw [← Finset.sum_product', ← Finset.sum_product']
  apply Finset.sum_nbij'
    (fun p =>
      ((fun d => if h : d = n then h ▸ p.2
          else (p.1 d).cast (by rw [Function.update_of_ne h])),
        p.1 n))
    (fun q =>
      ((fun d => if h : d = n then h ▸ q.2
          else (q.1 d).cast (by rw [Function.update_of_ne h])),
        q.1 n))
  · intro p _; exact Finset.mem_univ _
  · intro q _; exact Finset.mem_univ _
  · -- left inverse
    rintro ⟨outIdx, iₙ⟩ _
    apply Prod.ext
    · funext d
      by_cases h : d = n
      · subst h; simp
      · simp only [h, ↓reduceDIte]; apply Fin.ext; simp
    · apply Fin.ext; simp
  · -- right inverse
    rintro ⟨idx, jₙ⟩ _
    apply Prod.ext
    · funext d
      by_cases h : d = n
      · subst h; simp
      · simp only [h, ↓reduceDIte]; apply Fin.ext; simp
    · apply Fin.ext; simp
  · -- summand equality: `conj(Z outIdx) * (A r iₙ * Y arg)` equals
    -- `A r ((idx) n) * (conj(Z arg') * Y idx)` after reassociation, with
    -- `idx = update outIdx n iₙ`, `arg' = update idx n (outIdx n) = outIdx`.
    rintro ⟨outIdx, iₙ⟩ _
    dsimp only
    rw [show ∀ a b c : 𝕜, a * (b * c) = b * (a * c) from fun a b c => by ring]
    refine congrArg₂ (· * ·) ?_ (congrArg₂ (· * ·) ?_ ?_)
    · -- `A`-entry: row `(outIdx n).cast` matches, column `iₙ = (update outIdx n iₙ) n`.
      refine congrArg₂ A ?_ ?_
      · apply Fin.ext; simp
      · apply Fin.ext; simp
    · -- `conj (Z outIdx) = conj (Z (update (update outIdx n iₙ) n (outIdx n)))`.
      refine congrArg (starRingEnd 𝕜) (congrArg Z (funext fun d => ?_))
      apply Fin.ext
      by_cases h : d = n
      · subst h; simp
      · simp only [h, ↓reduceDIte, Fin.val_cast]
    · -- `Y (update outIdx n iₙ) = Y (update outIdx n iₙ)`.
      rfl

/-- For an orthonormal-column matrix `U` (`Uᴴ U = 1`), applying the adjoint
    factor `Uᴴ` after `×ₙ U` recovers the original tensor:
    `nModeProdAdjCast n U (Y ×ₙ U) = Y`. The `nModeProdAdjCast` wrapper
    applies the reindexed adjoint factor (`adjFactor n U`, i.e. `Uᴴ`
    `submatrix`-reindexed so its column type matches mode `n` after the
    first `×ₙ U`) *and* collapses the doubled `Function.update` back to the
    original shape `dims`. -/
theorem nModeProdAdjCast_nModeProd_self {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    nModeProdAdjCast n U (Y.nModeProd n U) = Y := by
  have hU' : Uᴴ * U = 1 := hU
  funext idx
  -- Pointwise: collapse `((Y ×ₙ U) ×ₙ Uᴴ) = Y ×ₙ (Uᴴ U) = Y ×ₙ 1 = Y`.
  rw [nModeProdAdjCast, cast_tensor_apply (h := updateUpdate_eq n J)]
  -- `(Y ×ₙ U) ×ₙ adjFactor = update_idem ▸ (Y ×ₙ (Uᴴ * U))` (same-mode comp).
  have hcomp := nModeProd_nModeProd_same n U Uᴴ Y
  -- `adjFactor n U` is exactly the reindexed `Uᴴ` appearing in `hcomp`.
  rw [show adjFactor n U =
        Uᴴ.submatrix id (fun j : Fin ((Function.update dims n J) n) =>
          j.cast (Function.update_self n J dims)) from rfl]
  rw [hcomp, hU']
  -- `Y ×ₙ (1) = cast Y` (n-mode product with the identity).
  rw [nModeProd_one]
  -- Now both sides are `Y` at an index, modulo composed `.val`-preserving casts.
  rw [cast_apply, cast_tensor_apply (h := (Function.update_eq_self n dims).symm)]
  refine congrArg Y ?_
  funext d
  apply Fin.ext
  simp [Fin.val_cast, cast_idx_val]

/-! ### Norm preservation under a semi-orthogonal factor -/

/-- If `U` has orthonormal columns (`Uᴴ U = 1`), mode-`n` multiplication
    preserves the squared Frobenius norm. -/
theorem frobNormSq_nModeProd_of_orthonormal {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    (Y.nModeProd n U).frobNormSq = Y.frobNormSq := by
  rw [frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd]
  -- Adjoint identity with `Z := Y ×ₙ U`, then collapse `Uᴴ U = 1`.
  rw [dotProd_nModeProd_right n U Y (Y.nModeProd n U),
    nModeProdAdjCast_nModeProd_self n U hU Y]

end Tensor
