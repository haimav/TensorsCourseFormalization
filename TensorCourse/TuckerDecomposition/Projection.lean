/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.NModeProd
import TensorCourse.Tensor.NModeInner
import TensorCourse.Tensor.FrobeniusNorm
import TensorCourse.Tensor.Cast
import TensorCourse.Matrix.SemiOrthogonal

/-!
# Tucker Decomposition: Mode-n Orthogonal Projections

Fix a factor matrix `U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜` with
orthonormal columns (`Uᴴ U = 1`). The associated **mode-`n` orthogonal
projection** is

    Pₙ Y = Y ×ₙ (U Uᴴ),

and its complement is `Pₙ^⊥ Y = Y − Pₙ Y`. The square factor `U Uᴴ` keeps
the shape `dims` (modulo the `Function.update_eq_self` cast, exactly as in
`nModeProd_one`), so `Pₙ` is an endomorphism of `Tensor 𝕜 D dims`.

This file proves the lecture's "Theoretical Exercise Set 1": `Pₙ` is an
orthogonal projection (idempotent and self-adjoint), `Pₙ` and `Pₙ^⊥` are
mutually orthogonal, the Pythagorean identity `‖Y‖² = ‖Pₙ Y‖² + ‖Pₙ^⊥ Y‖²`
holds, `Pₙ` does not increase the Frobenius norm, and projections on
distinct modes commute.

## Main definitions

* `Tensor.projMode n U Y = Y ×ₙ (U Uᴴ)` — the mode-`n` projection.
* `Tensor.projModePerp n U Y = Y − projMode n U Y` — its complement.

## Main results

* `Tensor.modeUnfold_projMode` — `(projMode n U Y).modeUnfold n = (U Uᴴ) * Y.modeUnfold n`
  (no submatrix reindex; the workhorse rewrite lemma).
* `Tensor.projMode_idempotent` — `Pₙ (Pₙ Y) = Pₙ Y`.
* `Tensor.projMode_self_adjoint` — `⟪Pₙ Y, Z⟫ₜ = ⟪Y, Pₙ Z⟫ₜ`.
* `Tensor.projModePerp_dotProd_eq_zero` — `⟪Pₙ Y, Pₙ^⊥ Y⟫ₜ = 0`.
* `Tensor.frobNormSq_projMode_add_projModePerp` — Pythagoras:
  `‖Y‖² = ‖Pₙ Y‖² + ‖Pₙ^⊥ Y‖²`.
* `Tensor.frobNormSq_projMode_le` — `‖Pₙ Y‖² ≤ ‖Y‖²`.
* `Tensor.projMode_projMode_comm` — projections on distinct modes commute.

## Transpose vs. conjugate transpose

The projector is `U Uᴴ` (conjugate transpose), consistent with the inner
product `⟪·,·⟫ₜ`, which conjugates its first argument; over `ℝ`, `ᴴ = ᵀ`,
recovering the lecture's `U Uᵀ`.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ} {R : Fin D → ℕ}

/-! ### Definition -/

/-- **Mode-`n` orthogonal projection** `Pₙ Y = Y ×ₙ (U Uᴴ)`. The square
    factor `U Uᴴ : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜` keeps the shape
    `dims`; the `cast` collapses the `Function.update dims n (dims n)` output
    shape back to `dims` (the `Function.update_eq_self` pattern of
    `nModeProd_one`). -/
def projMode (n : Fin D) (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (Y : Tensor 𝕜 D dims) : Tensor 𝕜 D dims :=
  cast (congrArg (Tensor 𝕜 D) (Function.update_eq_self n dims))
    (Y.nModeProd n (U * Uᴴ))

/-- **Complementary projection** `Pₙ^⊥ Y = Y − Pₙ Y`. -/
def projModePerp (n : Fin D) (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (Y : Tensor 𝕜 D dims) : Tensor 𝕜 D dims :=
  Y - projMode n U Y

theorem projModePerp_def (n : Fin D) (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (Y : Tensor 𝕜 D dims) :
    projModePerp n U Y = Y - projMode n U Y := rfl

/-- `Y = Pₙ Y + Pₙ^⊥ Y`. -/
theorem projMode_add_projModePerp (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y : Tensor 𝕜 D dims) :
    projMode n U Y + projModePerp n U Y = Y := by
  rw [projModePerp_def]; ring

/-! ### Matricization of the projection -/

/-- **Matricization of the mode-`n` projection.** The mode-`n` unfolding of
    `Pₙ Y` is the matrix product `(U Uᴴ) * Y.modeUnfold n` — no submatrix
    reindex on any axis. This is the workhorse from which idempotence and the
    norm bound follow by matrix algebra. -/
theorem modeUnfold_projMode (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y : Tensor 𝕜 D dims) :
    (projMode n U Y).modeUnfold n = (U * Uᴴ) * Y.modeUnfold n := by
  funext i j
  -- LHS: expand the unfolding entry, push the `cast` through, expand `nModeProd`.
  rw [modeUnfold_apply, projMode]
  rw [cast_tensor_apply (h := Function.update_eq_self n dims), nModeProd_apply]
  -- RHS: expand the matrix product.
  rw [Matrix.mul_apply]
  -- Both are sums over `Fin (dims n)`; match termwise.
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  -- The `U Uᴴ`-entry: row index reduces to `i` (its `.val` is `i.val`).
  have hrow : ((((separateModeEquiv n dims).symm
        (i, (remainingModesEquiv n dims).symm j)) n).cast
        (congrFun (Function.update_eq_self n dims) n).symm).cast
        (Function.update_self n (dims n) dims) = i := by
    apply Fin.ext
    simp only [Fin.val_cast]
    -- `(separateModeEquiv …).symm (i, _) n = i` at the `.val` level.
    rw [separateModeEquiv]
    simp
  rw [hrow]
  congr 1
  -- The `Y`-argument: at mode `n` it is `iₙ`; off mode `n` it matches the
  -- remaining-modes part of the unfolding's column index `j`.
  rw [modeUnfold_apply]
  congr 1
  funext d
  by_cases h : d = n
  · subst h
    simp only [↓reduceDIte]
    rw [separateModeEquiv]
    simp
  · simp only [h, ↓reduceDIte]
    apply Fin.ext
    simp only [Fin.val_cast]
    -- Off-mode `d`: both reduce to the `d`-component of
    -- `(remainingModesEquiv n dims).symm j`.
    rw [separateModeEquiv]
    simp [h]

/-! ### Pointwise formula -/

/-- **Pointwise formula for the mode-`n` projection** (cast-free). Evaluating
    `Pₙ W` at a multi-index `idx` contracts the mode-`n` coordinate against
    the projector `U Uᴴ`, leaving the other coordinates fixed:

      (Pₙ W) idx = ∑ k, (U Uᴴ) (idx n) k * W (Function.update idx n k).

    Here `Function.update idx n k` replaces `idx`'s mode-`n` coordinate by
    `k : Fin (dims n)`; no `Fin.cast` is needed because the projector is
    square (`Fin (dims n) × Fin (dims n)`). -/
theorem projMode_apply (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (W : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims d)) :
    projMode n U W idx =
      ∑ k : Fin (dims n), (U * Uᴴ) (idx n) k * W (Function.update idx n k) := by
  rw [projMode, cast_tensor_apply (h := Function.update_eq_self n dims),
    nModeProd_apply]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  -- The `(U Uᴴ)`-row index reduces to `idx n` (two `.val`-preserving casts).
  have hrow : (((idx n).cast (congrFun (Function.update_eq_self n dims) n).symm).cast
      (Function.update_self n (dims n) dims)) = idx n := by
    apply Fin.ext; simp
  rw [hrow]
  refine congrArg₂ (· * ·) rfl ?_
  -- The `W`-argument equals `Function.update idx n k` pointwise.
  refine congrArg W (funext fun d => ?_)
  by_cases h : d = n
  · subst h; simp
  · simp only [h, ↓reduceDIte, Function.update_of_ne h]
    apply Fin.ext; simp

/-! ### Idempotence -/

/-- **`Pₙ` is idempotent:** `Pₙ (Pₙ Y) = Pₙ Y`. Proved by matricization:
    `(Pₙ (Pₙ Y))_(n) = (U Uᴴ)(U Uᴴ) Y_(n) = (U Uᴴ) Y_(n) = (Pₙ Y)_(n)`
    via `projector_idem`, then `modeUnfold_injective`. -/
theorem projMode_idempotent (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    projMode n U (projMode n U Y) = projMode n U Y := by
  apply modeUnfold_injective n
  rw [modeUnfold_projMode, modeUnfold_projMode, ← Matrix.mul_assoc,
    Matrix.projector_idem U hU]

/-! ### Self-adjointness -/

/-- **`Pₙ` is self-adjoint:** `⟪Pₙ Y, Z⟫ₜ = ⟪Y, Pₙ Z⟫ₜ`.

    The adjoint of mode-`n` multiplication by `U Uᴴ` is mode-`n`
    multiplication by `(U Uᴴ)ᴴ = U Uᴴ` (`projector_conjTranspose`); the
    `nModeProdAdjCast` cast and the `projMode` cast both preserve `.val`, so
    they cancel pointwise. -/
theorem projMode_self_adjoint (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y Z : Tensor 𝕜 D dims) :
    ⟪projMode n U Y, Z⟫ₜ = ⟪Y, projMode n U Z⟫ₜ := by
  rw [dotProd_def, dotProd_def]
  -- Expand each `projMode` via the cast-free pointwise formula, then push the
  -- scalars in so each side is a single sum over the pair `(idx, k)`.
  simp only [projMode_apply]
  conv_lhs =>
    rw [show (fun idx => starRingEnd 𝕜
          (∑ k : Fin (dims n), (U * Uᴴ) (idx n) k * Y (Function.update idx n k)) * Z idx) =
        (fun idx => ∑ k : Fin (dims n),
          starRingEnd 𝕜 ((U * Uᴴ) (idx n) k) *
            (starRingEnd 𝕜 (Y (Function.update idx n k)) * Z idx))
      from funext fun idx => by
        rw [map_sum, Finset.sum_mul]
        refine Finset.sum_congr rfl (fun k _ => ?_)
        rw [map_mul, mul_assoc]]
  conv_rhs =>
    rw [show (fun idx => starRingEnd 𝕜 (Y idx) *
          ∑ k : Fin (dims n), (U * Uᴴ) (idx n) k * Z (Function.update idx n k)) =
        (fun idx => ∑ k : Fin (dims n),
          (U * Uᴴ) (idx n) k * (starRingEnd 𝕜 (Y idx) * Z (Function.update idx n k)))
      from funext fun idx => by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl (fun k _ => ?_)
        ring]
  -- Both sides are sums over the pair `(idx, k)`. Reindex via the self-inverse
  -- swap of the mode-`n` coordinate `(idx, k) ↦ (Function.update idx n k, idx n)`.
  rw [← Finset.sum_product', ← Finset.sum_product']
  apply Finset.sum_nbij'
    (fun p => (Function.update p.1 n p.2, p.1 n))
    (fun q => (Function.update q.1 n q.2, q.1 n))
  · intro p _; exact Finset.mem_univ _
  · intro q _; exact Finset.mem_univ _
  · rintro ⟨idx, k⟩ _
    apply Prod.ext
    · simp
    · simp
  · rintro ⟨idx, k⟩ _
    apply Prod.ext
    · simp
    · simp
  · -- Summand equality with `idx' = update idx n k`, so `idx' n = k`, and
    -- `update idx' n (idx n) = idx`. The conjugate transpose flips `U Uᴴ`.
    rintro ⟨idx, k⟩ _
    dsimp only
    -- `conj ((U Uᴴ) (idx n) k) = (U Uᴴ) k (idx n)` via `projector_conjTranspose`.
    have hUU : starRingEnd 𝕜 ((U * Uᴴ) (idx n) k) = (U * Uᴴ) k (idx n) := by
      have h2 := congrFun (congrFun (Matrix.projector_conjTranspose U) k) (idx n)
      rwa [Matrix.conjTranspose_apply] at h2
    rw [hUU, Function.update_self]
    -- `update (update idx n k) n (idx n) = idx`.
    rw [Function.update_idem, Function.update_eq_self]

/-! ### Orthogonality of `Pₙ` and `Pₙ^⊥` -/

/-- **`Pₙ Y` and `Pₙ^⊥ Y` are orthogonal:** `⟪Pₙ Y, Pₙ^⊥ Y⟫ₜ = 0`.
    From self-adjointness and idempotence:
    `⟪Pₙ Y, Y − Pₙ Y⟫ = ⟪Pₙ Y, Y⟫ − ⟪Pₙ Y, Pₙ Y⟫`, and
    `⟪Pₙ Y, Pₙ Y⟫ = ⟪Pₙ(Pₙ Y), Y⟫ = ⟪Pₙ Y, Y⟫` (self-adjoint then idempotent). -/
theorem projModePerp_dotProd_eq_zero (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    ⟪projMode n U Y, projModePerp n U Y⟫ₜ = 0 := by
  -- `⟪P, Y − P⟫ = ⟪P, Y⟫ − ⟪P, P⟫` (additivity in arg 2 + the `(-1)•` rewrite).
  have hsub : ⟪projMode n U Y, projModePerp n U Y⟫ₜ
      = ⟪projMode n U Y, Y⟫ₜ - ⟪projMode n U Y, projMode n U Y⟫ₜ := by
    rw [projModePerp_def, sub_eq_add_neg, show -projMode n U Y = (-1 : 𝕜) • projMode n U Y from by
      rw [neg_one_smul], dotProd_add_right, dotProd_smul_right]
    ring
  rw [hsub]
  -- `⟪Pₙ Y, Pₙ Y⟫ = ⟪Pₙ (Pₙ Y), Y⟫ = ⟪Pₙ Y, Y⟫` (self-adjoint then idempotent).
  rw [← projMode_self_adjoint n U (projMode n U Y) Y, projMode_idempotent n U hU Y]
  ring

/-! ### Pythagoras and the norm bound -/

/-- **Pythagorean identity:** `‖Y‖² = ‖Pₙ Y‖² + ‖Pₙ^⊥ Y‖²`. Since
    `Y = Pₙ Y + Pₙ^⊥ Y` and the two summands are orthogonal
    (`projModePerp_dotProd_eq_zero`), this is `frobNormSq_add_of_orthogonal`. -/
theorem frobNormSq_projMode_add_projModePerp (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    ‖Y‖_F² = ‖projMode n U Y‖_F² + ‖projModePerp n U Y‖_F² := by
  conv_lhs => rw [← projMode_add_projModePerp n U Y]
  exact frobNormSq_add_of_orthogonal (projModePerp_dotProd_eq_zero n U hU Y)

/-- **The projection does not increase the Frobenius norm:**
    `‖Pₙ Y‖² ≤ ‖Y‖²`. Immediate from Pythagoras and nonnegativity of
    `‖Pₙ^⊥ Y‖²`. -/
theorem frobNormSq_projMode_le (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    ‖projMode n U Y‖_F² ≤ ‖Y‖_F² := by
  rw [frobNormSq_projMode_add_projModePerp n U hU Y]
  have := frobNormSq_nonneg (projModePerp n U Y)
  linarith

/-! ### Different-mode commutativity -/

/-- **Projections on distinct modes commute:**
    `Pₘ (Pₙ Y) = Pₙ (Pₘ Y)` for `m ≠ n`.

    Both `projMode`s are `nModeProd`s composed with `Function.update_eq_self`
    casts; the underlying `nModeProd`s commute by `nModeProd_nModeProd_comm`.
    Proved pointwise via `projMode_apply`-style expansion, matching the two
    iterated sums termwise. -/
theorem projMode_projMode_comm {m n : Fin D} (hmn : m ≠ n)
    (Um : Matrix (Fin (dims m)) (Fin (R m)) 𝕜)
    (Un : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y : Tensor 𝕜 D dims) :
    projMode m Um (projMode n Un Y) = projMode n Un (projMode m Um Y) := by
  funext idx
  -- Expand both nested projections via the cast-free pointwise formula.
  simp only [projMode_apply]
  -- LHS: `∑ iₘ, (Um Umᴴ)(idx m) iₘ * ∑ iₙ, (Un Unᴴ)((update idx m iₘ) n) iₙ * Y arg`.
  -- RHS: `∑ iₙ, (Un Unᴴ)(idx n) iₙ * ∑ iₘ, (Um Umᴴ)((update idx n iₙ) m) iₘ * Y arg`.
  -- Off the updated mode the row indices simplify (`m ≠ n`); push the outer
  -- factor in and swap the LHS sums to match the RHS order.
  simp only [Function.update_of_ne hmn, Function.update_of_ne (Ne.symm hmn),
    Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  refine Finset.sum_congr rfl (fun iₘ _ => ?_)
  -- Reorder scalars so both sides are `(Um..) * ((Un..) * Y arg)`.
  rw [show ∀ a b c : 𝕜, a * (b * c) = b * (a * c) from fun a b c => by ring]
  refine congrArg₂ (· * ·) rfl (congrArg₂ (· * ·) rfl ?_)
  -- The two `Y`-arguments agree by `Function.update_comm` (`m ≠ n`).
  rw [Function.update_comm hmn]

end Tensor

end
