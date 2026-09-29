/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.OuterProd
import TensorCourse.Tensor.Vectorize
import TensorCourse.Tensor.ModeUnfold
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Mode Splits and Split Ranks

The **mode-`j` split** `X_{[j]}` of a tensor groups the first `j` modes into
the rows of a matrix and the remaining modes into the columns
(Lecture 6, §"Split Ranks and Tensor Train Decompositions"):

  `X_{[j]} = X_{((1,…,j),(j+1,…,D))}`.

We parametrize the split point through the operand's *shape*: a tensor with
shape `combinedDims I₁ I₂` (order `p + q`) splits canonically after its first
`p` modes. This reuses `OuterProd`'s `combinedDims` (`Fin.addCases`),
`splitLeft`/`splitRight`, and the Kolda–Bader flattening `multiIndexEquiv`
on each block, so the split lands natively in
`Matrix (Fin (∏ I₁)) (Fin (∏ I₂)) 𝕜` with no `submatrix` reindex.

The **split rank** is `rank (X_{[j]})`, the quantity that governs the TT
decomposition (analogue of the n-rank for Tucker).

## Main definitions

* `Tensor.combineIndex` — build a combined multi-index from a left and a
  right multi-index (inverse of `splitLeft`/`splitRight`).
* `Tensor.modeSplit` — the mode split as a matrix.
* `Tensor.splitRank` — `rank (modeSplit X)`.

## Main results

* `Tensor.modeSplit_apply` — closed-form pointwise formula.
* `Tensor.splitLeft_combineIndex`, `Tensor.splitRight_combineIndex`,
  `Tensor.combineIndex_splitLeft_splitRight` — round-trips.
* `Tensor.modeSplit_one_eq_modeUnfold_zero` — splitting after the first mode
  is the mode-0 unfolding (sanity check).
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {p q : ℕ} {I₁ : Fin p → ℕ} {I₂ : Fin q → ℕ}

/-! ### Combining two multi-indices -/

/-- Build a combined multi-index over `combinedDims I₁ I₂` from a left
    multi-index (over `I₁`) and a right multi-index (over `I₂`). This is the
    inverse of the `splitLeft`/`splitRight` pair. -/
def combineIndex (f₁ : (d : Fin p) → Fin (I₁ d)) (f₂ : (d : Fin q) → Fin (I₂ d)) :
    (d : Fin (p + q)) → Fin (combinedDims I₁ I₂ d) :=
  Fin.addCases
    (motive := fun d => Fin (combinedDims I₁ I₂ d))
    (fun d => Fin.cast (combinedDims_castAdd d).symm (f₁ d))
    (fun d => Fin.cast (combinedDims_natAdd d).symm (f₂ d))

@[simp]
theorem splitLeft_combineIndex (f₁ : (d : Fin p) → Fin (I₁ d))
    (f₂ : (d : Fin q) → Fin (I₂ d)) :
    splitLeft (combineIndex f₁ f₂) = f₁ := by
  funext d
  simp only [splitLeft, combineIndex, Fin.addCases_left]
  -- both sides are `f₁ d` up to the two cancelling casts
  apply Fin.ext
  simp

@[simp]
theorem splitRight_combineIndex (f₁ : (d : Fin p) → Fin (I₁ d))
    (f₂ : (d : Fin q) → Fin (I₂ d)) :
    splitRight (combineIndex f₁ f₂) = f₂ := by
  funext d
  simp only [splitRight, combineIndex, Fin.addCases_right]
  apply Fin.ext
  simp

theorem combineIndex_splitLeft_splitRight
    (idx : (d : Fin (p + q)) → Fin (combinedDims I₁ I₂ d)) :
    combineIndex (splitLeft idx) (splitRight idx) = idx := by
  funext d
  refine Fin.addCases (motive := fun d =>
    combineIndex (splitLeft idx) (splitRight idx) d = idx d) ?_ ?_ d
  · intro i
    simp only [combineIndex, Fin.addCases_left, splitLeft]
    apply Fin.ext
    simp
  · intro i
    simp only [combineIndex, Fin.addCases_right, splitRight]
    apply Fin.ext
    simp

/-- The pair `(splitLeft, splitRight) : multi-index ≃ (left × right)` packaged
    as an `Equiv`, with inverse `combineIndex`. -/
def splitIndexEquiv :
    ((d : Fin (p + q)) → Fin (combinedDims I₁ I₂ d)) ≃
      (((d : Fin p) → Fin (I₁ d)) × ((d : Fin q) → Fin (I₂ d))) where
  toFun idx := (splitLeft idx, splitRight idx)
  invFun fg := combineIndex fg.1 fg.2
  left_inv idx := combineIndex_splitLeft_splitRight idx
  right_inv fg := by
    refine Prod.ext ?_ ?_
    · exact splitLeft_combineIndex fg.1 fg.2
    · exact splitRight_combineIndex fg.1 fg.2

/-! ### The mode split -/

/-- The **mode split** of a tensor of shape `combinedDims I₁ I₂`: a matrix
    whose rows are indexed by the first `p` modes (flattened via
    `multiIndexEquiv I₁`) and columns by the last `q` modes (flattened via
    `multiIndexEquiv I₂`), in the Kolda–Bader column-major order. -/
def modeSplit (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    Matrix (Fin (∏ d, I₁ d)) (Fin (∏ d, I₂ d)) 𝕜 :=
  Matrix.of fun ri cj =>
    X (combineIndex ((multiIndexEquiv I₁).symm ri) ((multiIndexEquiv I₂).symm cj))

/-- Closed-form entry of the mode split. -/
theorem modeSplit_apply (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂))
    (ri : Fin (∏ d, I₁ d)) (cj : Fin (∏ d, I₂ d)) :
    modeSplit X ri cj =
      X (combineIndex ((multiIndexEquiv I₁).symm ri) ((multiIndexEquiv I₂).symm cj)) :=
  rfl

/-- The mode split is injective (it is a bijective reshape composed with
    `Matrix.of`). -/
theorem modeSplit_injective :
    Function.Injective (modeSplit (𝕜 := 𝕜) (I₁ := I₁) (I₂ := I₂)) := by
  intro X Y h
  ext idx
  have h' := congr_fun (congr_fun h (multiIndexEquiv I₁ (splitLeft idx)))
    (multiIndexEquiv I₂ (splitRight idx))
  simpa only [modeSplit_apply, Equiv.symm_apply_apply,
    combineIndex_splitLeft_splitRight] using h'

/-- The **split rank**: the rank of the mode split. The lecture's `rⱼ`. -/
def splitRank (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) : ℕ :=
  (modeSplit X).rank

theorem splitRank_def (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    splitRank X = (modeSplit X).rank := rfl

/-! ### Linearity -/

theorem modeSplit_add (X Y : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    modeSplit (X + Y) = modeSplit X + modeSplit Y := by
  funext ri cj
  simp [modeSplit_apply, Matrix.add_apply]

theorem modeSplit_smul (c : 𝕜) (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    modeSplit (c • X) = c • modeSplit X := by
  funext ri cj
  simp [modeSplit_apply, Matrix.smul_apply]

theorem modeSplit_sub (X Y : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    modeSplit (X - Y) = modeSplit X - modeSplit Y := by
  funext ri cj
  simp [modeSplit_apply, Matrix.sub_apply]

end Tensor
