/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.InnerProd
import TensorCourse.Tensor.Vectorize
import TensorCourse.Tensor.ModeUnfold

/-!
# Tensor Frobenius Norm

The (squared) Frobenius norm of a tensor is the sum of the squared norms
of its entries, `∑ idx, ‖T idx‖²`. It equals `RCLike.re ⟪T, T⟫ₜ` (see
`re_dotProd_self`), i.e. the Euclidean norm of the vectorization, and is
invariant under mode-`n` matricization (which only permutes the entries).

## Main definitions

* `Tensor.frobNormSq T = ∑ idx, ‖T idx‖²` — the squared Frobenius norm.
* `Tensor.frobNorm T = Real.sqrt (frobNormSq T)` — the Frobenius norm.
* Scoped notations `‖T‖_F` (= `frobNorm`) and `‖T‖_F²` (= `frobNormSq`).

## Main results

* `Tensor.frobNormSq_eq_re_dotProd` — `frobNormSq T = RCLike.re ⟪T, T⟫ₜ`.
* `Tensor.frobNormSq_nonneg`, `Tensor.frobNorm_nonneg`.
* `Tensor.frobNorm_sq` — `(frobNorm T)² = frobNormSq T`.
* `Tensor.frobNormSq_eq_zero_iff` — `frobNormSq T = 0 ↔ T = 0`.
* `Tensor.frobNormSq_vectorize` — the squared Frobenius norm is the
  Euclidean norm of the vectorization.
* `Tensor.frobNormSq_modeUnfold` — matricization preserves the squared
  Frobenius norm.
* `Tensor.frobNormSq_add_of_orthogonal` — Pythagorean identity for
  orthogonal tensors.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- Squared Frobenius norm of a tensor: `∑ idx, ‖T idx‖²` (a real number).
    Equals `RCLike.re ⟪T, T⟫ₜ` (see `re_dotProd_self`). -/
def frobNormSq (T : Tensor 𝕜 D dims) : ℝ := ∑ idx, ‖T idx‖ ^ 2

/-- Frobenius norm of a tensor. -/
def frobNorm (T : Tensor 𝕜 D dims) : ℝ := Real.sqrt T.frobNormSq

@[inherit_doc]
scoped[Tensor] notation "‖" T "‖_F" => Tensor.frobNorm T

@[inherit_doc]
scoped[Tensor] notation "‖" T "‖_F²" => Tensor.frobNormSq T

@[simp] theorem frobNormSq_def (T : Tensor 𝕜 D dims) :
    ‖T‖_F² = ∑ idx, ‖T idx‖ ^ 2 := rfl

theorem frobNorm_def (T : Tensor 𝕜 D dims) :
    ‖T‖_F = Real.sqrt T.frobNormSq := rfl

/-- The squared Frobenius norm equals the real part of `⟪T, T⟫ₜ`. -/
theorem frobNormSq_eq_re_dotProd (T : Tensor 𝕜 D dims) :
    ‖T‖_F² = RCLike.re ⟪T, T⟫ₜ := (re_dotProd_self T).symm

/-- The squared Frobenius norm is nonnegative. -/
theorem frobNormSq_nonneg (T : Tensor 𝕜 D dims) : 0 ≤ ‖T‖_F² :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- The Frobenius norm is nonnegative. -/
theorem frobNorm_nonneg (T : Tensor 𝕜 D dims) : 0 ≤ ‖T‖_F :=
  Real.sqrt_nonneg _

/-- The square of the Frobenius norm is the squared Frobenius norm. -/
theorem frobNorm_sq (T : Tensor 𝕜 D dims) : ‖T‖_F ^ 2 = ‖T‖_F² :=
  Real.sq_sqrt (frobNormSq_nonneg T)

/-- The squared Frobenius norm vanishes iff the tensor is zero. -/
theorem frobNormSq_eq_zero_iff {T : Tensor 𝕜 D dims} :
    ‖T‖_F² = 0 ↔ T = 0 := by
  constructor
  · intro h
    -- `frobNormSq T = re ⟪T,T⟫ₜ = 0` forces `⟪T,T⟫ₜ = 0`, hence `T = 0`.
    apply dotProd_self_eq_zero
    rw [dotProd_self_eq_ofReal]
    rw [frobNormSq_def] at h
    rw [h]
    simp
  · intro h
    subst h
    rw [frobNormSq_def]
    simp

/-- The squared Frobenius norm equals the squared Euclidean norm of the
    vectorization. -/
theorem frobNormSq_vectorize (T : Tensor 𝕜 D dims) :
    ‖T‖_F² = ∑ i, ‖vectorize T i‖ ^ 2 := by
  -- Both equal `re ⟪T, T⟫ₜ`; expand the inner product via vectorization.
  rw [frobNormSq_eq_re_dotProd, dotProd_eq_vectorize_dotProd]
  rw [map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  -- `re (conj v * v) = ‖v‖²`.
  rw [RCLike.conj_mul, ← RCLike.ofReal_pow, RCLike.ofReal_re]

/-- Matricization preserves the squared Frobenius norm: it equals the sum
    over all matrix entries of the mode-`n` unfolding. -/
theorem frobNormSq_modeUnfold (n : Fin D) (T : Tensor 𝕜 D dims) :
    ‖T‖_F² = ∑ i, ∑ j, ‖T.modeUnfold n i j‖ ^ 2 := by
  rw [frobNormSq_def]
  -- Reindex the entry sum through the defining unfolding bijection.
  let e : ((d : Fin D) → Fin (dims d)) ≃
      Fin (dims n) × Fin (Tensor.prodExcept n dims) :=
    (separateModeEquiv n dims).trans
      (Equiv.prodCongr (Equiv.refl (Fin (dims n))) (remainingModesEquiv n dims))
  rw [← Equiv.sum_comp e.symm (fun idx => ‖T idx‖ ^ 2)]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [modeUnfold_apply]
  -- `e.symm (i, j) = (separateModeEquiv n dims).symm (i, (remainingModesEquiv n dims).symm j)`.
  rfl

/-- **Pythagorean identity.** If `A` and `B` are orthogonal
    (`⟪A, B⟫ₜ = 0`), the squared Frobenius norm is additive. -/
theorem frobNormSq_add_of_orthogonal {A B : Tensor 𝕜 D dims}
    (h : ⟪A, B⟫ₜ = 0) :
    ‖A + B‖_F² = ‖A‖_F² + ‖B‖_F² := by
  rw [frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd]
  -- Expand `⟪A+B, A+B⟫ₜ` into four terms.
  rw [dotProd_add_left, dotProd_add_right, dotProd_add_right]
  -- The cross term `⟪B, A⟫ₜ = conj ⟪A, B⟫ₜ = 0`.
  have hBA : ⟪B, A⟫ₜ = 0 := by
    rw [dotProd_conj_symm B A, h, map_zero]
  rw [h, hBA, add_zero, zero_add]
  rw [map_add]
end Tensor
