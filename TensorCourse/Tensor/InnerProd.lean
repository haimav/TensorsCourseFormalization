/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.RCLike.Basic

/-!
# Tensor Inner Product

The inner product of two tensors of the same shape is the sum over all
multi-indices of the conjugate of the first tensor's entry times the second.

## Main definitions

* `Tensor.dotProd T₁ T₂ = ∑ idx, starRingEnd 𝕜 (T₁ idx) * T₂ idx`
* Scoped notation `⟪T₁, T₂⟫ₜ` (in the `Tensor` namespace)

## Main results

* `dotProd_add_right`, `dotProd_smul_right` — linearity in the second argument
* `dotProd_add_left` — additivity in the first argument (conjugate-linear)
* `dotProd_conj_symm` — conjugate symmetry: `⟪T₁, T₂⟫ₜ = star ⟪T₂, T₁⟫ₜ`
* `dotProd_self_eq_zero` — positive definiteness: `⟪T, T⟫ₜ = 0 → T = 0`
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- The inner product of two tensors: `∑ idx, conj (T₁ idx) * T₂ idx`.
    Conjugate-linear in the first argument and linear in the second.

    Note: the conjugation is on the **first** argument (the standard math
    convention for inner products). Over `ℝ` conjugation is the identity,
    so this coincides with the lecture's unconjugated `vec(X)ᵀ vec(Y)`. -/
def dotProd (T₁ T₂ : Tensor 𝕜 D dims) : 𝕜 :=
  ∑ idx : (d : Fin D) → Fin (dims d), starRingEnd 𝕜 (T₁ idx) * T₂ idx

@[inherit_doc]
scoped[Tensor] notation "⟪" T₁ ", " T₂ "⟫ₜ" => Tensor.dotProd T₁ T₂

theorem dotProd_def (T₁ T₂ : Tensor 𝕜 D dims) :
    ⟪T₁, T₂⟫ₜ = ∑ idx, starRingEnd 𝕜 (T₁ idx) * T₂ idx := rfl

/-- Additivity in the second argument. -/
theorem dotProd_add_right (T₁ T₂ T₃ : Tensor 𝕜 D dims) :
    ⟪T₁, T₂ + T₃⟫ₜ = ⟪T₁, T₂⟫ₜ + ⟪T₁, T₃⟫ₜ := by
  simp only [dotProd_def, add_apply, mul_add, Finset.sum_add_distrib]

/-- Scalar-linearity in the second argument. -/
theorem dotProd_smul_right (T₁ T₂ : Tensor 𝕜 D dims) (c : 𝕜) :
    ⟪T₁, c • T₂⟫ₜ = c * ⟪T₁, T₂⟫ₜ := by
  simp only [dotProd_def, smul_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro idx _
  ring

/-- Additivity in the first argument. (Combined with `dotProd_conj_smul_left`,
    this makes `dotProd` conjugate-linear in its first argument.) -/
theorem dotProd_add_left (T₁ T₂ T₃ : Tensor 𝕜 D dims) :
    ⟪T₁ + T₂, T₃⟫ₜ = ⟪T₁, T₃⟫ₜ + ⟪T₂, T₃⟫ₜ := by
  simp only [dotProd_def, add_apply, map_add, add_mul, Finset.sum_add_distrib]

/-- Conjugate scalar pulls out of the first argument. -/
theorem dotProd_conj_smul_left (c : 𝕜) (T₁ T₂ : Tensor 𝕜 D dims) :
    ⟪c • T₁, T₂⟫ₜ = starRingEnd 𝕜 c * ⟪T₁, T₂⟫ₜ := by
  simp only [dotProd_def, smul_apply, map_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro idx _
  ring

/-- Conjugate-symmetry: `⟪T₁, T₂⟫ₜ = star ⟪T₂, T₁⟫ₜ`. -/
theorem dotProd_conj_symm (T₁ T₂ : Tensor 𝕜 D dims) :
    ⟪T₁, T₂⟫ₜ = starRingEnd 𝕜 ⟪T₂, T₁⟫ₜ := by
  simp only [dotProd_def, map_sum, map_mul, starRingEnd_self_apply]
  refine Finset.sum_congr rfl ?_
  intro idx _
  ring

/-- Each diagonal summand of `⟪T, T⟫ₜ` equals `‖T idx‖^2` coerced to `𝕜`. -/
private lemma dotProd_self_summand (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims d)) :
    starRingEnd 𝕜 (T idx) * T idx = ((‖T idx‖ ^ 2 : ℝ) : 𝕜) := by
  rw [RCLike.conj_mul]
  push_cast
  rfl

/-- `⟪T, T⟫ₜ` is the `𝕜`-coercion of the real sum of squared norms. -/
theorem dotProd_self_eq_ofReal (T : Tensor 𝕜 D dims) :
    ⟪T, T⟫ₜ = (((∑ idx, ‖T idx‖ ^ 2) : ℝ) : 𝕜) := by
  simp only [dotProd_def, dotProd_self_summand, RCLike.ofReal_sum]

/-- `⟪T, T⟫ₜ` has real part equal to the sum of squared norms. -/
theorem re_dotProd_self (T : Tensor 𝕜 D dims) :
    RCLike.re ⟪T, T⟫ₜ = ∑ idx, ‖T idx‖ ^ 2 := by
  rw [dotProd_self_eq_ofReal, RCLike.ofReal_re]

/-- Positive definiteness: if `⟪T, T⟫ₜ = 0` then `T = 0`. -/
theorem dotProd_self_eq_zero {T : Tensor 𝕜 D dims} (h : ⟪T, T⟫ₜ = 0) :
    T = 0 := by
  have hsum : (∑ idx, ‖T idx‖ ^ 2) = 0 := by
    have : (((∑ idx, ‖T idx‖ ^ 2) : ℝ) : 𝕜) = 0 := by
      rw [← dotProd_self_eq_ofReal]; exact h
    exact_mod_cast this
  have hnn : ∀ idx ∈ (Finset.univ : Finset ((d : Fin D) → Fin (dims d))),
      0 ≤ ‖T idx‖ ^ 2 := fun idx _ => sq_nonneg _
  have hall := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hsum
  ext idx
  have hi := hall idx (Finset.mem_univ _)
  have : ‖T idx‖ = 0 := by
    have := sq_eq_zero_iff.mp hi
    exact this
  simpa using (norm_eq_zero.mp this)

end Tensor
