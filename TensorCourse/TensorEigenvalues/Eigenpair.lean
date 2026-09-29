/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Rayleigh

/-!
# Tensor Eigenpairs: the Z/H/ℓp Zoo (Lecture 8)

This file formalises the tensor eigenpair definitions of Lecture 8 and their
basic invariances. For a cubic order-`(D+1)` tensor `A` and `x : Fin N → 𝕜`:

* a **Z-eigenpair** `(λ, x)` satisfies `A ⬝ₜ x = λ • x` with `‖x‖₂ = 1`;
* an **H-eigenpair** `(λ, x)` satisfies `A ⬝ₜ x = λ • x^{[D]}`, where
  `x^{[D]}ᵢ = xᵢ^D` is the elementwise power (order `D+1` ⇒ power `D`);
* an **ℓp-eigenpair** `(λ, x)` satisfies `A ⬝ₜ x = λ • x^{[p-1]}` with
  `‖x‖_p = 1`, unifying the two (`p = 2` gives Z).

## Main definitions

* `CubicTensor.IsZEigenpair`, `CubicTensor.IsHEigenpair`,
  `CubicTensor.IsLpEigenpair`.

## Main results

* `CubicTensor.IsZEigenpair.neg_even_order` / `.neg_odd_order` — the sign
  symmetry of Z-eigenpairs: for an even-order tensor `(λ, -x)` is again a
  Z-eigenpair, and for an odd-order tensor `(-λ, -x)` is.
* `CubicTensor.IsHEigenpair.smul` — H-eigenpairs are scale-invariant: if
  `(λ, x)` is one then so is `(λ, c • x)` for every `c` (Lecture 8's remark
  that H-eigenvectors need no normalisation).
* `CubicTensor.isZEigenpair_iff_isLpEigenpair_two` — Z-eigenpairs are exactly
  the ℓ₂-eigenpairs; `CubicTensor.IsHEigenpair.isLpEigenpair_order` — a
  normalised H-eigenvector is an ℓ_{D+1}-eigenvector.
* `CubicTensor.rayleigh_eq_of_isZEigenpair` — the Rayleigh quotient at a
  Z-eigenvector is `λ · ∑ xᵢ²` (equal to `λ` on the real unit sphere).
-/

noncomputable section

open Finset BigOperators

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}
variable {A : CubicTensor 𝕜 (D + 1) N} {lam : 𝕜} {x : Fin N → 𝕜}

/-! ### Definitions -/

/-- `(λ, x)` is a **Z-eigenpair** of `A`: `A ⬝ₜ x = λ • x` and `‖x‖₂ = 1`.
    This is the lecture's default "tensor eigenpair". -/
def IsZEigenpair (A : CubicTensor 𝕜 (D + 1) N) (lam : 𝕜) (x : Fin N → 𝕜) : Prop :=
  A ⬝ₜ x = lam • x ∧ ∑ i, ‖x i‖ ^ 2 = 1

/-- `(λ, x)` is an **H-eigenpair** of `A`: `A ⬝ₜ x = λ • x^{[D]}` where
    `x^{[D]}ᵢ = xᵢ^D` (for an order-`(D+1)` tensor). No normalisation is
    imposed — H-eigenpairs are scale-invariant (`IsHEigenpair.smul`). -/
def IsHEigenpair (A : CubicTensor 𝕜 (D + 1) N) (lam : 𝕜) (x : Fin N → 𝕜) : Prop :=
  A ⬝ₜ x = lam • (fun i => x i ^ D)

/-- `(λ, x)` is an **ℓp-eigenpair** of `A` (Lim's family): `A ⬝ₜ x = λ • x^{[p-1]}`
    with `‖x‖_p = 1`. Recovers Z-eigenpairs at `p = 2`. -/
def IsLpEigenpair (p : ℕ) (A : CubicTensor 𝕜 (D + 1) N) (lam : 𝕜) (x : Fin N → 𝕜) :
    Prop :=
  A ⬝ₜ x = lam • (fun i => x i ^ (p - 1)) ∧ ∑ i, ‖x i‖ ^ p = 1

/-! ### Sign symmetry of Z-eigenpairs (Lecture 8) -/

/-- **Even-order sign symmetry.** For an even-order tensor (order `D+1` even),
    if `(λ, x)` is a Z-eigenpair then so is `(λ, -x)`: applying an even-order
    tensor to `-x` leaves `A ⬝ₜ x` unchanged. -/
theorem IsZEigenpair.neg_even_order (h : IsZEigenpair A lam x) (hord : Even (D + 1)) :
    IsZEigenpair A lam (-x) := by
  have hDodd : Odd D := Nat.not_even_iff_odd.mp (Nat.even_add_one.mp hord)
  refine ⟨?_, ?_⟩
  · rw [apply_neg_v, hDodd.neg_one_pow, h.1, neg_one_smul, smul_neg]
  · simpa only [Pi.neg_apply, norm_neg] using h.2

/-- **Odd-order sign symmetry.** For an odd-order tensor (order `D+1` odd), if
    `(λ, x)` is a Z-eigenpair then `(-λ, -x)` is: applying an odd-order tensor
    to `-x` negates `A ⬝ₜ x`, which is absorbed by flipping the eigenvalue. -/
theorem IsZEigenpair.neg_odd_order (h : IsZEigenpair A lam x) (hord : Odd (D + 1)) :
    IsZEigenpair A (-lam) (-x) := by
  have hDeven : Even D := Nat.not_odd_iff_even.mp (Nat.odd_add_one.mp hord)
  refine ⟨?_, ?_⟩
  · rw [apply_neg_v, hDeven.neg_one_pow, h.1, one_smul, neg_smul, smul_neg, neg_neg]
  · simpa only [Pi.neg_apply, norm_neg] using h.2

/-! ### Scale invariance of H-eigenpairs (Lecture 8) -/

/-- **H-eigenpairs are scale-invariant.** If `(λ, x)` is an H-eigenpair then so
    is `(λ, c • x)` for every scalar `c`. This is why the H-eigenpair
    definition imposes no normalisation. -/
theorem IsHEigenpair.smul (h : IsHEigenpair A lam x) (c : 𝕜) :
    IsHEigenpair A lam (c • x) := by
  unfold IsHEigenpair at h ⊢
  rw [apply_smul_v, h]
  funext i
  simp only [Pi.smul_apply, smul_eq_mul, mul_pow]
  ring

/-! ### Relations within the zoo (Lecture 8) -/

/-- **Z-eigenpairs are exactly the ℓ₂-eigenpairs.** The elementwise power
    `x^{[2-1]} = x^{[1]} = x`, and the ℓ₂-norm constraint is the Euclidean one. -/
theorem isZEigenpair_iff_isLpEigenpair_two :
    IsZEigenpair A lam x ↔ IsLpEigenpair 2 A lam x := by
  have hx : (fun i => x i ^ (2 - 1)) = x := by funext i; simp
  unfold IsZEigenpair IsLpEigenpair
  rw [hx]

/-- **A normalised H-eigenvector is an ℓ_{D+1}-eigenvector.** For an order-`(D+1)`
    tensor the H-power `D` equals `(D+1)-1`, so an H-eigenpair with `‖x‖_{D+1}=1`
    is an ℓ_{D+1}-eigenpair. -/
theorem IsHEigenpair.isLpEigenpair_order (h : IsHEigenpair A lam x)
    (hnorm : ∑ i, ‖x i‖ ^ (D + 1) = 1) :
    IsLpEigenpair (D + 1) A lam x := by
  refine ⟨?_, hnorm⟩
  unfold IsHEigenpair at h
  rw [Nat.add_sub_cancel]
  exact h

/-! ### Rayleigh quotient at a Z-eigenvector -/

/-- **The Rayleigh quotient at a Z-eigenvector is `λ · ∑ xᵢ²`.** On the real
    unit sphere `∑ xᵢ² = 1`, so `R_A(x) = λ`: the eigenvalue is read off the
    Rayleigh quotient. -/
theorem rayleigh_eq_of_isZEigenpair (h : IsZEigenpair A lam x) :
    rayleigh A x = lam * ∑ i, x i ^ 2 := by
  rw [rayleigh_def, h.1, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

end CubicTensor
