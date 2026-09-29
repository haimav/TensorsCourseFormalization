/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Eigenpair

/-!
# The Identity Tensor and the Shifted Higher-Order Power Method (Lecture 8)

To build the *shifted* higher-order power method, Lecture 8 seeks a tensor `E`
that acts like the identity under tensor apply. It shows:

* there is no tensor acting as the identity on *all* of `𝕜^N` (only on the unit
  sphere, and even there only for even orders);
* for even order such an "identity on the sphere" tensor `E_{D,N}` exists;
* shifting `A ↦ A + μ E` preserves Z-eigenvectors and shifts the eigenvalue by
  `μ` — the key fact behind the shifted HOPM.

This file formalises the parts of that development that do not depend on the
explicit `E_{D,N}` construction: it works with an *abstract* tensor `E`
satisfying `IdentityOnSphere E`, which is exactly the property the lecture's
`E_{D,N}` has (`E_{D,N} 𝐱 = 𝐱` for `‖𝐱‖₂ = 1`).

## Main definitions

* `CubicTensor.IdentityOnSphere E` — `E ⬝ₜ x = x` for every unit vector `x`.

## Main results

* `CubicTensor.apply_smul_unit` — off the sphere `E` scales by `c^D`, not `c`:
  `E ⬝ₜ (c • x) = c^D • x` for unit `x`. So `E` is the identity on the ray
  `{c • x}` only where `c^D = c` — it is *not* the identity off the unit sphere.
* `CubicTensor.not_identityOnSphere_of_odd_order` — for odd order there is no
  identity-on-sphere tensor at all (Lecture 8: `E(-x) = x ≠ -x`).
* `CubicTensor.isZEigenpair_shift` — **the shift fact:** `x` is a Z-eigenpair of
  `A` with eigenvalue `λ` iff it is a Z-eigenpair of `A + μ E` with eigenvalue
  `λ + μ`.
* `CubicTensor.rayleigh_shift`, `CubicTensor.rayleigh_of_identityOnSphere` — the
  Rayleigh quotient of the shifted tensor.
-/

noncomputable section

open Finset BigOperators

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}
variable {A E : CubicTensor 𝕜 (D + 1) N} {lam μ : 𝕜} {x : Fin N → 𝕜}

/-- **`E` acts like the identity on the unit sphere:** `E ⬝ₜ x = x` whenever
    `‖x‖₂ = 1`. This is the defining property of the lecture's `E_{D,N}`
    (whose explicit construction is orthogonal to everything below). -/
def IdentityOnSphere (E : CubicTensor 𝕜 (D + 1) N) : Prop :=
  ∀ x : Fin N → 𝕜, (∑ i, ‖x i‖ ^ 2 = 1) → E ⬝ₜ x = x

/-! ### Not an identity off the sphere -/

/-- **Off the unit sphere `E` scales by `c^D`, not by `c`.** For a unit vector
    `x`, `E ⬝ₜ (c • x) = c^D • x`. Hence `E` fixes a scaled vector `c • x` only
    when `c^D = c`; in particular it is *not* the identity map on `𝕜^N`. This
    is the lecture's observation that an identity-on-sphere tensor cannot act
    like the identity on any vector of non-unit norm. -/
theorem apply_smul_unit (hE : IdentityOnSphere E) (hx : ∑ i, ‖x i‖ ^ 2 = 1)
    (c : 𝕜) : E ⬝ₜ (c • x) = c ^ D • x := by
  rw [apply_smul_v, hE x hx]

/-- **No identity-on-sphere tensor exists for odd order.** For an odd-order
    tensor (order `D+1` odd) and `N ≥ 1`, no tensor acts as the identity on the
    unit sphere: applying it to `-x` would give `x` (even-power homogeneity),
    yet the sphere-identity forces `-x`, so `x = -x`, impossible for a nonzero
    unit vector. (Lecture 8: "`E(-x) = x ≠ -x`".) -/
theorem not_identityOnSphere_of_odd_order (hord : Odd (D + 1)) (hN : 0 < N)
    (E : CubicTensor 𝕜 (D + 1) N) : ¬ IdentityOnSphere E := by
  intro hE
  have hDeven : Even D := Nat.not_odd_iff_even.mp (Nat.odd_add_one.mp hord)
  set i0 : Fin N := ⟨0, hN⟩ with hi0
  set e : Fin N → 𝕜 := fun i => if i = i0 then 1 else 0 with he
  -- `e` is the first standard basis vector; it lies on the unit sphere.
  have hunit : ∑ i, ‖e i‖ ^ 2 = 1 := by
    simp only [he, apply_ite (‖·‖), norm_one, norm_zero]
    simp [Finset.sum_ite_eq']
  have h1 : E ⬝ₜ e = e := hE e hunit
  have hunit' : ∑ i, ‖(-e) i‖ ^ 2 = 1 := by
    simpa only [Pi.neg_apply, norm_neg] using hunit
  have h2 : E ⬝ₜ (-e) = -e := hE (-e) hunit'
  rw [apply_neg_v, hDeven.neg_one_pow, one_smul, h1] at h2
  -- `h2 : e = -e`; evaluate at `i0` to get `1 = -1`, contradicting char 0.
  have he0 : e i0 = 1 := by rw [he]; simp
  have hval := congrFun h2 i0
  rw [he0, Pi.neg_apply, he0] at hval
  -- `hval : (1 : 𝕜) = -1`
  have h2z : (2 : 𝕜) = 0 := by linear_combination hval
  norm_num at h2z

/-! ### The shift fact (Lecture 8) -/

/-- **Shifting preserves Z-eigenvectors and shifts the eigenvalue.** For an
    identity-on-sphere `E`, `x` is a Z-eigenpair of `A` with eigenvalue `λ` iff
    it is a Z-eigenpair of the shifted tensor `A + μ E` with eigenvalue `λ + μ`.
    This is the fact underlying the shifted higher-order power method: running
    HOPM on `A + μ E` finds the Z-eigenvectors of `A`. -/
theorem isZEigenpair_shift (hE : IdentityOnSphere E) (μ : 𝕜) :
    IsZEigenpair A lam x ↔ IsZEigenpair (A + μ • E) (lam + μ) x := by
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨?_, h2⟩
    rw [apply_add_T, apply_smul_T, h1, hE x h2, ← add_smul]
  · rintro ⟨h1, h2⟩
    refine ⟨?_, h2⟩
    rw [apply_add_T, apply_smul_T, hE x h2, add_smul] at h1
    exact add_right_cancel h1

/-! ### The shifted Rayleigh quotient -/

/-- **The Rayleigh quotient is affine in the shift.** `R_{A + μ E}(x) = R_A(x) +
    μ · R_E(x)` — pure linearity of the Rayleigh quotient in the tensor. -/
theorem rayleigh_shift (μ : 𝕜) :
    rayleigh (A + μ • E) x = rayleigh A x + μ * rayleigh E x := by
  rw [rayleigh_add_T, rayleigh_smul_T]

/-- On the unit sphere the Rayleigh quotient of an identity-on-sphere tensor is
    `∑ xᵢ²` (equal to `1` on the real unit sphere), matching the lecture's
    `R_{A+μE}(x) = xᵀ A x + μ (xᵀ x)^{D/2}` specialised to `‖x‖₂ = 1`. -/
theorem rayleigh_of_identityOnSphere (hE : IdentityOnSphere E)
    (hx : ∑ i, ‖x i‖ ^ 2 = 1) : rayleigh E x = ∑ i, x i ^ 2 := by
  rw [rayleigh_def, hE x hx]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  ring

end CubicTensor
