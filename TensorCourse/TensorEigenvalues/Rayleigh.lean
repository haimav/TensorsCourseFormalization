/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Apply

/-!
# The Tensor Rayleigh Quotient (Lecture 8)

For a cubic order-`(D+1)` tensor `A` and a vector `x`, the *tensor Rayleigh
quotient* is the (unnormalised) form

  `R_A(x) = xᵀ (A ⬝ₜ x) = ∑ᵢ xᵢ (A ⬝ₜ x)ᵢ`.

This is the tensor analogue of the matrix Rayleigh quotient `xᵀ A x`. We take
the *bilinear* (transpose, unconjugated) form, matching the lecture's
`𝐱ᵀ 𝒜 𝐱`; over `ℝ` it is the usual quantity, and over `ℂ` it is the
holomorphic bilinear form (not the sesquilinear inner product).

## Main definitions

* `CubicTensor.rayleigh A x` — the tensor Rayleigh quotient `xᵀ (A ⬝ₜ x)`.

## Main results

* `CubicTensor.rayleigh_eq_sum` — the fully-expanded multilinear form
  `R_A(x) = ∑_{idx} A(idx) · ∏_d x(idx d)`, symmetric in all `D+1` slots.
* `CubicTensor.rayleigh_smul_v` — homogeneity of degree `D+1`:
  `R_A(c • x) = c^{D+1} · R_A(x)`.
* `CubicTensor.rayleigh_add_T`, `CubicTensor.rayleigh_smul_T` — linearity in
  the tensor argument (used for the shifted Rayleigh quotient).
-/

noncomputable section

open Finset BigOperators

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}

/-- **The tensor Rayleigh quotient** `R_A(x) = xᵀ (A ⬝ₜ x) = ∑ᵢ xᵢ (A ⬝ₜ x)ᵢ`. -/
def rayleigh (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) : 𝕜 :=
  ∑ i : Fin N, x i * (A ⬝ₜ x) i

theorem rayleigh_def (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    rayleigh A x = ∑ i : Fin N, x i * (A ⬝ₜ x) i := rfl

/-- **The Rayleigh quotient as a symmetric multilinear form.** Expanding the
    apply, `R_A(x)` is the full contraction of `A` with `D+1` copies of `x`:
    `R_A(x) = ∑_{idx : Fin (D+1) → Fin N} A(idx) · ∏_d x(idx d)`. All `D+1`
    index slots enter symmetrically, which is why the eigenvalue theory is
    cleanest for symmetric `A`. -/
theorem rayleigh_eq_sum (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    rayleigh A x =
      ∑ idx : Fin (D + 1) → Fin N, A idx * ∏ d : Fin (D + 1), x (idx d) := by
  -- Combine the outer `∑ i` and inner `∑ rest` into a single sum over
  -- `Fin (D+1) → Fin N` via the `Fin.cons` equivalence.
  rw [rayleigh_def]
  simp only [apply_apply, Finset.mul_sum]
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N))
        (fun idx => A idx * ∏ d : Fin (D + 1), x (idx d)),
      Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  -- The summand at `(i, rest)` is `A (cons i rest) * ∏_{D+1} x (cons i rest d)`;
  -- reindex the pair through `consEquiv` and split off the first factor.
  have hc : (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) (i, rest)
      = Fin.cons i rest := by
    funext d; exact Fin.consEquiv_apply _ _ _
  simp only [hc, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]
  ring

/-- **Homogeneity of the Rayleigh quotient.** `R_A(c • x) = c^{D+1} · R_A(x)`
    (degree one more than tensor apply, since `xᵀ` contributes an extra
    factor of `x`). -/
theorem rayleigh_smul_v (A : CubicTensor 𝕜 (D + 1) N) (c : 𝕜) (x : Fin N → 𝕜) :
    rayleigh A (c • x) = c ^ (D + 1) * rayleigh A x := by
  rw [rayleigh_def, rayleigh_def, apply_smul_v, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp only [Pi.smul_apply, smul_eq_mul, pow_succ]
  ring

/-- The Rayleigh quotient is additive in the tensor argument. -/
theorem rayleigh_add_T (A B : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    rayleigh (A + B) x = rayleigh A x + rayleigh B x := by
  rw [rayleigh_def, rayleigh_def, rayleigh_def, apply_add_T, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp only [Pi.add_apply]
  ring

/-- Scalar multiplication of the tensor pulls out of the Rayleigh quotient. -/
theorem rayleigh_smul_T (c : 𝕜) (A : CubicTensor 𝕜 (D + 1) N) (x : Fin N → 𝕜) :
    rayleigh (c • A) x = c * rayleigh A x := by
  rw [rayleigh_def, rayleigh_def, apply_smul_T, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

end CubicTensor
