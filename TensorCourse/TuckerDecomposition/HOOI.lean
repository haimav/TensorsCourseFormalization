/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TuckerDecomposition.Defs
import TensorCourse.Tensor.FrobeniusNorm
import TensorCourse.Tensor.InnerProd

/-!
# Tucker Decomposition: HOOI dual-problem equivalence

The higher-order orthogonal iteration (HOOI) of De Lathauwer, De Moor,
Vandewalle (2000) maximises, over orthonormal-column factor matrices `U`,
the energy captured by the projection of `X` onto the factor subspaces.
Its centrepiece is the **primal/dual equivalence**: with the optimal core
`S = X ×-chain Uᴴ` plugged into the reconstruction, the primal (residual)
objective and the dual (captured-energy) objective add up to `‖X‖²`, so
minimising the primal is the same as maximising the dual.

Concretely, writing `X̂ = tuckerTensor (tuckerTensor X Uᴴ) U` for the
reconstruction with the optimal core,

    hooiPrimal X U = ‖X − X̂‖²,    hooiDual X U = ‖tuckerTensor X Uᴴ‖²,

and for orthonormal-column factors (`(U d)ᴴ (U d) = 1`),

    hooiPrimal X U = ‖X‖² − hooiDual X U.

## Main definitions

* `Tensor.hooiPrimal X U` — the primal objective `‖X − X̂‖²`
  (eq:Tucker-primal).
* `Tensor.hooiDual X U` — the dual objective `‖tuckerTensor X Uᴴ‖²`
  (eq:Tucker-dual), the squared norm of the optimal core.

## Main results

* `Tensor.dotProd_tuckerTensor_right` — the **Tucker adjoint identity**
  `⟪X, tuckerTensor S U⟫ₜ = ⟪tuckerTensor X (fun d => (U d)ᴴ), S⟫ₜ`. Moving
  a Tucker reconstruction across the inner product conjugate-transposes
  every factor (cast-free: the closed-form definition makes both sides
  sums over `idx × jdx`). The general-`D`, general-shape analogue of the
  n-mode adjoint identity `dotProd_nModeProd_right`.
* `Tensor.frobNormSq_tuckerTensor_of_orthonormal` — norm preservation
  under semi-orthogonal factors: `‖tuckerTensor S U‖² = ‖S‖²` when
  `(U d)ᴴ (U d) = 1` for all `d`.
* `Tensor.hooi_primal_eq_normSq_sub_dual` (**the dual equivalence**, no
  axioms) — `hooiPrimal X U = ‖X‖² − hooiDual X U`.

## Transpose vs. conjugate transpose

These are adjoint-flavoured results (the inner product `⟪·,·⟫ₜ` conjugates
its first argument), so factor matrices carry `ᴴ`; over `ℝ`, `ᴴ = ᵀ`,
recovering the lecture.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {I J : Fin D → ℕ}

/-! ### The Tucker adjoint identity -/

/-- **Tucker adjoint identity.** Moving a Tucker reconstruction across the
    inner product replaces every factor by its conjugate transpose:

      ⟪X, tuckerTensor S U⟫ₜ = ⟪tuckerTensor X (fun d => (U d)ᴴ), S⟫ₜ.

    This is the general-`D`, general-shape analogue of the n-mode adjoint
    identity `dotProd_nModeProd_right`, and unlike it is **cast-free**: the
    closed-form definition of `tuckerTensor` makes both sides double sums
    over `idx` (output multi-indices of `X`) and `jdx` (core multi-indices
    of `S`), which agree after `Finset.sum_comm`. -/
theorem dotProd_tuckerTensor_right (X : Tensor 𝕜 D I) (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    ⟪X, tuckerTensor S U⟫ₜ = ⟪tuckerTensor X (fun d => (U d)ᴴ), S⟫ₜ := by
  rw [dotProd_def, dotProd_def]
  -- LHS: ∑ idx, conj (X idx) * ∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d).
  -- Push the conjugated `X`-factor into the inner sum.
  conv_lhs =>
    rw [show (fun idx => starRingEnd 𝕜 (X idx) * tuckerTensor S U idx) =
          (fun idx => ∑ jdx : (d : Fin D) → Fin (J d),
            starRingEnd 𝕜 (X idx) * (S jdx * ∏ d : Fin D, U d (idx d) (jdx d)))
        from funext fun idx => by rw [tuckerTensor_apply, Finset.mul_sum]]
  -- RHS: ∑ jdx, conj (tuckerTensor X Uᴴ jdx) * S jdx.
  -- Expand the conjugated core entry; `conj (∏ d, conj (U d ..)) = ∏ d, U d ..`.
  conv_rhs =>
    rw [show (fun jdx => starRingEnd 𝕜 (tuckerTensor X (fun d => (U d)ᴴ) jdx) * S jdx) =
          (fun jdx => ∑ idx : (d : Fin D) → Fin (I d),
            starRingEnd 𝕜 (X idx) * (S jdx * ∏ d : Fin D, U d (idx d) (jdx d)))
        from funext fun jdx => by
          rw [tuckerTensor_apply, map_sum, Finset.sum_mul]
          refine Finset.sum_congr rfl fun idx _ => ?_
          rw [map_mul]
          -- `conj (∏ d, (U d)ᴴ (jdx d) (idx d)) = ∏ d, U d (idx d) (jdx d)`.
          rw [show starRingEnd 𝕜 (∏ d : Fin D, (U d)ᴴ (jdx d) (idx d)) =
                ∏ d : Fin D, U d (idx d) (jdx d) from by
            rw [map_prod]
            refine Finset.prod_congr rfl fun d _ => ?_
            rw [Matrix.conjTranspose_apply, starRingEnd_apply, star_star]]
          ring]
  -- Both sides are double sums over `idx × jdx`; swap the order on one side.
  rw [Finset.sum_comm]

/-! ### Norm preservation under semi-orthogonal factors -/

/-- **Norm preservation.** If every factor `U d` has orthonormal columns
    (`(U d)ᴴ (U d) = 1`), the Tucker reconstruction preserves the squared
    Frobenius norm of the core: `‖tuckerTensor S U‖² = ‖S‖²`.

    Proof: `‖tuckerTensor S U‖² = re⟪tuckerTensor S U, tuckerTensor S U⟫`;
    slide the outer reconstruction across the inner product
    (`dotProd_tuckerTensor_right`) to get
    `re⟪tuckerTensor (tuckerTensor S U) Uᴴ, S⟫`; collapse the composition
    `tuckerTensor (tuckerTensor S U) Uᴴ = tuckerTensor S (fun d => Uᴴ U)
     = tuckerTensor S 1 = S` (`tuckerTensor_tuckerTensor`, `(U d)ᴴ (U d) = 1`,
    `tuckerTensor_one`); the result is `re⟪S, S⟫ = ‖S‖²`. -/
theorem frobNormSq_tuckerTensor_of_orthonormal (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜)
    (hU : ∀ d, (U d)ᴴ * (U d) = 1) :
    ‖tuckerTensor S U‖_F² = ‖S‖_F² := by
  rw [frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd]
  -- Slide the outer reconstruction across the inner product.
  rw [dotProd_tuckerTensor_right (tuckerTensor S U) S U]
  -- Collapse `tuckerTensor (tuckerTensor S U) Uᴴ = S`.
  rw [tuckerTensor_tuckerTensor]
  rw [show (fun d => (U d)ᴴ * U d) =
        (fun _ => (1 : Matrix (Fin (J _)) (Fin (J _)) 𝕜)) from funext hU]
  rw [tuckerTensor_one]

/-! ### The HOOI primal and dual objectives -/

variable {dims : Fin D → ℕ} {R : Fin D → ℕ}

/-- **HOOI primal objective** (eq:Tucker-primal). With the optimal core
    `S = X ×-chain Uᴴ = tuckerTensor X (fun d => (U d)ᴴ)` plugged back into
    the reconstruction, this is the squared Frobenius residual

      hooiPrimal X U = ‖X − tuckerTensor (tuckerTensor X Uᴴ) U‖². -/
def hooiPrimal (X : Tensor 𝕜 D dims)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (R d)) 𝕜) : ℝ :=
  (X - tuckerTensor (tuckerTensor X (fun d => (U d)ᴴ)) U).frobNormSq

/-- **HOOI dual objective** (eq:Tucker-dual): the squared Frobenius norm of
    the optimal core,

      hooiDual X U = ‖tuckerTensor X (fun d => (U d)ᴴ)‖². -/
def hooiDual (X : Tensor 𝕜 D dims)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (R d)) 𝕜) : ℝ :=
  (tuckerTensor X (fun d => (U d)ᴴ)).frobNormSq

/-! ### A difference-expansion helper -/

/-- Expansion of the squared Frobenius norm of a difference:
    `‖A − B‖² = ‖A‖² + ‖B‖² − 2 · re⟪A, B⟫`. -/
theorem frobNormSq_sub (A B : Tensor 𝕜 D dims) :
    ‖A - B‖_F² = ‖A‖_F² + ‖B‖_F² - 2 * RCLike.re ⟪A, B⟫ₜ := by
  rw [frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd, frobNormSq_eq_re_dotProd]
  -- Write `A − B = A + (-1) • B` and expand the inner product bilinearly.
  rw [show A - B = A + (-1 : 𝕜) • B from by rw [neg_one_smul]; abel]
  rw [dotProd_add_left, dotProd_add_right, dotProd_add_right]
  rw [dotProd_smul_right, dotProd_conj_smul_left, dotProd_conj_smul_left,
    dotProd_smul_right]
  -- Turn each `(-1 : 𝕜) * z` (and `conj (-1) = -1`) into a negation, so the
  -- four cross terms become `⟪A,A⟫ + (-⟪A,B⟫) + (-⟪B,A⟫) + ⟪B,B⟫`.
  simp only [map_neg, map_one, neg_one_mul, mul_neg, neg_neg]
  -- `re` is additive and respects negation; `re ⟪B,A⟫ = re ⟪A,B⟫`.
  rw [map_add, map_add, map_add, map_neg, map_neg]
  rw [show ⟪B, A⟫ₜ = starRingEnd 𝕜 ⟪A, B⟫ₜ from dotProd_conj_symm B A, RCLike.conj_re]
  ring

/-! ### The dual-problem equivalence -/

/-- **HOOI dual-problem equivalence** (plan item 15, no axioms). With
    orthonormal-column factor matrices (`(U d)ᴴ (U d) = 1`), the primal and
    dual objectives satisfy

      hooiPrimal X U = ‖X‖² − hooiDual X U,

    so minimising the residual `hooiPrimal` is equivalent to maximising the
    captured energy `hooiDual`.

    Proof: set `X̂ = tuckerTensor (tuckerTensor X Uᴴ) U`. Expanding the
    residual (`frobNormSq_sub`),
    `hooiPrimal = ‖X‖² + ‖X̂‖² − 2 · re⟪X, X̂⟫`. Both `‖X̂‖²` and `re⟪X, X̂⟫`
    equal the dual `g = ‖tuckerTensor X Uᴴ‖²`:
    * `‖X̂‖² = g` by norm preservation
      (`frobNormSq_tuckerTensor_of_orthonormal`, peeling the outer `U`);
    * `re⟪X, X̂⟫ = g` by the Tucker adjoint identity
      (`dotProd_tuckerTensor_right`): sliding the outer `U` across turns
      `⟪X, tuckerTensor (tuckerTensor X Uᴴ) U⟫` into
      `⟪tuckerTensor X Uᴴ, tuckerTensor X Uᴴ⟫`, whose real part is `g`.
    Hence `hooiPrimal = ‖X‖² + g − 2g = ‖X‖² − g`. -/
theorem hooi_primal_eq_normSq_sub_dual (X : Tensor 𝕜 D dims)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (R d)) 𝕜)
    (hU : ∀ d, (U d)ᴴ * (U d) = 1) :
    hooiPrimal X U = X.frobNormSq - hooiDual X U := by
  -- Abbreviate the optimal core `S` and the dual objective `g`.
  set S : Tensor 𝕜 D R := tuckerTensor X (fun d => (U d)ᴴ) with hS
  rw [hooiPrimal, hooiDual, ← hS]
  -- Expand the residual `‖X − tuckerTensor S U‖²`.
  rw [frobNormSq_sub X (tuckerTensor S U)]
  -- `‖tuckerTensor S U‖² = ‖S‖²` (norm preservation).
  rw [frobNormSq_tuckerTensor_of_orthonormal S U hU]
  -- `re⟪X, tuckerTensor S U⟫ = re⟪tuckerTensor X Uᴴ, S⟫ = re⟪S, S⟫ = ‖S‖²`.
  rw [dotProd_tuckerTensor_right X S U, ← hS, ← frobNormSq_eq_re_dotProd]
  -- `‖X‖² + ‖S‖² − 2‖S‖² = ‖X‖² − ‖S‖²`.
  rw [frobNormSq_def, frobNormSq_def]
  ring

end Tensor

end
