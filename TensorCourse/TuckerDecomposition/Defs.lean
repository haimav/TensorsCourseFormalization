/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import TensorCourse.Tensor.Cast
import TensorCourse.Tensor.Vectorize
import TensorCourse.Tensor.NModeProd
import TensorCourse.Tensor.Diagonal
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.Weighted
import TensorCourse.Matrix.SemiOrthogonal
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Fin.VecNotation
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# Tucker Decomposition

This file defines the Tucker decomposition — the "compression" generalisation
of the SVD — as a closed-form reconstruction of a tensor from a *core* tensor
and a list of *factor matrices*, together with its core algebraic theory.

## Main definitions

* `Tensor.tuckerTensor S U` — the Tucker reconstruction
  `⟦S; U⁽⁰⁾,…,U⁽ᴰ⁻¹⁾⟧` of shape `I` from a core `S : Tensor 𝕜 D J` and factor
  matrices `U d : Matrix (Fin (I d)) (Fin (J d)) 𝕜`:
  `tuckerTensor S U idx = ∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)`.
* `Tensor.TuckerDecomp` — a Tucker decomposition packaged as a single object
  (core + factor matrices), mirroring `CPDecomp`; `TuckerDecomp.tensor` is its
  reconstruction (definitionally `tuckerTensor`) and `TuckerDecomp.IsOrthonormal`
  the orthogonal-factors condition.
* Scoped order-3 display sugar `⟦S ⨾ U, V, W⟧` (separator `⨾`, U+2A3E).

## Main results

* `tuckerTensor_one` — identity factors leave the tensor unchanged.
* `tuckerTensor_single_mode` — factors all identity except at mode `n`
  collapse to a single n-mode product `X ×ₙ P`.
* `tuckerTensor_tuckerTensor` — **the composition lemma**: composing two
  Tucker reconstructions multiplies the factor matrices mode-by-mode.
* `tuckerTensor_transform` — non-uniqueness: invertible transforms move
  freely between the core and the factors.
* `cpRank_tuckerTensor_le` — Tucker induces CP: `cpRank ≤ ∏ d, J d`.
* `cpWeighted_eq_tuckerTensor_diagonal` — CP (σ-form) is Tucker with a
  diagonal core.
* `tuckerTensor_two_eq_matrix_mul` — order-2 sanity check: the Tucker
  reconstruction is `U * S * Vᵀ` (the SVD-compression picture).

## Notation conventions

`I : Fin D → ℕ` are the output (ambient) dimensions, `J : Fin D → ℕ` the core
dimensions; factor matrix `U d : Matrix (Fin (I d)) (Fin (J d)) 𝕜`. The
Tucker bracket is a multi-token literal, not a `ₜ`-operator (cf. the project
conventions). Purely algebraic identities use plain transpose `ᵀ`.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Definition -/

/-- **Tucker reconstruction.** For a core `S : Tensor 𝕜 D J` and factor
    matrices `U d : Matrix (Fin (I d)) (Fin (J d)) 𝕜`, the reconstructed
    tensor `⟦S; U⁽⁰⁾,…,U⁽ᴰ⁻¹⁾⟧` of shape `I`:

      tuckerTensor S U idx = ∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)

    where `jdx` ranges over core multi-indices `(d : Fin D) → Fin (J d)`. -/
def tuckerTensor {D : ℕ} {I J : Fin D → ℕ} (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) : Tensor 𝕜 D I :=
  fun idx => ∑ jdx : (d : Fin D) → Fin (J d), S jdx * ∏ d : Fin D, U d (idx d) (jdx d)

@[simp] theorem tuckerTensor_apply {D : ℕ} {I J : Fin D → ℕ} (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜)
    (idx : (d : Fin D) → Fin (I d)) :
    tuckerTensor S U idx =
      ∑ jdx : (d : Fin D) → Fin (J d), S jdx * ∏ d : Fin D, U d (idx d) (jdx d) :=
  rfl

/-! ### The packaged decomposition `TuckerDecomp`

The function `tuckerTensor` is the primary, general-`D` API; the algebraic
theory (composition, transform, matricized/vectorized forms, optimal core,
error bounds, HOOI) is all phrased on it. `TuckerDecomp` is a thin packaging
on top — a named home for the lecture's noun *"a Tucker decomposition"* and
the object the existence theorem `thm:hosvd` produces — mirroring how
`CPDecomp` packages a CP decomposition. Its reconstruction `TuckerDecomp.tensor`
is *definitionally* `tuckerTensor`, so every `tuckerTensor` lemma applies to it
after `tensor_eq`. -/

/-- A **Tucker decomposition** packaged as a single object: a core tensor
    `core : Tensor 𝕜 D J` together with one factor matrix per mode,
    `factors d : Matrix (Fin (I d)) (Fin (J d)) 𝕜`. This gives the lecture's
    `⟦S; U⁽⁰⁾,…,U⁽ᴰ⁻¹⁾⟧` a named home, mirroring `CPDecomp`. As in the function
    form, `I` are the output (ambient) dimensions and `J` the core dimensions. -/
structure TuckerDecomp (𝕜 : Type*) [RCLike 𝕜] (D : ℕ) (I J : Fin D → ℕ) where
  /-- The core tensor, of shape `J`. -/
  core : Tensor 𝕜 D J
  /-- `factors d : Matrix (Fin (I d)) (Fin (J d)) 𝕜` is the mode-`d` factor. -/
  factors : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜

/-- Reconstruct the tensor from a packaged Tucker decomposition. By definition
    this is the function-level `tuckerTensor tk.core tk.factors`. -/
def TuckerDecomp.tensor {D : ℕ} {I J : Fin D → ℕ} (tk : TuckerDecomp 𝕜 D I J) :
    Tensor 𝕜 D I :=
  tuckerTensor tk.core tk.factors

/-- `TuckerDecomp.tensor` is definitionally the function-level `tuckerTensor`,
    exposing the whole `tuckerTensor` API to packaged decompositions. -/
@[simp] theorem TuckerDecomp.tensor_eq {D : ℕ} {I J : Fin D → ℕ}
    (tk : TuckerDecomp 𝕜 D I J) :
    tk.tensor = tuckerTensor tk.core tk.factors := rfl

/-- A packaged Tucker decomposition has **orthonormal factors** when every
    factor matrix has orthonormal columns (`(U d)ᴴ (U d) = 1`). This is the
    lecture's "orthogonal factor matrices" condition (cf. the discussion after
    eq:Tucker-3order-transform), attainable for any decomposition by
    Gram–Schmidt / QR / SVD on each factor and pushing the coefficients into
    the core. -/
def TuckerDecomp.IsOrthonormal {D : ℕ} {I J : Fin D → ℕ}
    (tk : TuckerDecomp 𝕜 D I J) : Prop :=
  ∀ d, Matrix.HasOrthonormalCols (tk.factors d)

/-! ### Linearity in the core tensor -/

/-- `tuckerTensor` is additive in the core tensor. -/
theorem tuckerTensor_add_core {D : ℕ} {I J : Fin D → ℕ} (S T : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    tuckerTensor (S + T) U = tuckerTensor S U + tuckerTensor T U := by
  ext idx
  simp only [tuckerTensor_apply, Tensor.add_apply, add_mul]
  rw [Finset.sum_add_distrib]

/-- `tuckerTensor` is homogeneous in the core tensor. -/
theorem tuckerTensor_smul_core {D : ℕ} {I J : Fin D → ℕ} (c : 𝕜)
    (S : Tensor 𝕜 D J) (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    tuckerTensor (c • S) U = c • tuckerTensor S U := by
  ext idx
  simp only [tuckerTensor_apply, Tensor.smul_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun jdx _ => ?_
  ring

/-! ### Composition lemma (the engine) -/

/-- **Tucker composition.** Composing two Tucker reconstructions multiplies
    the factor matrices mode-by-mode:

      tuckerTensor (tuckerTensor X V) U = tuckerTensor X (fun d => U d * V d).

    The single most reusable identity of the lecture. -/
theorem tuckerTensor_tuckerTensor {D : ℕ} {I J K : Fin D → ℕ}
    (X : Tensor 𝕜 D K)
    (V : (d : Fin D) → Matrix (Fin (J d)) (Fin (K d)) 𝕜)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    tuckerTensor (tuckerTensor X V) U = tuckerTensor X (fun d => U d * V d) := by
  ext idx
  simp only [tuckerTensor_apply]
  -- LHS: ∑ jdx, (∑ kdx, X kdx * ∏ d, V d (jdx d) (kdx d)) * ∏ d, U d (idx d) (jdx d)
  -- Push the outer factor into the inner sum, then swap the two sums.
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm]
  -- Now: ∑ kdx, ∑ jdx, (X kdx * ∏ d, V d (jdx d) (kdx d)) * ∏ d, U d (idx d) (jdx d)
  refine Finset.sum_congr rfl (fun kdx _ => ?_)
  -- RHS inner term: X kdx * ∏ d, (U d * V d) (idx d) (kdx d).
  -- Expand the matrix product entry and turn the ∏-of-sums into a sum over jdx.
  simp only [Matrix.mul_apply]
  -- RHS: X kdx * ∏ d, ∑ j, U d (idx d) j * V d j (kdx d).
  rw [Fintype.prod_sum (fun d (j : Fin (J d)) => U d (idx d) j * V d j (kdx d))]
  -- RHS: X kdx * ∑ jdx, ∏ d, (U d (idx d) (jdx d) * V d (jdx d) (kdx d)).
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  -- Match the single jdx-term on both sides.
  rw [Finset.prod_mul_distrib]
  ring

/-! ### Identity factors -/

/-- A Tucker reconstruction with identity factor matrices is the identity:
    the Kronecker-delta entries of `(1 : Matrix …)` collapse the core sum. -/
theorem tuckerTensor_one {D : ℕ} {I : Fin D → ℕ} (X : Tensor 𝕜 D I) :
    tuckerTensor X (fun _ => (1 : Matrix (Fin (I _)) (Fin (I _)) 𝕜)) = X := by
  ext idx
  simp only [tuckerTensor_apply]
  -- ∑ jdx, X jdx * ∏ d, (1 : Matrix) (idx d) (jdx d)
  -- The product is 1 if jdx = idx (all coordinates agree), else 0.
  rw [Finset.sum_eq_single idx]
  · simp only [Matrix.one_apply_eq, Finset.prod_const_one, mul_one]
  · intro jdx _ hjdx
    -- jdx ≠ idx, so some coordinate differs, killing the product.
    have : ∃ d, idx d ≠ jdx d := by
      by_contra hc
      push Not at hc
      exact hjdx (funext (fun d => (hc d).symm))
    obtain ⟨d, hd⟩ := this
    rw [Finset.prod_eq_zero (Finset.mem_univ d), mul_zero]
    rw [Matrix.one_apply_ne hd]
  · intro h; exact absurd (Finset.mem_univ idx) h

/-! ### Single-mode reduction -/

/-- The single-mode factor list: the identity matrix at every mode except `n`,
    where it is `P`. The off-mode `(1 : Matrix …)`s force the core to have shape
    `Function.update I n Jn`; the mode-`n` `P` reshapes mode `n` from `Jn` to
    `I n`. Each factor is reindexed via `submatrix` so that mode `d`'s column
    type is `Fin ((Function.update I n Jn) d)`. -/
def singleModeFactor {D : ℕ} {I : Fin D → ℕ} {Jn : ℕ} (n : Fin D)
    (P : Matrix (Fin (I n)) (Fin Jn) 𝕜) :
    (d : Fin D) → Matrix (Fin (I d)) (Fin ((Function.update I n Jn) d)) 𝕜 :=
  fun d =>
    if h : d = n then
      P.submatrix
        (fun i : Fin (I d) => i.cast (by rw [h]))
        (fun j : Fin ((Function.update I n Jn) d) => j.cast (by rw [h, Function.update_self]))
    else
      (1 : Matrix (Fin (I d)) (Fin (I d)) 𝕜).submatrix id
        (fun j : Fin ((Function.update I n Jn) d) => j.cast (Function.update_of_ne h Jn I))

/-- **Single-mode reduction.** A Tucker product whose factors are all identities
    except at mode `n` collapses to a single n-mode product `X ×ₙ P`. The output
    cast bridges `update (update I n Jn) n (I n) = I` (a `Function.update_idem`
    then `Function.update_eq_self` composite). -/
theorem tuckerTensor_single_mode {D : ℕ} {I : Fin D → ℕ} {Jn : ℕ} (n : Fin D)
    (P : Matrix (Fin (I n)) (Fin Jn) 𝕜) (X : Tensor 𝕜 D (Function.update I n Jn)) :
    tuckerTensor X (singleModeFactor n P)
      = cast (congrArg (Tensor 𝕜 D)
          ((Function.update_idem Jn (I n) I).trans (Function.update_eq_self n I)))
          (X.nModeProd n (P.submatrix id
            (fun j : Fin ((Function.update I n Jn) n) =>
              j.cast (Function.update_self n Jn I)))) := by
  ext idx
  -- RHS: evaluate the cast, then expand via `nModeProd_apply`.
  have heq : Function.update (Function.update I n Jn) n (I n) = I :=
    (Function.update_idem Jn (I n) I).trans (Function.update_eq_self n I)
  rw [cast_tensor_apply (h := heq), nModeProd_apply]
  -- LHS: ∑ jdx, X jdx * ∏ d, (singleModeFactor n P) d (idx d) (jdx d).
  simp only [tuckerTensor_apply]
  -- The constraint-tuple map sending the free mode-`n` index `jn` to the
  -- multi-index that agrees with `idx` off mode `n` and equals `jn` at `n`.
  set φ : Fin ((Function.update I n Jn) n) → ((d : Fin D) → Fin ((Function.update I n Jn) d)) :=
    fun jn d => if h : d = n then h ▸ jn
                else (idx d).cast (Function.update_of_ne h Jn I).symm with hφ
  -- Abbreviation for the LHS summand.
  set f : ((d : Fin D) → Fin ((Function.update I n Jn) d)) → 𝕜 :=
    fun jdx => X jdx * ∏ d : Fin D, (singleModeFactor n P) d (idx d) (jdx d) with hf
  -- `φ` is injective on the index set.
  have hφinj : Set.InjOn φ ↑(Finset.univ : Finset (Fin ((Function.update I n Jn) n))) := by
    intro a _ b _ hab
    have h := congrFun hab n
    simp only [hφ, dif_pos] at h
    simpa using h
  symm
  -- The RHS sum (over the free mode-`n` index) equals the sum of `f` over the
  -- image of `φ`; then extend back to all multi-indices since `f` vanishes off
  -- the image (the off-mode identity factors force a zero).
  trans (∑ jdx ∈ Finset.univ.image φ, f jdx)
  · rw [Finset.sum_image (fun a ha b hb => hφinj (Finset.mem_coe.mpr ha) (Finset.mem_coe.mpr hb))]
    refine Finset.sum_congr rfl (fun iₙ _ => ?_)
    simp only [hf]
    -- Split the product at mode `n`.
    rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
          (fun d => (singleModeFactor n P) d (idx d) (φ iₙ d)) (Finset.mem_univ n)]
    -- Off-mode factors are all `1` because `φ iₙ d` matches `idx d` there.
    have hoff : ∀ d ∈ (Finset.univ : Finset (Fin D)).erase n,
        (singleModeFactor n P) d (idx d) (φ iₙ d) = 1 := by
      intro d hd
      have hdn : d ≠ n := Finset.ne_of_mem_erase hd
      have hcol : idx d = (φ iₙ d).cast (Function.update_of_ne hdn Jn I) := by
        apply Fin.ext; simp [hφ, dif_neg hdn]
      simp only [singleModeFactor, dif_neg hdn, Matrix.submatrix_apply, id, Matrix.one_apply]
      rw [if_pos hcol]
    rw [Finset.prod_congr rfl hoff, Finset.prod_const_one, mul_one]
    -- Match the mode-`n` factor (the submatrix entry of `P`) and the
    -- `X`-arguments, reordering the product.
    rw [mul_comm]
    congr 1
    · -- The `X`-arguments agree pointwise.
      congr 1
      funext d
      by_cases hdn : d = n
      · subst hdn
        simp only [hφ, dif_pos rfl, ↓reduceDIte]
      · simp only [hφ, dif_neg hdn, hdn, ↓reduceDIte]
        apply Fin.ext; simp
    · -- The mode-`n` factor equals the reindexed `P`-entry.
      simp only [singleModeFactor, Matrix.submatrix_apply, id]
      congr 1
      apply Fin.ext; simp [hφ]
  · -- Off the image of `φ`, the off-mode identity factors force `f = 0`.
    apply Finset.sum_subset (Finset.subset_univ _)
    intro jdx _ hjdx
    -- `jdx` is not of the form `φ iₙ`, so it disagrees with `idx` at some `d ≠ n`.
    have hne : ∃ d, d ≠ n ∧ (jdx d).val ≠ (idx d).val := by
      by_contra hc
      push Not at hc
      apply hjdx
      rw [Finset.mem_image]
      refine ⟨jdx n, Finset.mem_univ _, ?_⟩
      funext d
      simp only [hφ]
      by_cases hdn : d = n
      · subst hdn; simp only [↓reduceDIte]
      · simp only [dif_neg hdn]
        apply Fin.ext
        simp only [Fin.val_cast]
        exact (hc d hdn).symm
    obtain ⟨d, hdn, hval⟩ := hne
    simp only [hf]
    rw [Finset.prod_eq_zero (Finset.mem_univ d), mul_zero]
    simp only [singleModeFactor, dif_neg hdn, Matrix.submatrix_apply, id, Matrix.one_apply]
    rw [if_neg]
    intro hcontra
    apply hval
    have h2 : (idx d).val = ((jdx d).cast (Function.update_of_ne hdn Jn I)).val := by rw [← hcontra]
    simpa using h2.symm

/-! ### Non-uniqueness -/

/-- **Non-uniqueness (eq:Tucker-3order-transform), general D.** For invertible
    `P d`, pushing invertible transforms into the core and their inverses into
    the factors leaves the tensor unchanged. One-line corollary of the
    composition lemma. -/
theorem tuckerTensor_transform {D : ℕ} {I J : Fin D → ℕ} (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜)
    (P : (d : Fin D) → Matrix (Fin (J d)) (Fin (J d)) 𝕜)
    (hP : ∀ d, IsUnit (P d).det) :
    tuckerTensor S U
      = tuckerTensor (tuckerTensor S P) (fun d => U d * (P d)⁻¹) := by
  rw [tuckerTensor_tuckerTensor]
  congr 1
  funext d
  rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ (hP d), Matrix.mul_one]

/-! ### Tucker ↔ CP -/

/-- **Tucker induces CP.** The Tucker reconstruction is a CP sum over core
    multi-indices, with weight `S jdx` and rank-one factor columns
    `fun i => U d i (jdx d)`. Hence `cpRank` is bounded by `∏ d, J d`.

    We build a `CPDecomp 𝕜 D I (∏ d, J d)` by reindexing core multi-indices
    through `multiIndexEquiv J`, absorbing the scalar `S jdx` into the mode-0
    factor column. Requires `0 < D` (an order-0 tensor is a single scalar and
    the CP-rank framework is degenerate there). -/
theorem cpRank_tuckerTensor_le {D : ℕ} {I J : Fin D → ℕ} (hD : 0 < D)
    (S : Tensor 𝕜 D J) (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    cpRank (tuckerTensor S U) ≤ ∏ d, J d := by
  -- Build the CP decomposition with rank ∏ d, J d, indexed by core multi-indices.
  -- Absorb the scalar `S jdx` into the mode-0 factor column.
  refine cpRank_le_of_cpDecomp
    { factors := fun r d i =>
        if d = ⟨0, hD⟩ then
          S ((multiIndexEquiv J).symm r) * U d i ((multiIndexEquiv J).symm r d)
        else U d i ((multiIndexEquiv J).symm r d) } ?_
  ext idx
  simp only [CPDecomp.tensor, tuckerTensor_apply]
  -- RHS: ∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)
  -- LHS: ∑ r, ∏ d, factors r d (idx d), reindex r ↦ jdx via multiIndexEquiv J.
  rw [← Equiv.sum_comp (multiIndexEquiv J)
        (fun r => ∏ d : Fin D,
          (if d = ⟨0, hD⟩ then
            S ((multiIndexEquiv J).symm r) * U d (idx d) ((multiIndexEquiv J).symm r d)
          else U d (idx d) ((multiIndexEquiv J).symm r d)))]
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  simp only [Equiv.symm_apply_apply]
  -- Split the product at mode 0 to extract the absorbed scalar.
  rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
        (fun d => if d = ⟨0, hD⟩ then
          S jdx * U d (idx d) (jdx d) else U d (idx d) (jdx d))
        (Finset.mem_univ ⟨0, hD⟩)]
  rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
        (fun d => U d (idx d) (jdx d)) (Finset.mem_univ ⟨0, hD⟩)]
  rw [if_pos rfl]
  rw [Finset.prod_congr rfl (fun d hd => by
    rw [if_neg (Finset.ne_of_mem_erase hd)])]
  ring

/-- **CP (σ-form) induces Tucker with a diagonal core.** With all core
    dimensions equal to `R`, the σ-form CP `⟦σ ;; U⁰, …⟧` equals the Tucker
    reconstruction whose core is the diagonal tensor `diagtensor(σ)`. The core
    sum collapses to the diagonal `jdx = fun _ => r`. -/
theorem cpWeighted_eq_tuckerTensor_diagonal {D : ℕ} {I : Fin D → ℕ} {R : ℕ}
    (hD : 0 < D) (σ : Fin R → 𝕜)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin R) 𝕜)
    (cp : CPDecomp 𝕜 D I R)
    (hcp : ∀ r d i, cp.factors r d i = U d i r) :
    cp.tensorWeighted σ = tuckerTensor (CubicTensor.diagonal σ) U := by
  ext idx
  simp only [CPDecomp.tensorWeighted, tuckerTensor_apply, hcp]
  -- RHS: ∑ jdx : (d → Fin R), diagonal σ jdx * ∏ d, U d (idx d) (jdx d)
  -- diagonal σ jdx = 0 unless jdx is the constant tuple fun _ => r.
  symm
  -- Summand of the RHS over core multi-indices.
  set g : ((d : Fin D) → Fin R) → 𝕜 :=
    fun jdx => (CubicTensor.diagonal σ) jdx * ∏ d : Fin D, U d (idx d) (jdx d) with hg
  -- The injective constant-tuple map `c r = (fun _ => r)`.
  set c : Fin R → ((d : Fin D) → Fin R) := fun r => (fun _ : Fin D => r) with hc
  have hcinj : Set.InjOn c ↑(Finset.univ : Finset (Fin R)) := by
    intro a _ b _ hab
    have := congrFun hab ⟨0, hD⟩
    simpa [hc] using this
  -- Step 1: the full RHS sum equals the sum over the image of `c`, since `g`
  -- vanishes off the constant tuples.
  have hsubset : (Finset.univ.image c).sum g = (Finset.univ : Finset _).sum g := by
    apply Finset.sum_subset (Finset.subset_univ _)
    intro jdx _ hjdx
    have hnc : ¬ ∀ d₁ d₂ : Fin D, jdx d₁ = jdx d₂ := by
      intro hconst
      apply hjdx
      rw [Finset.mem_image]
      refine ⟨jdx ⟨0, hD⟩, Finset.mem_univ _, ?_⟩
      simp only [hc]
      exact funext (fun d => (hconst d ⟨0, hD⟩).symm)
    simp only [CubicTensor.diagonal, hnc, if_false, zero_mul]
  -- Step 2: sum over the image is the reindexed sum, and `g (c r) = σ r * …`.
  calc (Finset.univ : Finset _).sum g
      = (Finset.univ.image c).sum g := hsubset.symm
    _ = ∑ r : Fin R, g (c r) := Finset.sum_image (fun a _ b _ => hcinj (by simp) (by simp))
    _ = ∑ r : Fin R, σ r * ∏ d : Fin D, U d (idx d) r := by
        refine Finset.sum_congr rfl (fun r _ => ?_)
        simp only [hg, hc, CubicTensor.diagonal_const_apply σ r hD]

/-- Wrapper specialising `cpWeighted_eq_tuckerTensor_diagonal` to the
    "canonical" CP whose factor entries are exactly the `U`-columns. This
    removes the auxiliary `cp`/`hcp` parameters; pass only `U` and `σ`. -/
theorem tensorWeighted_eq_tuckerTensor_diagonal {D : ℕ} {I : Fin D → ℕ} {R : ℕ}
    (hD : 0 < D) (σ : Fin R → 𝕜)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin R) 𝕜) :
    (CPDecomp.mk (fun r d i => U d i r) : CPDecomp 𝕜 D I R).tensorWeighted σ
      = tuckerTensor (CubicTensor.diagonal σ) U :=
  cpWeighted_eq_tuckerTensor_diagonal hD σ U
    (CPDecomp.mk (fun r d i => U d i r)) (fun _ _ _ => rfl)

/-! ### Order-3 display sugar: the Tucker bracket `⟦S ⨾ U, V, W⟧`

The bracket `⟦S ⨾ U⁽⁰⁾, …, U⁽ᴰ⁻¹⁾⟧` is display sugar for `tuckerTensor S U`
on order ≤ 3 examples (the function form `tuckerTensor S U` is the primary,
general-`D` API). The separator `⨾` (U+2A3E) is parse-distinct from the
plain CP bracket `⟦…⟧` and the σ-form `⟦σ ;; …⟧`.

Unlike the CP brackets, each factor `U⁽ᵈ⁾ : Matrix (Fin (I d)) (Fin (J d)) 𝕜`
has a *different* type per mode, so the factor dispatcher is a dependent
`Fin.cases` chain returning matrices (not columns). -/

/-- Build the dependent `Fin.cases` chain dispatching `d : Fin D` to the
    appropriate factor *matrix*. Helper macro for `⟦S ⨾ …⟧`.

    Given factors `U₁, …, U_D` (as syntax), produces the term
    `Fin.cases U₁ (Fin.cases U₂ (… Fin.elim0))` of type
    `(d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜`. -/
syntax (name := tuckerCases) "tuckerCases%" "[" term,* "]" : term

macro_rules
  | `(tuckerCases% []) => `(fun i => i.elim0)
  | `(tuckerCases% [$a:term]) => `(Fin.cases $a (fun j => j.elim0))
  | `(tuckerCases% [$a:term, $as:term,*]) => `(Fin.cases $a (tuckerCases% [$as,*]))

/-- Tucker bracket notation. `⟦S ⨾ U₁, U₂, …, U_D⟧ := tuckerTensor S U`,
    where `U` is the dependent factor dispatcher. Requires `D ≥ 2`: the
    single-element form `⟦S ⨾ U⟧` has no matching `macro_rules` clause (the
    rule below needs at least two comma-separated factors) and will not
    elaborate; use `tuckerTensor` directly for `D ≤ 1`. -/
scoped syntax (name := tuckerBrackets) "⟦" term " ⨾ " term,+ "⟧" : term

macro_rules
  | `(⟦$s:term ⨾ $a:term, $bs:term,*⟧) =>
      `(tuckerTensor $s (tuckerCases% [$a, $bs,*]))

/-! #### Per-arity reduction lemmas for the Tucker bracket -/

section Bracket2

variable {I₀ I₁ J₀ J₁ : ℕ} (S : Tensor 𝕜 2 ![J₀, J₁])
  (U : Matrix (Fin I₀) (Fin J₀) 𝕜) (V : Matrix (Fin I₁) (Fin J₁) 𝕜)

/-- Pointwise reduction of the order-2 Tucker bracket. -/
theorem tuckerBracket_apply₂ (idx : (d : Fin 2) → Fin (![I₀, I₁] d)) :
    (⟦S ⨾ U, V⟧ : Tensor 𝕜 2 ![I₀, I₁]) idx =
      ∑ jdx : (d : Fin 2) → Fin (![J₀, J₁] d),
        S jdx * (U (idx 0) (jdx 0) * V (idx 1) (jdx 1)) := by
  simp only [tuckerTensor_apply]
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  rw [show (Finset.univ : Finset (Fin 2)) = {0, 1} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_singleton]
  rfl

end Bracket2

section Bracket3

variable {I₀ I₁ I₂ J₀ J₁ J₂ : ℕ} (S : Tensor 𝕜 3 ![J₀, J₁, J₂])
  (U : Matrix (Fin I₀) (Fin J₀) 𝕜) (V : Matrix (Fin I₁) (Fin J₁) 𝕜)
  (W : Matrix (Fin I₂) (Fin J₂) 𝕜)

/-- Pointwise reduction of the order-3 Tucker bracket. -/
theorem tuckerBracket_apply₃ (idx : (d : Fin 3) → Fin (![I₀, I₁, I₂] d)) :
    (⟦S ⨾ U, V, W⟧ : Tensor 𝕜 3 ![I₀, I₁, I₂]) idx =
      ∑ jdx : (d : Fin 3) → Fin (![J₀, J₁, J₂] d),
        S jdx * (U (idx 0) (jdx 0) * V (idx 1) (jdx 1) * W (idx 2) (jdx 2)) := by
  simp only [tuckerTensor_apply]
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  rw [show (Finset.univ : Finset (Fin 3)) = {0, 1, 2} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
    Finset.prod_singleton]
  change S jdx * (U (idx 0) (jdx 0) * (V (idx 1) (jdx 1) * W (idx 2) (jdx 2))) = _
  ring

end Bracket3

/-! ### Order-2 sanity check (SVD-compression picture) -/

/-- **Order-2 Tucker equals the matrix product `U * S * Vᵀ`.** This is the
    SVD-compression picture of the introduction: with an order-2 core read as
    the matrix `S`, the order-2 Tucker reconstruction `⟦S ⨾ U, V⟧`, read as a
    matrix, is `U S Vᵀ`. Mirrors `cpWeighted_two_eq_matrix_mul`. -/
theorem tuckerTensor_two_eq_matrix_mul {I₀ I₁ J₀ J₁ : ℕ}
    (S : Matrix (Fin J₀) (Fin J₁) 𝕜)
    (U : Matrix (Fin I₀) (Fin J₀) 𝕜) (V : Matrix (Fin I₁) (Fin J₁) 𝕜)
    (i : Fin I₀) (j : Fin I₁) :
    (⟦(fun jdx : (d : Fin 2) → Fin (![J₀, J₁] d) => S (jdx 0) (jdx 1)) ⨾ U, V⟧ :
        Tensor 𝕜 2 ![I₀, I₁])
        (fun d => match d with | ⟨0, _⟩ => i | ⟨1, _⟩ => j)
      = (U * S * Vᵀ) i j := by
  rw [tuckerBracket_apply₂]
  -- LHS: ∑ jdx, S (jdx 0) (jdx 1) * (U i (jdx 0) * V j (jdx 1))
  -- Flatten the Pi-sum over `Fin 2 → Fin _` to a sum over `Fin J₀ × Fin J₁`.
  rw [← Equiv.sum_comp (piFinTwoEquiv (fun d => Fin (![J₀, J₁] d))).symm
        (fun jdx => S (jdx 0) (jdx 1) * (U i (jdx 0) * V j (jdx 1)))]
  rw [Fintype.sum_prod_type]
  -- Reduce the `piFinTwoEquiv.symm` projections to the pair components. LHS is now
  -- `∑ j₀, ∑ j₁, S j₀ j₁ * (U i j₀ * V j j₁)`.
  simp only [piFinTwoEquiv, Equiv.coe_fn_symm_mk, Fin.cons_zero, Fin.cons_one]
  -- RHS: (U * S * Vᵀ) i j = ∑ j₁, (∑ j₀, U i j₀ * S j₀ j₁) * Vᵀ j₁ j.
  rw [Matrix.mul_apply]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Finset.sum_mul]
  -- RHS is `∑ j₁, ∑ j₀, U i j₀ * S j₀ j₁ * V j j₁`; swap to `∑ j₀, ∑ j₁`.
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun j₀ _ => ?_)
  refine Finset.sum_congr rfl (fun j₁ _ => ?_)
  ring

end Tensor
