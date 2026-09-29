/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import TensorCourse.Tensor.OuterProd
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.KhatriRao
import TensorCourse.Matrix.KRank
import TensorCourse.Matrix.KroneckerLemmas
import Mathlib.GroupTheory.Perm.Basic

/-!
# Factor Equivalence and Kruskal's Uniqueness Theorem

This file introduces the **factor-equivalence** predicate for CP
decompositions: two `R`-term CP decompositions of an order-`D` tensor
are factor-equivalent if their rank-1 summands agree, pointwise as
tensors, after a permutation of the rank index. This is the lecture's
``equality up to scaling and permutation'' for the rank-1 outer
products.

## Main definitions

* `Tensor.rankOneTerm cp r` — the `r`-th rank-1 component tensor of a
  CP decomposition: `(idx ↦ ∏ d, cp.factors r d (idx d))`.
* `Tensor.FactorEquivD cp₁ cp₂` — general-`D` factor-equivalence
  predicate: there is a permutation `π : Equiv.Perm (Fin R)` such that
  the `r`-th rank-1 term of `cp₁` equals the `(π r)`-th rank-1 term of
  `cp₂` for every `r`.
* `Tensor.FactorEquiv₃` — order-3 specialisation, defined here. The
  legacy definition in `SimultaneousDiagonalization.lean` has been
  moved here.

## Main results

* `Tensor.FactorEquivD.toFactorEquiv₃` — bridging lemma: a general-`D`
  factor-equivalence specialised to `D = 3` and `dims = ![I, J, K]`
  yields an order-3 factor-equivalence (`FactorEquiv₃`).

## Axioms

* `Tensor.kruskal_uniqueness` (axiom A9) — Kruskal's CP uniqueness
  theorem: a sufficient condition for two `R`-term CP decompositions of
  the same tensor to be factor-equivalent is that the sum of the k-ranks
  of one decomposition's factor matrices is at least `2R + (D − 1)`. The
  proof (Sidiropoulos and Bro 2000) is non-elementary and is not
  currently in Mathlib; we axiomatise it. The axiom is invoked by
  `simDiag_uniqueness` to close the lecture's Theorem
  `thm:simultaneous-diagonalization` (claim 1).

## References

* Sidiropoulos, N. D. and Bro, R., "On the uniqueness of multilinear
  decomposition of N-way arrays", Journal of Chemometrics 14 (2000),
  229-239.
* Kruskal, J. B., "Three-way arrays: rank and uniqueness of trilinear
  decompositions", Linear Algebra Appl. 18 (1977), 95-138.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Rank-1 terms of a CP decomposition -/

/-- The `r`-th rank-1 component tensor of a CP decomposition: the
    tensor whose value at multi-index `idx` is `∏ d, cp.factors r d (idx d)`. -/
def CPDecomp.rankOneTerm {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}
    (cp : CPDecomp 𝕜 D dims R) (r : Fin R) : Tensor 𝕜 D dims :=
  fun idx => ∏ d : Fin D, cp.factors r d (idx d)

@[simp]
theorem CPDecomp.rankOneTerm_apply {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}
    (cp : CPDecomp 𝕜 D dims R) (r : Fin R)
    (idx : (d : Fin D) → Fin (dims d)) :
    cp.rankOneTerm r idx = ∏ d : Fin D, cp.factors r d (idx d) := rfl

/-! ### General-`D` factor equivalence -/

/-- **Factor equivalence (general `D`).** Two `R`-term CP decompositions
    `cp₁`, `cp₂` of a tensor of order `D` are *factor-equivalent* if
    their rank-1 components agree as tensors under a permutation of the
    rank index. This is the natural ``up to scaling and permutation''
    equivalence: the rank-1 outer product `u₀ ◯ u₁ ◯ ⋯ ◯ u_{D-1}` is
    invariant under simultaneously rescaling the factors `uᵢ ↦ λᵢ uᵢ`
    with `∏ λᵢ = 1`, and the predicate quotients by exactly this
    ambiguity by working with rank-1 component tensors directly. -/
def FactorEquivD {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}
    (cp₁ cp₂ : CPDecomp 𝕜 D dims R) : Prop :=
  ∃ π : Equiv.Perm (Fin R),
    ∀ r : Fin R, cp₁.rankOneTerm r = cp₂.rankOneTerm (π r)

/-! ### Order-3 factor equivalence -/

/-- Two order-3 CP decompositions of the same tensor type are *factor-
    equivalent* if their rank-1 summands agree pointwise after a
    permutation of the rank index. The equality is stated at the level
    of the rank-1 *outer product* `(uᵢ ◯ₜ vᵢ ◯ₜ wᵢ)`, not the
    individual factors — this is the weaker ``up to scaling and
    permutation'' equivalence used in the lecture. -/
def FactorEquiv₃ {I J K R : ℕ}
    (cp₁ cp₂ : CPDecomp 𝕜 3 ![I, J, K] R) : Prop :=
  ∃ π : Equiv.Perm (Fin R),
    ∀ r : Fin R,
      (Tensor.toTensor1 (fun i : Fin I => cp₁.factors r 0 i) ◯ₜ
          Tensor.toTensor1 (fun j : Fin J => cp₁.factors r 1 j) ◯ₜ
          Tensor.toTensor1 (fun k : Fin K => cp₁.factors r 2 k)) =
        (Tensor.toTensor1 (fun i : Fin I => cp₂.factors (π r) 0 i) ◯ₜ
          Tensor.toTensor1 (fun j : Fin J => cp₂.factors (π r) 1 j) ◯ₜ
          Tensor.toTensor1 (fun k : Fin K => cp₂.factors (π r) 2 k))

/-! ### Bridging `FactorEquivD` (at `D = 3`) to `FactorEquiv₃` -/

/-- For a CP decomposition `cp : CPDecomp 𝕜 3 ![I, J, K] R`, the `r`-th
    rank-1 term, evaluated at `(i, j, k)`, equals
    `cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 k`. -/
theorem CPDecomp.rankOneTerm_three_apply
    {I J K R : ℕ} (cp : CPDecomp 𝕜 3 ![I, J, K] R) (r : Fin R)
    (idx : (d : Fin 3) → Fin (![I, J, K] d)) :
    cp.rankOneTerm r idx =
      cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) * cp.factors r 2 (idx 2) := by
  rw [CPDecomp.rankOneTerm_apply, Fin.prod_univ_three]

/-- **Bridge.** The general-`D` factor equivalence `FactorEquivD`
    specialised to `D = 3` and `dims = ![I, J, K]` implies the
    order-3 specific `FactorEquiv₃`. The two predicates are pointwise
    equal as functions of the multi-index; this conversion repackages
    the rank-1 product as a triple outer product of order-1 tensors. -/
theorem FactorEquivD.toFactorEquiv₃
    {I J K R : ℕ} {cp₁ cp₂ : CPDecomp 𝕜 3 ![I, J, K] R}
    (h : FactorEquivD cp₁ cp₂) : FactorEquiv₃ cp₁ cp₂ := by
  obtain ⟨π, hπ⟩ := h
  refine ⟨π, fun r => ?_⟩
  -- Show the pointwise equality on (i, j, k) multi-indices.
  ext idx
  -- LHS = u(idx_left_left) * v(idx_left_right) * w(idx_right);
  -- after splitLeft/splitRight reductions and `Tensor.toTensor1_apply`,
  -- the entries are `factors r 0 (idx 0)`, `factors r 1 (idx 1)`,
  -- `factors r 2 (idx 2)` with type-equal but definitionally cast indices.
  -- The hypothesis `hπ r` says the order-3 rank-1 term tensors agree;
  -- we evaluate it at the corresponding multi-index.
  -- The output type of the triple outer product is
  -- `Tensor 𝕜 (1 + 1 + 1) (combinedDims (combinedDims _ _) _)` whose
  -- multi-index `idx` is *not* the same type as the order-3 multi-index
  -- on `![I, J, K]`. We construct the corresponding order-3 multi-index
  -- pointwise, prove it equals the appropriate splitLeft/splitRight
  -- coordinates, and apply `hπ r`.
  -- Build the order-3 multi-index `(i, j, k) := (splitLeft (splitLeft idx),
  -- splitRight (splitLeft idx), splitRight idx)`.
  set i : Fin I := splitLeft (splitLeft idx) 0 with hi_def
  set j : Fin J := splitRight (splitLeft idx) 0 with hj_def
  set k : Fin K := splitRight idx 0 with hk_def
  -- Define a Fin 3 multi-index from (i, j, k).
  let idx3 : (d : Fin 3) → Fin (![I, J, K] d) := fun d =>
    match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => k
    | ⟨n+3, h⟩ => absurd h (by omega)
  -- Apply hypothesis pointwise at idx3.
  have hr := congrFun (hπ r) idx3
  rw [CPDecomp.rankOneTerm_three_apply, CPDecomp.rankOneTerm_three_apply] at hr
  -- hr : cp₁.factors r 0 (idx3 0) * cp₁.factors r 1 (idx3 1) * cp₁.factors r 2 (idx3 2) =
  --      cp₂.factors (π r) 0 (idx3 0) * ... * cp₂.factors (π r) 2 (idx3 2)
  -- The `idx3 0 = i`, `idx3 1 = j`, `idx3 2 = k` reductions are definitional.
  -- Now compute LHS and RHS of the goal by unfolding the triple outer product.
  change ((Tensor.toTensor1 (fun i : Fin I => cp₁.factors r 0 i) ◯ₜ
          Tensor.toTensor1 (fun j : Fin J => cp₁.factors r 1 j) ◯ₜ
          Tensor.toTensor1 (fun k : Fin K => cp₁.factors r 2 k))) idx =
        ((Tensor.toTensor1 (fun i : Fin I => cp₂.factors (π r) 0 i) ◯ₜ
          Tensor.toTensor1 (fun j : Fin J => cp₂.factors (π r) 1 j) ◯ₜ
          Tensor.toTensor1 (fun k : Fin K => cp₂.factors (π r) 2 k))) idx
  simp only [outerProd_apply, Tensor.toTensor1_apply]
  exact hr

/-! ### Kruskal's CP uniqueness theorem (axiom A9) -/

/-- **Kruskal's CP uniqueness theorem (axiom A9).** A sufficient
    condition for two `R`-term CP decompositions `cp` and `cp'` of the
    same order-`D` tensor to be factor-equivalent (in the sense of
    `FactorEquivD`) is that the sum of the k-ranks of one
    decomposition's factor matrices is at least `2R + (D − 1)`.

    This is the lecture's Theorem `thm:kruskal`. It is stated as an
    axiom because the proof (Sidiropoulos and Bro 2000, generalising
    Kruskal 1977) is non-elementary and is not currently in Mathlib.

    Used in `simDiag_uniqueness` (the lecture's Theorem
    `thm:simultaneous-diagonalization`, claim 1) by combining with the
    bridging lemmas of `Matrix.kRank` showing that the simultaneous-
    diagonalization hypotheses imply
    `kᵤ + kᵥ + kᵥᵥ ≥ 2R + 2 = 2R + (3 − 1)`.

    Reference: Sidiropoulos, N. D. and Bro, R., "On the uniqueness of
    multilinear decomposition of N-way arrays", Journal of Chemometrics
    14 (2000), 229-239. -/
axiom kruskal_uniqueness
    {𝕜 : Type*} [RCLike 𝕜] {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}
    (cp cp' : CPDecomp 𝕜 D dims R)
    (heq : cp.tensor = cp'.tensor)
    (h : (∑ n : Fin D, Matrix.kRank (cp.factorMatrix n)) ≥ 2 * R + (D - 1)) :
    FactorEquivD cp cp'

end Tensor

end
