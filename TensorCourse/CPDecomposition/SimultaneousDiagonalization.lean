/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.VNModeProd
import TensorCourse.Tensor.OuterProd
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.KhatriRao
import TensorCourse.CPDecomposition.FactorEquiv
import TensorCourse.Matrix.KhatriRao
import TensorCourse.Matrix.KroneckerLemmas
import TensorCourse.Matrix.PinvAxioms
import TensorCourse.Matrix.KRank
import Mathlib.LinearAlgebra.LinearIndependent.Defs
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.GroupTheory.Perm.Basic

/-!
# Simultaneous Diagonalization Algorithm for CP — structural results

This file collects the algebraic content of the simultaneous-diagonalization
algorithm for order-3 CP decomposition (Sanchez 1990 "direct trilinear
decomposition", modern form Leurgans-Ross-Abel 1993). The lecture states
this material in §"Simultaneous Diagonalization Algorithm for CP".

## Main definitions

* `Tensor.FactorEquiv₃` — equivalence of two order-3 CP factor lists up
  to permutation, formulated as equality of rank-1 outer products
  (matching the lecture verbatim, weaker than factor-by-factor equality).

## Main results

* `Tensor.cpDecomp_three_vNModeProd_two_eq_matrix_mul` — the matrix-pencil
  structural lemma. For an order-3 plain CP decomposition `cp` and a
  contraction vector `a : Fin K → 𝕜` along mode 2, the contracted tensor
  `cp.tensor ×̄₃ a`, evaluated pointwise at index `(i, j)`, equals the
  `(i, j)`-entry of the matrix product

  `cp.factorMatrix 0 * Matrix.diagonal ((cp.factorMatrix 2)ᵀ *ᵥ a) * (cp.factorMatrix 1)ᵀ`.

  This is the matrix pencil of the algorithm: the family
  `{cp.tensor ×̄₃ a : a ∈ Fin K → 𝕜}` of matrices that the algorithm
  simultaneously diagonalises.

* `Tensor.khatriRao_full_col_rank_of_lin_indep` — Khatri-Rao full column
  rank: if the columns of `U` and `V` are each linearly independent, then
  the columns of `V ⊙ₜ U` are linearly independent.

* `Tensor.simDiag_uniqueness` — the Leurgans-Ross-Abel uniqueness
  theorem for order-3 CP under simultaneous-diagonalization-style
  independence hypotheses. Proved by reduction to Kruskal's theorem
  (axiom A9, `Tensor.kruskal_uniqueness`).

## Notation discipline

Per project convention (and `plan_lecture4.md` guideline 5), unconjugated
dot/matrix-vector products use Mathlib's `Matrix.dotProduct` (`⬝ᵥ`) and
`Matrix.mulVec` (`*ᵥ`). The tensor inner product `⟪·,·⟫ₜ` is
*conjugate*-linear in its first argument and would silently change
semantics over `ℂ`; it is not used here.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### CP-specific genericity axiom -/

/-- **Existence of generic contraction vectors.** Given `2 ≤ R` and
    `W : K × R` whose columns are pairwise non-collinear (no column is a
    scalar multiple of another), there exist contraction vectors
    `a, b : Fin K → 𝕜` such that `Wᵀ *ᵥ a` and `Wᵀ *ᵥ b` have no zero
    entries and the eigenvalue ratios `(Wᵀ *ᵥ a) r / (Wᵀ *ᵥ b) r` are
    pairwise distinct (in cross-multiplied form).

    This is the lecture's "almost surely" assumption made explicit: the
    bad set in `(a, b)`-space is a finite union of proper algebraic
    subvarieties (hence Lebesgue-measure zero); existence follows. We
    axiomatize because Mathlib lacks a direct "Zariski-dense set has
    full Lebesgue measure" lemma at matrix-level.

    **The hypothesis `2 ≤ R` is essential.** With `R ≥ 2` and pairwise
    non-collinearity (`hW`), each column of `W` is forced to be nonzero
    (a zero column would be a `0`-multiple of any other column, violating
    `hW`), so the bad set really is a proper subvariety and the statement
    is the true Zariski-density fact. Without `2 ≤ R` the statement is
    vacuous and in fact inconsistent: for `R ≤ 1` the hypothesis `hW`
    holds vacuously (there are no distinct pairs `r ≠ t`), yet the
    conclusion would assert that `Wᵀ *ᵥ a` is everywhere nonzero — false
    for `W = 0`, where `Wᵀ *ᵥ a = 0` for every `a`. -/
axiom exists_generic_contraction
    {K R : ℕ} (hr : 2 ≤ R)
    (W : Matrix (Fin K) (Fin R) 𝕜)
    (hW : ∀ r t : Fin R, r ≠ t →
      ¬ ∃ c : 𝕜, (fun k : Fin K => W k r) = c • (fun k : Fin K => W k t)) :
    ∃ a b : Fin K → 𝕜,
      (∀ r : Fin R, (Wᵀ *ᵥ a) r ≠ 0) ∧
      (∀ r : Fin R, (Wᵀ *ᵥ b) r ≠ 0) ∧
      (∀ r t : Fin R, r ≠ t →
        (Wᵀ *ᵥ a) r * (Wᵀ *ᵥ b) t ≠ (Wᵀ *ᵥ a) t * (Wᵀ *ᵥ b) r)

/-! ### Matrix-pencil structural lemma -/

/-- **Matrix-pencil structural lemma (Lemma in §``Simultaneous
    Diagonalization Algorithm for CP'').** For an order-3 plain CP
    decomposition `cp : CPDecomp 𝕜 3 ![I, J, K] R` and any contraction
    vector `a : Fin K → 𝕜` along mode 2, the contracted tensor
    `cp.tensor ×̄₃ a` is, pointwise at multi-index `(i, j)`, the
    `(i, j)`-entry of the matrix

    `U * diagonal(Wᵀ *ᵥ a) * Vᵀ`,

    where `U := cp.factorMatrix 0`, `V := cp.factorMatrix 1`,
    `W := cp.factorMatrix 2` are the column-stacked factor matrices.

    The diagonal weight is `(Wᵀ *ᵥ a) r = ∑ k, W k r * a k = ∑ k,
    cp.factors r 2 k * a k`, the lecture's `D_a` matrix.

    The two matrices `cp.tensor ×̄₃ a` and `cp.tensor ×̄₃ b` arising
    from independent contraction vectors `a, b` form the *matrix
    pencil* that the simultaneous-diagonalization algorithm
    diagonalises.

    The output type of `cp.tensor.vNModeProd 2 a` is
    `Tensor 𝕜 2 (fun d => ![I,J,K] ((2 : Fin 3).succAbove d))`; for
    `d = 0, 1` the indices `(2 : Fin 3).succAbove 0` and
    `(2 : Fin 3).succAbove 1` reduce by `rfl` to `0` and `1`
    respectively, so the output multi-index pair `(idx 0, idx 1)`
    naturally corresponds to a matrix entry `(i, j) : Fin I × Fin J`. -/
theorem cpDecomp_three_vNModeProd_two_eq_matrix_mul
    {I J K R : ℕ}
    (cp : CPDecomp 𝕜 3 ![I, J, K] R)
    (a : Fin K → 𝕜)
    (idx : (d : Fin 2) → Fin (![I, J, K] ((2 : Fin 3).succAbove d))) :
    (cp.tensor.vNModeProd 2 a) idx =
      (cp.factorMatrix 0 * Matrix.diagonal ((cp.factorMatrix 2)ᵀ *ᵥ a)
          * (cp.factorMatrix 1)ᵀ) (idx 0) (idx 1) := by
  -- LHS: expand vNModeProd, then `cp.tensor` to a sum-of-products.
  rw [vNModeProd_apply]
  -- LHS is now `∑ iₙ : Fin K, a iₙ * cp.tensor (Fin.insertNth 2 iₙ idx)`.
  -- Inside, `cp.tensor (Fin.insertNth 2 iₙ idx) = ∑ r, ∏ d, factors r d ((insertNth 2 iₙ idx) d)`.
  -- Step 1: rewrite each summand to expose the rank-1 product factorisation.
  have summand_eq : ∀ k : Fin K,
      a k * cp.tensor (Fin.insertNth (2 : Fin 3) k idx) =
        ∑ r : Fin R,
          a k * (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) *
                  cp.factors r 2 k) := by
    intro k
    -- `cp.tensor x = ∑ r, ∏ d, factors r d (x d)` definitionally.
    -- Use `Fin.insertNth (α := fun d => Fin (![I,J,K] d))` to pin the family.
    change a k * (∑ r : Fin R, ∏ d : Fin 3,
        cp.factors r d
          (Fin.insertNth (α := fun d : Fin 3 => Fin (![I, J, K] d))
            (2 : Fin 3) k idx d)) = _
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    -- Expand the product over `Fin 3` as `f 0 * f 1 * f 2`.
    rw [show (Finset.univ : Finset (Fin 3)) = {0, 1, 2} from rfl]
    rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
        Finset.prod_singleton]
    -- Compute the three `Fin.insertNth 2 k idx` lookups, with the `α` family
    -- pinned to `fun d => Fin (![I,J,K] d)` to match the surrounding context.
    set α : Fin 3 → Type := fun d => Fin (![I, J, K] d) with hα
    have hi0 : (Fin.insertNth (α := α) (2 : Fin 3) k idx) 0 = idx 0 := by
      have hidx : ((2 : Fin 3).succAbove (0 : Fin 2) : Fin 3) = (0 : Fin 3) := rfl
      have h := Fin.insertNth_apply_succAbove (α := α)
        (i := (2 : Fin 3)) (x := k) (p := idx) 0
      -- `h : Fin.insertNth ... ((2:Fin 3).succAbove 0) = idx 0`
      -- Replace `(2:Fin 3).succAbove 0` with `0`.
      have heq_val : ∀ (j j' : Fin 3) (heq : j = j'),
          (Fin.insertNth (α := α) (2 : Fin 3) k idx j) =
          heq ▸ Fin.insertNth (α := α) (2 : Fin 3) k idx j' := by
        intros j j' heq; cases heq; rfl
      have := heq_val ((2 : Fin 3).succAbove 0) 0 hidx
      -- `this : Fin.insertNth ... ((2:Fin 3).succAbove 0) = hidx ▸ ... 0`
      -- But the `α` cast is `Fin (![I,J,K] _) = Fin (![I,J,K] 0)`, both reducing
      -- to `Fin I`, so the `▸` is essentially the identity. Bridge via `Fin.ext`.
      apply Fin.ext
      have := h
      -- Both sides reduce to the same `.val`.
      change (Fin.insertNth (α := α) (2 : Fin 3) k idx 0).val = (idx 0).val
      have key : (Fin.insertNth (α := α) (2 : Fin 3) k idx
          ((2 : Fin 3).succAbove 0)).val = (idx 0).val := by
        rw [h]
      -- `((2 : Fin 3).succAbove 0)` is definitionally `0`, so the .val agrees.
      have : (Fin.insertNth (α := α) (2 : Fin 3) k idx 0).val =
          (Fin.insertNth (α := α) (2 : Fin 3) k idx
            ((2 : Fin 3).succAbove 0)).val := by
        congr 1
      rw [this, key]
    have hi1 : (Fin.insertNth (α := α) (2 : Fin 3) k idx) 1 = idx 1 := by
      have hidx : ((2 : Fin 3).succAbove (1 : Fin 2) : Fin 3) = (1 : Fin 3) := rfl
      have h := Fin.insertNth_apply_succAbove (α := α)
        (i := (2 : Fin 3)) (x := k) (p := idx) 1
      apply Fin.ext
      change (Fin.insertNth (α := α) (2 : Fin 3) k idx 1).val = (idx 1).val
      have key : (Fin.insertNth (α := α) (2 : Fin 3) k idx
          ((2 : Fin 3).succAbove 1)).val = (idx 1).val := by
        rw [h]
      have : (Fin.insertNth (α := α) (2 : Fin 3) k idx 1).val =
          (Fin.insertNth (α := α) (2 : Fin 3) k idx
            ((2 : Fin 3).succAbove 1)).val := by
        congr 1
      rw [this, key]
    have hi2 : (Fin.insertNth (α := α) (2 : Fin 3) k idx) 2 = k :=
      Fin.insertNth_apply_same _ _ _
    rw [hi0, hi1, hi2]
    ring
  -- Transform the LHS via Eq.trans of explicit equalities (avoids the
  -- alpha-equivalence pitfall that has bitten `rw` on `Finset.sum`-shaped
  -- expressions in this proof).
  have step_A : (∑ iₙ : Fin K, a iₙ * cp.tensor (Fin.insertNth 2 iₙ idx)) =
      ∑ iₙ : Fin K, ∑ r : Fin R,
        a iₙ * (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) *
                cp.factors r 2 iₙ) :=
    Finset.sum_congr rfl (fun iₙ _ => summand_eq iₙ)
  have step_B : (∑ iₙ : Fin K, ∑ r : Fin R,
        a iₙ * (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) *
                cp.factors r 2 iₙ)) =
      ∑ r : Fin R, ∑ iₙ : Fin K,
        a iₙ * (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) *
                cp.factors r 2 iₙ) := Finset.sum_comm
  have step_C : (∑ r : Fin R, ∑ iₙ : Fin K,
        a iₙ * (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1) *
                cp.factors r 2 iₙ)) =
      ∑ r : Fin R, cp.factors r 0 (idx 0) *
        ((∑ k : Fin K, cp.factors r 2 k * a k) * cp.factors r 1 (idx 1)) := by
    refine Finset.sum_congr rfl fun r _ => ?_
    -- LHS = ∑ iₙ, a iₙ * (u₀ * u₁ * w iₙ).
    -- RHS = u₀ * ((∑ k, w k * a k) * u₁) where the inner sum scalar
    --       and u₁ commute with the bound-variable-free parts.
    -- Both equal `(u₀ * u₁) * ∑ k, a k * w k`. Compute via `Finset.sum_mul`/
    -- `Finset.mul_sum` and `ring`.
    have lhs_eq : (∑ iₙ : Fin K, a iₙ * (cp.factors r 0 (idx 0) *
            cp.factors r 1 (idx 1) * cp.factors r 2 iₙ)) =
        (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1)) *
          ∑ k : Fin K, a k * cp.factors r 2 k := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => ?_
      ring
    have rhs_eq : cp.factors r 0 (idx 0) *
        ((∑ k : Fin K, cp.factors r 2 k * a k) * cp.factors r 1 (idx 1)) =
        (cp.factors r 0 (idx 0) * cp.factors r 1 (idx 1)) *
          ∑ k : Fin K, a k * cp.factors r 2 k := by
      have hcomm : (∑ k : Fin K, cp.factors r 2 k * a k) =
          ∑ k : Fin K, a k * cp.factors r 2 k :=
        Finset.sum_congr rfl fun k _ => mul_comm _ _
      rw [hcomm]
      ring
    rw [lhs_eq, rhs_eq]
  -- Chain the LHS through to the desired form.
  refine (step_A.trans (step_B.trans step_C)).trans ?_
  -- RHS: expand the matrix product.
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl fun r _ => ?_
  -- RHS r-summand: (factorMatrix 0 * diagonal ...) (idx 0) r * (factorMatrix 1)ᵀ r (idx 1)
  --              = (factorMatrix 0 * diagonal ...) (idx 0) r * factorMatrix 1 (idx 1) r
  rw [Matrix.transpose_apply]
  -- Compute `(factorMatrix 0 * diagonal d) (idx 0) r = factorMatrix 0 (idx 0) r * d r`.
  rw [Matrix.mul_diagonal]
  rw [CPDecomp.factorMatrix_apply, CPDecomp.factorMatrix_apply]
  -- After `Matrix.mul_diagonal`, the goal involves
  -- `(cp.factorMatrix 2)ᵀ *ᵥ a) r` — unfold it to a sum.
  -- We unfold `mulVec`/`dotProduct` directly without specifying a bound-variable
  -- name, then convert to the `k`-named form via `Finset.sum_congr` (alpha
  -- conversion is sound but not syntactic).
  have hdot : ((cp.factorMatrix 2)ᵀ *ᵥ a) r = ∑ k : Fin K, cp.factors r 2 k * a k := by
    change dotProduct ((cp.factorMatrix 2)ᵀ r) a =
        ∑ k : Fin K, cp.factors r 2 k * a k
    refine Finset.sum_congr rfl fun k _ => ?_
    rfl
  rw [hdot]
  ring

/-! ### Khatri-Rao full column rank for vector pairs -/

/-- **Khatri-Rao full column rank.** If the columns of
    `U : Matrix (Fin I) (Fin R) 𝕜` and the columns of
    `V : Matrix (Fin J) (Fin R) 𝕜` are each linearly independent (over
    `Fin R`), then the columns of `V ⊙ₜ U : Matrix (Fin (J * I)) (Fin R) 𝕜`
    are linearly independent.

    **Proof strategy.** Work pointwise with
    `Fintype.linearIndependent_iff`. Suppose `∑ r, α r • (V ⊙ₜ U)·r = 0`.
    The `(j, i)`-entry of this equation (for any `j, i`) is
    `∑ r, α r * V j r * U i r = 0`. Fix `i` and view this as a linear
    combination of the columns of `V` with coefficients `α r * U i r`:
    by linear independence of `V`'s columns, `α r * U i r = 0` for every
    `r` and every `i`. Hence for each `r`, the vector
    `(α r * U i r)_i = α r • U·r` is zero. By linear independence of
    `U`'s columns, `α r = 0`.

    This argument avoids the lecture's "right-multiply by an isolating
    vector" approach (which would require dual-basis machinery) by
    factoring through pointwise linear combinations on each row index
    independently. -/
theorem khatriRao_full_col_rank_of_lin_indep
    {I J R : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜) (V : Matrix (Fin J) (Fin R) 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hV : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V j r)) :
    LinearIndependent 𝕜
      (fun r : Fin R => fun ij : Fin (J * I) => (V ⊙ₜ U) ij r) := by
  rw [Fintype.linearIndependent_iff]
  intro α hα
  -- `hα : ∑ r, α r • (column r of V ⊙ₜ U) = 0`
  -- View pointwise at row index `ij = finProdFinEquiv (j, i)`.
  -- For each pair `(j, i) : Fin J × Fin I`, the `(j, i)`-entry yields
  -- `∑ r, α r * V j r * U i r = 0`.
  have h_pt : ∀ j : Fin J, ∀ i : Fin I,
      ∑ r : Fin R, α r * V j r * U i r = 0 := by
    intro j i
    -- Read off the `ij = finProdFinEquiv (j, i)`-entry of `hα`.
    have h := congrFun hα (finProdFinEquiv (j, i))
    -- LHS of `h` is `(∑ r, α r • (V ⊙ₜ U) ij _column) (finProdFinEquiv (j, i))`,
    -- which by `Pi`/`Finset.sum_apply` and the smul-on-`Fin R → 𝕜` is
    -- `∑ r, α r * (V ⊙ₜ U) (finProdFinEquiv (j, i)) r`.
    -- RHS is `0`.
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul,
      Matrix.khatriRao_apply_pair, Pi.zero_apply] at h
    -- After `khatriRao_apply_pair`, `(V ⊙ₜ U) (finProdFinEquiv (j, i)) r = V j r * U i r`.
    -- So `h : ∑ r, α r * (V j r * U i r) = 0`. Reassociate to match the goal.
    rw [show (∑ r : Fin R, α r * V j r * U i r)
            = ∑ r : Fin R, α r * (V j r * U i r) from by
      refine Finset.sum_congr rfl fun r _ => ?_; ring]
    exact h
  -- For each fixed `i`, view `∑ r, (α r * U i r) * V j r = 0` as a linear
  -- combination of `V`'s columns at coefficient `(α r * U i r)`.
  -- By linear independence of `V`'s columns, `α r * U i r = 0` for all r, i.
  have h_alpha_U : ∀ i : Fin I, ∀ r : Fin R, α r * U i r = 0 := by
    intro i
    -- Apply `Fintype.linearIndependent_iff.mp hV` to coefficients `r ↦ α r * U i r`.
    apply (Fintype.linearIndependent_iff.mp hV) (fun r => α r * U i r)
    funext j
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    -- Goal: ∑ r, (α r * U i r) * V j r = 0. Reorder to ∑ r, α r * V j r * U i r.
    have := h_pt j i
    rw [show (∑ r : Fin R, (α r * U i r) * V j r) =
            ∑ r : Fin R, α r * V j r * U i r from by
        refine Finset.sum_congr rfl fun r _ => by ring]
    exact this
  -- For each fixed `r`, the function `i ↦ α r * U i r = (α r) • U·r` is zero.
  -- By linear independence of `U`'s columns, `α r = 0`.
  -- We apply `Fintype.linearIndependent_iff.mp hU` to coefficients
  -- `t ↦ if t = r then α r else 0` — but a cleaner route is direct:
  -- the family `(fun r => U · r)` linearly independent gives that
  -- `α r • (U · r) = 0` ⟹ `α r = 0`. We use `LinearIndependent.eq_zero_of_smul_mem_span`
  -- or just `linearIndependent_iff'`. Easiest: invoke linear independence via
  -- `Fintype.linearIndependent_iff` once more, applied to the single-coordinate
  -- coefficient family.
  intro r
  -- Use that `U`'s columns are linearly independent in particular `U·r ≠ 0`,
  -- and that `α r • U·r = 0`.
  -- Define β t := if t = r then α r else 0. Then ∑ t, β t • U·t = α r • U·r.
  -- We show α r • U·r = 0 (it's `h_alpha_U i r` for every i, viewed as a
  -- function of i).
  have hzero : (fun i : Fin I => α r * U i r) = 0 := by
    funext i
    exact h_alpha_U i r
  -- Now apply `Fintype.linearIndependent_iff.mp hU` with coefficients
  -- `t ↦ if t = r then α r else 0`.
  classical
  have h := (Fintype.linearIndependent_iff.mp hU)
    (fun t => if t = r then α r else 0) ?_ r
  · simpa using h
  · funext i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply,
      ite_mul, zero_mul]
    -- Goal: (∑ x, if x = r then α r * U i x else 0) = 0.
    have hsum : (∑ x : Fin R, if x = r then α r * U i x else 0) =
        α r * U i r := by
      rw [Finset.sum_eq_single r]
      · simp
      · intro t _ ht
        simp [ht]
      · intro hns; exact absurd (Finset.mem_univ r) hns
    rw [hsum]
    exact congrFun hzero i

/-! ### Simultaneous-diagonalization uniqueness theorem (statement only) -/

/-- **Simultaneous-diagonalization uniqueness theorem (Theorem
    `thm:simultaneous-diagonalization`, Leurgans-Ross-Abel 1993).**
    Under linear independence of `cp`'s mode-0 and mode-1 columns, and
    pairwise non-collinearity of `cp`'s mode-2 columns, every `R`-term
    CP decomposition of the same tensor is factor-equivalent (in the
    sense of `FactorEquiv₃`).

    **Proof structure.** Following the lecture
    (``uniqueness is immediate from Theorem [thm:kruskal]''), the
    proof reduces to **Kruskal's theorem** (`Tensor.kruskal_uniqueness`,
    axiom A9). The bridging lemmas of `Matrix.kRank` show that the
    independence hypotheses imply
    `kᵤ + kᵥ + kᵥᵥ ≥ R + R + 2 = 2R + (3 − 1)`, which is Kruskal's
    sufficient condition.

    The mode-2 hypothesis `hW` says: any two distinct mode-2 columns
    `cp.factors r 2` and `cp.factors t 2` are not scalar multiples of
    each other (in either direction, since `r ≠ t` is symmetric). The
    additional hypothesis `hr : 2 ≤ R` excludes the degenerate cases
    `R = 0` (zero tensor) and `R = 1` (rank-1 tensor) for which the
    pairwise-LI mode-2 input is vacuous and `FactorEquiv₃` is trivial
    by inspection; the main content is `R ≥ 2`. -/
theorem simDiag_uniqueness
    {I J K R : ℕ}
    (cp : CPDecomp 𝕜 3 ![I, J, K] R)
    (hr : 2 ≤ R)
    (hU : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => cp.factors r 0 i))
    (hV : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => cp.factors r 1 j))
    (hW : ∀ r t : Fin R, r ≠ t →
      ¬ ∃ c : 𝕜, (fun k : Fin K => cp.factors r 2 k) =
        c • (fun k : Fin K => cp.factors t 2 k))
    (cp' : CPDecomp 𝕜 3 ![I, J, K] R)
    (h : cp'.tensor = cp.tensor) :
    FactorEquiv₃ cp cp' := by
  -- k-rank of the mode-0 factor matrix equals R (full LI).
  have hkU : (cp.factorMatrix 0).kRank = R := by
    apply Matrix.kRank_eq_of_lin_indep
    convert hU using 1
  -- k-rank of the mode-1 factor matrix equals R.
  have hkV : (cp.factorMatrix 1).kRank = R := by
    apply Matrix.kRank_eq_of_lin_indep
    convert hV using 1
  -- k-rank of the mode-2 factor matrix is ≥ 2 (pairwise LI).
  have hkW : 2 ≤ (cp.factorMatrix 2).kRank := by
    apply Matrix.kRank_pairwise_lin_indep_ge_two _ hr
    intro r t hst
    convert hW r t hst using 0
  -- Sum of k-ranks ≥ 2R + (3 − 1) = 2R + 2.
  have hsum : (∑ n : Fin 3, (cp.factorMatrix n).kRank) ≥ 2 * R + (3 - 1) := by
    rw [Fin.sum_univ_three, hkU, hkV]
    omega
  -- Apply Kruskal's theorem (axiom A9) to obtain `FactorEquivD cp cp'`.
  have hker := kruskal_uniqueness cp cp' h.symm hsum
  -- Convert `FactorEquivD` to `FactorEquiv₃` via the order-3 bridge.
  exact hker.toFactorEquiv₃

/-! ### Phase 4.5: Constructive recovery via simultaneous diagonalization -/

/-! #### Helper lemmas for the constructive proof -/

/-- If `U` has linearly independent columns and `d : Fin R → 𝕜` is
    everywhere nonzero, then `U * Matrix.diagonal d` has linearly
    independent columns. -/
private lemma lin_indep_mul_diagonal {I R : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜)
    (d : Fin R → 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hd : ∀ r : Fin R, d r ≠ 0) :
    LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => (U * Matrix.diagonal d) i r) := by
  rw [Fintype.linearIndependent_iff]
  intro α hα
  -- (U * diagonal d) i r = U i r * d r.
  have hentry : ∀ i r, (U * Matrix.diagonal d) i r = U i r * d r := by
    intro i r; rw [Matrix.mul_apply]
    simp [Matrix.diagonal, Finset.sum_ite_eq']
  -- ∑ r, α r • (column r of U·diag d) = 0 ⇒ ∑ r, α r * U i r * d r = 0 ∀ i.
  have h_pt : ∀ i : Fin I, ∑ r : Fin R, (α r * d r) * U i r = 0 := by
    intro i
    have h := congrFun hα i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h
    rw [show (∑ r : Fin R, (α r * d r) * U i r)
            = ∑ r : Fin R, α r * (U * Matrix.diagonal d) i r from by
      refine Finset.sum_congr rfl fun r _ => ?_
      rw [hentry]; ring]
    exact h
  -- α r * d r = 0 for all r, by LI of U's columns.
  have h_alpha_d : ∀ r : Fin R, α r * d r = 0 := by
    apply (Fintype.linearIndependent_iff.mp hU) (fun r => α r * d r)
    funext i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    exact h_pt i
  -- d r ≠ 0 ⇒ α r = 0.
  intro r
  exact (mul_left_eq_zero_iff (hd r)).mp (h_alpha_d r)
where
  /-- A sublemma: `a * b = 0` and `b ≠ 0` imply `a = 0`. -/
  mul_left_eq_zero_iff {a b : 𝕜} (hb : b ≠ 0) : a * b = 0 ↔ a = 0 := by
    constructor
    · intro h; rcases mul_eq_zero.mp h with ha | hb'
      · exact ha
      · exact absurd hb' hb
    · intro h; rw [h, zero_mul]

/-- Pencil-quotient identity. For `U : I × R` with linearly independent
    columns, `V : J × R` with linearly independent columns, and
    `Db : Fin R → 𝕜` everywhere nonzero, we have

    `(U * diag(Da) * Vᵀ) * (U * diag(Db) * Vᵀ).pinv
       = U * diag(fun r => Da r / Db r) * U.pinv`.

    Steps: A4 (reverse-order pinv) twice to break up the right factor's
    pinv; A5 to compute `(diag Db).pinv = diag(1/Db)`; A3 to drop
    `Vᵀ * (Vᵀ).pinv = 1`; combine the two diagonals via
    `Matrix.diagonal_mul_diagonal`. -/
lemma pencil_quotient_eq
    {I J R : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜)
    (V : Matrix (Fin J) (Fin R) 𝕜)
    (Da Db : Fin R → 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hV : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V j r))
    (hDb : ∀ r : Fin R, Db r ≠ 0) :
    (U * Matrix.diagonal Da * Vᵀ) *
        (U * Matrix.diagonal Db * Vᵀ).pinv =
      U * Matrix.diagonal (fun r => Da r * (Db r)⁻¹) * U.pinv := by
  -- Set abbreviations and prove the requisite full-rank facts.
  set DDb := Matrix.diagonal Db with hDDb_def
  -- `U * diag Db` has LI columns: U does, `diag Db` is invertible.
  have h_UDb_li : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => (U * Matrix.diagonal Db) i r) :=
    lin_indep_mul_diagonal U Db hU hDb
  -- Vᵀ has LI rows: rows of Vᵀ = columns of V, which is hV.
  have h_VT_rows_li : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => (Vᵀ) r j) := by
    convert hV using 0
  -- (U * diag Db * Vᵀ).pinv = (Vᵀ).pinv * (U * diag Db).pinv via A4.
  have h_step1 :
      (U * Matrix.diagonal Db * Vᵀ).pinv =
        (Vᵀ).pinv * (U * Matrix.diagonal Db).pinv :=
    Matrix.pinv_mul_full_rank (U * Matrix.diagonal Db) Vᵀ
      h_UDb_li h_VT_rows_li
  -- (U * diag Db).pinv = (diag Db).pinv * U.pinv via A4.
  -- diag(Db) has LI rows: row r of diag(Db) is `j ↦ if r = j then Db r else 0`,
  -- and Db is everywhere nonzero.
  have h_diag_rows_li : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin R => (Matrix.diagonal Db) r j) := by
    rw [Fintype.linearIndependent_iff]
    intro α hα r
    have h := congrFun hα r
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h
    -- h : ∑ x, α x * diagonal Db x r = 0
    -- diagonal Db x r = if x = r then Db x else 0.
    have hred : (∑ x : Fin R, α x * (Matrix.diagonal Db) x r) = α r * Db r := by
      rw [Finset.sum_eq_single r]
      · simp [Matrix.diagonal]
      · intros b _ hb
        rw [show (Matrix.diagonal Db) b r = 0 from by simp [Matrix.diagonal, hb]]
        ring
      · intro hns; exact absurd (Finset.mem_univ r) hns
    rw [hred] at h
    rcases mul_eq_zero.mp h with hs | hb
    · exact hs
    · exact absurd hb (hDb r)
  have h_step2 : (U * Matrix.diagonal Db).pinv =
      (Matrix.diagonal Db).pinv * U.pinv :=
    Matrix.pinv_mul_full_rank U (Matrix.diagonal Db) hU h_diag_rows_li
  -- (diag Db).pinv = diag (1/Db) via A5.
  have h_step3 : (Matrix.diagonal Db).pinv =
      Matrix.diagonal (fun r => (Db r)⁻¹) :=
    Matrix.diagonal_pinv_of_nonzero Db hDb
  -- Vᵀ * (Vᵀ).pinv = 1 via A3.
  have h_step4 : Vᵀ * (Vᵀ).pinv = 1 :=
    Matrix.mul_pinv_self_of_full_row_lin_indep Vᵀ h_VT_rows_li
  -- diag(Da) * diag(1/Db) = diag(Da · 1/Db) via Matrix.diagonal_mul_diagonal.
  have h_diag_mul :
      Matrix.diagonal Da * Matrix.diagonal (fun r => (Db r)⁻¹) =
      Matrix.diagonal (fun r => Da r * (Db r)⁻¹) :=
    Matrix.diagonal_mul_diagonal _ _
  -- Now compose all the steps. Use associativity to stitch.
  rw [h_step1, h_step2, h_step3]
  -- Goal: (U * diag Da * Vᵀ) * ((Vᵀ).pinv * (diag (1/Db) * U.pinv)) =
  --       U * diag (Da · 1/Db) * U.pinv
  rw [show (U * Matrix.diagonal Da * Vᵀ) *
        ((Vᵀ).pinv * (Matrix.diagonal (fun r => (Db r)⁻¹) * U.pinv)) =
      U * Matrix.diagonal Da * (Vᵀ * (Vᵀ).pinv) *
        Matrix.diagonal (fun r => (Db r)⁻¹) * U.pinv from by
    simp [Matrix.mul_assoc]]
  rw [h_step4, Matrix.mul_one]
  rw [show U * Matrix.diagonal Da *
        Matrix.diagonal (fun r => (Db r)⁻¹) * U.pinv =
      U * (Matrix.diagonal Da * Matrix.diagonal (fun r => (Db r)⁻¹)) * U.pinv from by
    rw [Matrix.mul_assoc U _ _]]
  rw [h_diag_mul]

/-- **Rank-preservation auxiliary lemma (textbook fact).** If the matrix
    `U · diag(d) · Vᵀ = U' · diag(d') · (V')ᵀ` with `U`, `V`, `U'`, `V'` each
    having linearly independent columns and `d` everywhere nonzero, then `d'`
    is everywhere nonzero.

    This is a textbook consequence of the rank theorem: under the LI-column
    hypotheses on `U`, `V`, the LHS `U · diag(d) · Vᵀ` is a *full-rank
    factorization* and has rank exactly `r` (Bernstein, Fact 6.4.9 p. 379:
    "if `A = BC` with `B` and `C` of compatible full ranks, then
    `rank A = min(rank B, rank C)`"). The RHS has rank at most
    `#{r | d' r ≠ 0}`. Equality forces `d'` everywhere nonzero.

    The proof composes Mathlib's `Matrix.rank_diagonal`, `Matrix.rank_mul_le`,
    and the project's pseudoinverse axioms (A2, A3) — the identity
    `U.pinv · (U · diag d · Vᵀ) · (Vᵀ).pinv = diag d` provides the lower
    bound `rank(diag d) ≤ rank(U · diag d · Vᵀ)`. -/
private theorem rank_preservation_diag
    {I J R : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜)
    (V : Matrix (Fin J) (Fin R) 𝕜)
    (U' : Matrix (Fin I) (Fin R) 𝕜)
    (V' : Matrix (Fin J) (Fin R) 𝕜)
    (d d' : Fin R → 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hV : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V j r))
    (hV' : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V' j r))
    (hd : ∀ r : Fin R, d r ≠ 0)
    (heq : U * Matrix.diagonal d * Vᵀ = U' * Matrix.diagonal d' * (V')ᵀ) :
    ∀ r : Fin R, d' r ≠ 0 := by
  classical
  -- Rows of `Vᵀ` are LI iff columns of `V` are LI.
  have hVT_rows : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => (Vᵀ) r j) := by convert hV using 0
  have hV'T_rows : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => ((V')ᵀ) r j) := by convert hV' using 0
  -- Pseudoinverse left/right inverses (axioms A2, A3).
  have hUpinv : U.pinv * U = (1 : Matrix (Fin R) (Fin R) 𝕜) :=
    Matrix.pinv_mul_self_of_lin_indep U hU
  have hVTpinv : Vᵀ * (Vᵀ).pinv = (1 : Matrix (Fin R) (Fin R) 𝕜) :=
    Matrix.mul_pinv_self_of_full_row_lin_indep Vᵀ hVT_rows
  -- Recover `diag d` from `U · diag d · Vᵀ` via left-multiplying by `U.pinv`
  -- and right-multiplying by `(Vᵀ).pinv`.
  have h_recover :
      U.pinv * (U * Matrix.diagonal d * Vᵀ) * (Vᵀ).pinv =
        Matrix.diagonal d := by
    calc U.pinv * (U * Matrix.diagonal d * Vᵀ) * (Vᵀ).pinv
        = (U.pinv * U) * Matrix.diagonal d * (Vᵀ * (Vᵀ).pinv) := by
              simp [Matrix.mul_assoc]
      _ = (1 : Matrix (Fin R) (Fin R) 𝕜) * Matrix.diagonal d *
            (1 : Matrix (Fin R) (Fin R) 𝕜) := by rw [hUpinv, hVTpinv]
      _ = Matrix.diagonal d := by simp
  -- `card {r | d r ≠ 0} = R` since `d` is everywhere nonzero.
  have hd_card : Fintype.card {r : Fin R // d r ≠ 0} = R := by
    have h_equiv : {r : Fin R // d r ≠ 0} ≃ Fin R :=
      Equiv.subtypeUnivEquiv (fun r => hd r)
    simp [Fintype.card_congr h_equiv]
  -- Hence `rank(diag d) = R`.
  have h_rank_diag_d : (Matrix.diagonal d).rank = R := by
    rw [Matrix.rank_diagonal]; exact hd_card
  -- The LHS has rank ≤ rank(diag d) = R.
  have hLHS_le : (U * Matrix.diagonal d * Vᵀ).rank ≤ R := by
    calc (U * Matrix.diagonal d * Vᵀ).rank
        ≤ (U * Matrix.diagonal d).rank := Matrix.rank_mul_le_left _ _
      _ ≤ (Matrix.diagonal d).rank := Matrix.rank_mul_le_right _ _
      _ = R := h_rank_diag_d
  have hLHS_ge : R ≤ (U * Matrix.diagonal d * Vᵀ).rank := by
    have h1 : (Matrix.diagonal d).rank ≤
        (U.pinv * (U * Matrix.diagonal d * Vᵀ) * (Vᵀ).pinv).rank := by
      rw [h_recover]
    calc R = (Matrix.diagonal d).rank := h_rank_diag_d.symm
      _ ≤ (U.pinv * (U * Matrix.diagonal d * Vᵀ) * (Vᵀ).pinv).rank := h1
      _ ≤ (U.pinv * (U * Matrix.diagonal d * Vᵀ)).rank :=
            Matrix.rank_mul_le_left _ _
      _ ≤ (U * Matrix.diagonal d * Vᵀ).rank := Matrix.rank_mul_le_right _ _
  have hLHS_eq_r : (U * Matrix.diagonal d * Vᵀ).rank = R :=
    le_antisymm hLHS_le hLHS_ge
  -- The RHS has rank ≤ `#{r | d' r ≠ 0}`.
  have hRHS_le_card :
      (U' * Matrix.diagonal d' * (V')ᵀ).rank ≤
        Fintype.card {r : Fin R // d' r ≠ 0} := by
    have h1 : (U' * Matrix.diagonal d' * (V')ᵀ).rank ≤
        (U' * Matrix.diagonal d').rank :=
      Matrix.rank_mul_le_left _ _
    have h2 : (U' * Matrix.diagonal d').rank ≤ (Matrix.diagonal d').rank :=
      Matrix.rank_mul_le_right _ _
    have h3 : (Matrix.diagonal d').rank =
        Fintype.card {r : Fin R // d' r ≠ 0} := Matrix.rank_diagonal d'
    linarith
  -- From `heq`, the two ranks are equal; combine bounds to conclude.
  have h_card_eq : Fintype.card {r : Fin R // d' r ≠ 0} = R := by
    have hRHS_eq : (U' * Matrix.diagonal d' * (V')ᵀ).rank = R := by
      rw [← heq]; exact hLHS_eq_r
    have hcard_le_r :
        Fintype.card {r : Fin R // d' r ≠ 0} ≤ Fintype.card (Fin R) :=
      Fintype.card_subtype_le _
    have : R ≤ Fintype.card {r : Fin R // d' r ≠ 0} := by
      have := hRHS_eq.symm.le.trans hRHS_le_card
      exact this
    have hr : Fintype.card (Fin R) = R := Fintype.card_fin R
    omega
  -- A subtype of `Fin R` with cardinality `R` is all of `Fin R`.
  intro r hr_d
  have h_card_zero : Fintype.card {r : Fin R // d' r = 0} = 0 := by
    have h_split :
        Fintype.card {r : Fin R // d' r ≠ 0} +
            Fintype.card {r : Fin R // d' r = 0} = R := by
      classical
      have := Fintype.card_subtype_compl (α := Fin R) (p := fun r => d' r ≠ 0)
      -- `card {r // ¬ (d' r ≠ 0)} = card (Fin R) - card {r // d' r ≠ 0}`
      have h_eq : Fintype.card {r : Fin R // ¬ (d' r ≠ 0)} =
          Fintype.card {r : Fin R // d' r = 0} := by
        apply Fintype.card_congr
        exact Equiv.subtypeEquivProp (by ext; simp)
      have hr : Fintype.card (Fin R) = R := Fintype.card_fin R
      omega
    omega
  have hne : Nonempty {r : Fin R // d' r = 0} := ⟨⟨r, hr_d⟩⟩
  have : Fintype.card {r : Fin R // d' r = 0} ≠ 0 := Fintype.card_ne_zero
  exact this h_card_zero

/-- **Eigenvector recovery from a matrix-pencil pair (Lemma 8, part 2,
    in the lecture's notation).** Suppose two matrix factorizations
    share a pencil pair: there exist diagonal-scaling vectors
    `Da, Db, D'a, D'b : Fin R → 𝕜` with `Db, D'b` everywhere nonzero,
    and matrices `U, U' : Fin I × Fin R`, `V, V' : Fin J × Fin R`
    with linearly independent columns, satisfying
    `U · diag(Da) · Vᵀ = U' · diag(D'a) · (V')ᵀ` and
    `U · diag(Db) · Vᵀ = U' · diag(D'b) · (V')ᵀ`. Suppose furthermore
    that the diagonal entries `Da r · (Db r)⁻¹` of the quotient
    `diag(Da)·diag(Db)⁻¹` are pairwise distinct. Then the columns of
    `U` and `U'` agree up to a permutation `π : Equiv.Perm (Fin R)`
    and per-column nonzero scaling `c : Fin R → 𝕜`, with the
    pencil-quotient eigenvalues matching under `π`.

    The proof is the natural composition of the pencil-quotient
    identity (`pencil_quotient_eq`, applied to both sides) with the
    eigendecomposition-uniqueness axiom A7
    (`Matrix.eigendecomp_unique_of_distinct`). This is the abstract
    eigenvector-recovery consequence used inside
    `simDiag_uniqueness_constructive`. -/
theorem pencil_eigendecomp_unique
    {I J R : ℕ}
    (U U' : Matrix (Fin I) (Fin R) 𝕜)
    (V V' : Matrix (Fin J) (Fin R) 𝕜)
    (Da Db D'a D'b : Fin R → 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hV : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V j r))
    (hU' : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U' i r))
    (hV' : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V' j r))
    (hDb : ∀ r : Fin R, Db r ≠ 0)
    (hD'b : ∀ r : Fin R, D'b r ≠ 0)
    (hΛ_inj : Function.Injective (fun r : Fin R => Da r * (Db r)⁻¹))
    (h_pencil_a :
      (U * Matrix.diagonal Da * Vᵀ : Matrix (Fin I) (Fin J) 𝕜) =
        U' * Matrix.diagonal D'a * (V')ᵀ)
    (h_pencil_b :
      (U * Matrix.diagonal Db * Vᵀ : Matrix (Fin I) (Fin J) 𝕜) =
        U' * Matrix.diagonal D'b * (V')ᵀ) :
    ∃ π : Equiv.Perm (Fin R),
      (∀ r, Da r * (Db r)⁻¹ = D'a (π r) * (D'b (π r))⁻¹) ∧
      (∀ r, ∃ c : 𝕜, c ≠ 0 ∧ ∀ i, U' i (π r) = c * U i r) := by
  -- Pencil quotient on the cp side: A · B.pinv = U · diag(Da/Db) · U.pinv.
  have h_quot_cp :
      (U * Matrix.diagonal Da * Vᵀ) *
          (U * Matrix.diagonal Db * Vᵀ).pinv =
        U * Matrix.diagonal (fun r => Da r * (Db r)⁻¹) * U.pinv :=
    pencil_quotient_eq U V Da Db hU hV hDb
  -- Pencil quotient on the cp' side: A' · (B').pinv = U' · diag(D'a/D'b) · U'.pinv.
  have h_quot_cp' :
      (U' * Matrix.diagonal D'a * (V')ᵀ) *
          (U' * Matrix.diagonal D'b * (V')ᵀ).pinv =
        U' * Matrix.diagonal (fun r => D'a r * (D'b r)⁻¹) * U'.pinv :=
    pencil_quotient_eq U' V' D'a D'b hU' hV' hD'b
  -- The two pencil quotients are equal because the underlying pencils are.
  have h_quot_eq :
      U * Matrix.diagonal (fun r => Da r * (Db r)⁻¹) * U.pinv =
        U' * Matrix.diagonal (fun r => D'a r * (D'b r)⁻¹) * U'.pinv := by
    rw [← h_quot_cp, ← h_quot_cp', h_pencil_a, h_pencil_b]
  -- Apply axiom A7 (eigendecomposition uniqueness with distinct eigenvalues).
  exact Matrix.eigendecomp_unique_of_distinct U U'
    (fun r => Da r * (Db r)⁻¹)
    (fun r => D'a r * (D'b r)⁻¹)
    hU hU' hΛ_inj h_quot_eq

/-- **Lemma 8 (full statement, single decomposition form).** Bundles the
    pencil-quotient identities with the eigenvector-recovery consequences.

    Let `A := U · diag(Da) · Vᵀ` and `B := U · diag(Db) · Vᵀ` with `U`,
    `V` having linearly independent columns and `Db` everywhere nonzero.
    Write `Λ r := Da r · (Db r)⁻¹` for the diagonal of the pencil
    quotient. Then this single theorem packages:

    * (a1) `A · B† = U · diag(Λ) · U†`;
    * (a2) provided `Da` is also everywhere nonzero,
      `(A† · B)ᵀ = V · diag(Λ⁻¹) · V†` (where `Λ⁻¹ r := (Da r)⁻¹ · Db r`);
    * (b1) the columns of `U` are eigenvectors of `A · B†` with
      eigenvalues `Λ r`;
    * (b1V) symmetrically (under the same extra hypothesis as (a2)),
      the columns of `V` are eigenvectors of `(A† · B)ᵀ` with reciprocal
      eigenvalues `(Da r)⁻¹ · Db r = Λ r⁻¹`;
    * (b2) provided `Λ` is injective and `Da` is nonzero, the eigenvectors
      of `A · B†` with eigenvalue `Λ i` are exactly the nonzero scalar
      multiples of column `i` of `U` — i.e., any nonzero `x` with
      `(A · B†) ·ᵥ x = Λ i • x` is of the form `c · (column i of U)`
      with `c ≠ 0`;
    * (b3) symmetrically for `V` against `(A† · B)ᵀ` (same hypotheses);
    * (b4) the reciprocal-eigenvalue pairing matches column `r` of `U`
      with column `r` of `V` (built into the indexing of (b1)/(b1V)).

    The seven conjuncts are returned as nested `And`s in the order
    listed; named accessors via destructuring are recommended at use
    sites. The (b)-block (b2)/(b3) is gated on the additional hypotheses
    `Function.Injective Λ` and `∀ r, Da r ≠ 0`; (a2)/(b1V) is gated on
    `∀ r, Da r ≠ 0`.

    Reference: lecture4_CPDecomposition.org, Lemma 8 (lines 440–451). -/
theorem pencil_decomp_lemma8
    {I J R : ℕ}
    (U : Matrix (Fin I) (Fin R) 𝕜)
    (V : Matrix (Fin J) (Fin R) 𝕜)
    (Da Db : Fin R → 𝕜)
    (hU : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin I => U i r))
    (hV : LinearIndependent 𝕜 (fun r : Fin R => fun j : Fin J => V j r))
    (hDb : ∀ r : Fin R, Db r ≠ 0) :
    -- (a1) Forward pencil-quotient identity.
    ((U * Matrix.diagonal Da * Vᵀ) * (U * Matrix.diagonal Db * Vᵀ).pinv =
        U * Matrix.diagonal (fun r => Da r * (Db r)⁻¹) * U.pinv) ∧
    -- (a2) Symmetric pencil-quotient identity (conditional on Da ≠ 0).
    ((∀ r, Da r ≠ 0) →
      ((U * Matrix.diagonal Da * Vᵀ).pinv * (U * Matrix.diagonal Db * Vᵀ))ᵀ =
        V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r) * V.pinv) ∧
    -- (b1) Each column of U is an eigenvector of A · B†.
    (∀ i : Fin R,
      ((U * Matrix.diagonal Da * Vᵀ) * (U * Matrix.diagonal Db * Vᵀ).pinv)
          *ᵥ (fun row => U row i) =
        (Da i * (Db i)⁻¹) • (fun row => U row i)) ∧
    -- (b1V) Each column of V is an eigenvector of (A† · B)ᵀ (conditional on Da ≠ 0).
    ((∀ r, Da r ≠ 0) → ∀ i : Fin R,
      (((U * Matrix.diagonal Da * Vᵀ).pinv *
            (U * Matrix.diagonal Db * Vᵀ))ᵀ) *ᵥ (fun row => V row i) =
        ((Da i)⁻¹ * Db i) • (fun row => V row i)) ∧
    -- (b2) Absolute uniqueness of eigenvectors of A · B† (conditional on
    -- distinct eigenvalues and Da ≠ 0).
    (Function.Injective (fun r : Fin R => Da r * (Db r)⁻¹) → (∀ r, Da r ≠ 0) →
      ∀ (i : Fin R) (x : Fin I → 𝕜),
        x ≠ 0 →
        ((U * Matrix.diagonal Da * Vᵀ) * (U * Matrix.diagonal Db * Vᵀ).pinv)
            *ᵥ x = (Da i * (Db i)⁻¹) • x →
        ∃ c : 𝕜, c ≠ 0 ∧ x = c • (fun row => U row i)) ∧
    -- (b3) Absolute uniqueness of eigenvectors of (A† · B)ᵀ (conditional on
    -- distinct eigenvalues and Da ≠ 0).
    (Function.Injective (fun r : Fin R => Da r * (Db r)⁻¹) → (∀ r, Da r ≠ 0) →
      ∀ (i : Fin R) (y : Fin J → 𝕜),
        y ≠ 0 →
        (((U * Matrix.diagonal Da * Vᵀ).pinv *
              (U * Matrix.diagonal Db * Vᵀ))ᵀ) *ᵥ y =
            ((Da i)⁻¹ * Db i) • y →
        ∃ c : 𝕜, c ≠ 0 ∧ y = c • (fun row => V row i)) := by
  classical
  -- Abbreviations.
  set A : Matrix (Fin I) (Fin J) 𝕜 := U * Matrix.diagonal Da * Vᵀ with hA_def
  set B : Matrix (Fin I) (Fin J) 𝕜 := U * Matrix.diagonal Db * Vᵀ with hB_def
  set Λ : Fin R → 𝕜 := fun r => Da r * (Db r)⁻¹ with hΛ_def
  -- Derived linear-independence facts.
  have hU_pinv_self : U.pinv * U = (1 : Matrix (Fin R) (Fin R) 𝕜) :=
    Matrix.pinv_mul_self_of_lin_indep U hU
  have hV_pinv_self : V.pinv * V = (1 : Matrix (Fin R) (Fin R) 𝕜) :=
    Matrix.pinv_mul_self_of_lin_indep V hV
  -- (a1): direct pencil-quotient identity.
  have h_a1 : A * B.pinv = U * Matrix.diagonal Λ * U.pinv := by
    simpa [hA_def, hB_def, hΛ_def] using
      pencil_quotient_eq U V Da Db hU hV hDb
  -- Helper: V · diag(Db) · Uᵀ = Bᵀ.
  -- (a2): symmetric pencil-quotient via transpose. We need Da ≠ 0.
  have h_a2 : (∀ r, Da r ≠ 0) →
      (A.pinv * B)ᵀ = V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r) * V.pinv := by
    intro hDa
    -- (A.pinv * B)ᵀ = Bᵀ * (A.pinv)ᵀ = Bᵀ * (Aᵀ).pinv.
    have hT1 : (A.pinv * B)ᵀ = Bᵀ * (Aᵀ).pinv := by
      rw [Matrix.transpose_mul, Matrix.pinv_transpose]
    -- Aᵀ = V · diag Da · Uᵀ.
    have hAT : Aᵀ = V * Matrix.diagonal Da * Uᵀ := by
      have h1 : (U * Matrix.diagonal Da * Vᵀ)ᵀ =
          (Vᵀ)ᵀ * (U * Matrix.diagonal Da)ᵀ := Matrix.transpose_mul _ _
      have h2 : (U * Matrix.diagonal Da)ᵀ = (Matrix.diagonal Da)ᵀ * Uᵀ :=
        Matrix.transpose_mul _ _
      have h3 : (Matrix.diagonal Da : Matrix (Fin R) (Fin R) 𝕜)ᵀ =
          Matrix.diagonal Da := Matrix.diagonal_transpose _
      calc Aᵀ = (U * Matrix.diagonal Da * Vᵀ)ᵀ := by rw [hA_def]
        _ = (Vᵀ)ᵀ * (U * Matrix.diagonal Da)ᵀ := h1
        _ = V * (U * Matrix.diagonal Da)ᵀ := by rw [Matrix.transpose_transpose]
        _ = V * ((Matrix.diagonal Da)ᵀ * Uᵀ) := by rw [h2]
        _ = V * (Matrix.diagonal Da * Uᵀ) := by rw [h3]
        _ = V * Matrix.diagonal Da * Uᵀ := by rw [Matrix.mul_assoc]
    -- Bᵀ = V · diag Db · Uᵀ.
    have hBT : Bᵀ = V * Matrix.diagonal Db * Uᵀ := by
      have h1 : (U * Matrix.diagonal Db * Vᵀ)ᵀ =
          (Vᵀ)ᵀ * (U * Matrix.diagonal Db)ᵀ := Matrix.transpose_mul _ _
      have h2 : (U * Matrix.diagonal Db)ᵀ = (Matrix.diagonal Db)ᵀ * Uᵀ :=
        Matrix.transpose_mul _ _
      have h3 : (Matrix.diagonal Db : Matrix (Fin R) (Fin R) 𝕜)ᵀ =
          Matrix.diagonal Db := Matrix.diagonal_transpose _
      calc Bᵀ = (U * Matrix.diagonal Db * Vᵀ)ᵀ := by rw [hB_def]
        _ = (Vᵀ)ᵀ * (U * Matrix.diagonal Db)ᵀ := h1
        _ = V * (U * Matrix.diagonal Db)ᵀ := by rw [Matrix.transpose_transpose]
        _ = V * ((Matrix.diagonal Db)ᵀ * Uᵀ) := by rw [h2]
        _ = V * (Matrix.diagonal Db * Uᵀ) := by rw [h3]
        _ = V * Matrix.diagonal Db * Uᵀ := by rw [Matrix.mul_assoc]
    -- Now apply pencil_quotient_eq with U/V swapped, Da/Db swapped.
    have h_swap :
        (V * Matrix.diagonal Db * Uᵀ) * (V * Matrix.diagonal Da * Uᵀ).pinv =
          V * Matrix.diagonal (fun r => Db r * (Da r)⁻¹) * V.pinv :=
      pencil_quotient_eq V U Db Da hV hU hDa
    -- Rewrite (A.pinv * B)ᵀ using hT1, hAT, hBT.
    rw [hT1, hBT, hAT, h_swap]
    -- Final step: rewrite `Db r * (Da r)⁻¹ = (Da r)⁻¹ * Db r` (commutativity).
    have hdiag : Matrix.diagonal (fun r : Fin R => Db r * (Da r)⁻¹) =
        Matrix.diagonal (fun r : Fin R => (Da r)⁻¹ * Db r) := by
      congr 1; funext r; ring
    rw [hdiag]
  -- Helper: column i of W as W *ᵥ Pi.single i 1.
  have h_col_eq : ∀ {N : ℕ} (W : Matrix (Fin N) (Fin R) 𝕜) (i : Fin R),
      (fun row : Fin N => W row i) = W *ᵥ (Pi.single i (1 : 𝕜)) := by
    intros N W i
    funext row
    simp [Matrix.mulVec, dotProduct, Pi.single, Function.update]
  -- Helper: M *ᵥ (column i of W) when M · W has known structure.
  -- (b1): U columns are eigenvectors of A · B† with eigenvalue Λ.
  have h_b1 : ∀ i : Fin R,
      (A * B.pinv) *ᵥ (fun row => U row i) = Λ i • (fun row => U row i) := by
    intro i
    rw [h_a1, h_col_eq U i]
    -- Goal: (U · diag Λ · U.pinv) *ᵥ U *ᵥ Pi.single i 1 = Λ i • U *ᵥ Pi.single i 1.
    -- Combine all the matrix-vector products into a single product.
    rw [Matrix.mulVec_mulVec]
    -- Goal: (U·diag Λ·U.pinv * U) *ᵥ Pi.single i 1 = Λ i • U *ᵥ Pi.single i 1.
    rw [show U * Matrix.diagonal Λ * U.pinv * U =
          U * Matrix.diagonal Λ * (U.pinv * U) from Matrix.mul_assoc _ _ _]
    rw [hU_pinv_self, Matrix.mul_one]
    -- Goal: (U · diag Λ) *ᵥ Pi.single i 1 = Λ i • U *ᵥ Pi.single i 1.
    funext row
    simp only [Pi.smul_apply, smul_eq_mul]
    rw [Matrix.mulVec, Matrix.mulVec]
    change dotProduct _ _ = Λ i * dotProduct _ _
    -- Both sides reduce to one entry via Pi.single.
    have hL : dotProduct (fun r => (U * Matrix.diagonal Λ) row r)
        (Pi.single i (1 : 𝕜)) = (U * Matrix.diagonal Λ) row i := by
      simp [dotProduct, Pi.single, Function.update]
    have hR : dotProduct (fun r => U row r) (Pi.single i (1 : 𝕜)) = U row i := by
      simp [dotProduct, Pi.single, Function.update]
    rw [hL, hR]
    rw [Matrix.mul_apply]
    rw [Finset.sum_eq_single i (fun r _ hri => by
        simp [Matrix.diagonal, hri]) (fun h => absurd (Finset.mem_univ i) h)]
    simp [Matrix.diagonal]
    ring
  -- (b1V): V columns are eigenvectors of (A.pinv * B)ᵀ with eigenvalue Λ⁻¹ (conditional).
  have h_b1V : (∀ r, Da r ≠ 0) → ∀ i : Fin R,
      ((A.pinv * B)ᵀ) *ᵥ (fun row => V row i) =
        ((Da i)⁻¹ * Db i) • (fun row => V row i) := by
    intro hDa i
    rw [h_a2 hDa, h_col_eq V i]
    -- Goal: (V · diag(Da⁻¹·Db) · V.pinv) *ᵥ V *ᵥ Pi.single i 1 =
    --       ((Da i)⁻¹ * Db i) • V *ᵥ Pi.single i 1.
    rw [Matrix.mulVec_mulVec]
    rw [show V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r) * V.pinv * V =
          V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r) * (V.pinv * V) from
      Matrix.mul_assoc _ _ _]
    rw [hV_pinv_self, Matrix.mul_one]
    funext row
    simp only [Pi.smul_apply, smul_eq_mul]
    rw [Matrix.mulVec, Matrix.mulVec]
    change dotProduct _ _ = ((Da i)⁻¹ * Db i) * dotProduct _ _
    have hL : dotProduct
        (fun r => (V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r)) row r)
        (Pi.single i (1 : 𝕜)) =
        (V * Matrix.diagonal (fun r => (Da r)⁻¹ * Db r)) row i := by
      simp [dotProduct, Pi.single, Function.update]
    have hR : dotProduct (fun r => V row r) (Pi.single i (1 : 𝕜)) = V row i := by
      simp [dotProduct, Pi.single, Function.update]
    rw [hL, hR]
    rw [Matrix.mul_apply]
    rw [Finset.sum_eq_single i (fun r _ hri => by
        simp [Matrix.diagonal, hri]) (fun h => absurd (Finset.mem_univ i) h)]
    simp [Matrix.diagonal]
    ring
  -- Helper: absolute eigenvector uniqueness for `M = U · diag Λ · U.pinv`,
  -- with Λ injective and everywhere nonzero, against the column structure of U.
  -- This is the proof of (b2); (b3) reuses it with V swapped in.
  have h_abs_uniq :
      ∀ {N : ℕ} (W : Matrix (Fin N) (Fin R) 𝕜) (μ : Fin R → 𝕜)
        (_hW : LinearIndependent 𝕜 (fun r : Fin R => fun i : Fin N => W i r))
        (_hμ_inj : Function.Injective μ) (_hμ_ne : ∀ r, μ r ≠ 0)
        (i : Fin R) (x : Fin N → 𝕜),
        x ≠ 0 →
        (W * Matrix.diagonal μ * W.pinv) *ᵥ x = μ i • x →
        ∃ c : 𝕜, c ≠ 0 ∧ x = c • (fun row => W row i) := by
    intros N W μ hW hμ_inj hμ_ne i x hx_ne hx_eig
    have hW_pinv_self : W.pinv * W = (1 : Matrix (Fin R) (Fin R) 𝕜) :=
      Matrix.pinv_mul_self_of_lin_indep W hW
    -- Let y := W.pinv *ᵥ x.
    set y : Fin R → 𝕜 := W.pinv *ᵥ x with hy_def
    -- Step 1: W *ᵥ (diag μ *ᵥ y) = μ i • x.
    have hstep1 : W *ᵥ (Matrix.diagonal μ *ᵥ y) = μ i • x := by
      have : (W * Matrix.diagonal μ * W.pinv) *ᵥ x =
          W *ᵥ (Matrix.diagonal μ *ᵥ (W.pinv *ᵥ x)) := by
        rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec]
      rw [← this, hx_eig]
    -- Step 2: apply W.pinv to both sides:
    --   W.pinv *ᵥ (W *ᵥ (diag μ *ᵥ y)) = diag μ *ᵥ y (by pinv·W = 1)
    --   W.pinv *ᵥ (μ i • x) = μ i • y.
    have hstep2 : Matrix.diagonal μ *ᵥ y = μ i • y := by
      have hL : W.pinv *ᵥ (W *ᵥ (Matrix.diagonal μ *ᵥ y)) =
          Matrix.diagonal μ *ᵥ y := by
        rw [Matrix.mulVec_mulVec, hW_pinv_self, Matrix.one_mulVec]
      have hR : W.pinv *ᵥ (μ i • x) = μ i • y := by
        rw [Matrix.mulVec_smul]
      rw [← hL, hstep1, hR]
    -- Step 3: pointwise, μ r · y_r = μ i · y_r, hence y_r = 0 for r ≠ i.
    have hys : ∀ r : Fin R, r ≠ i → y r = 0 := by
      intro r hri
      have h := congrFun hstep2 r
      simp only [Pi.smul_apply, smul_eq_mul] at h
      rw [Matrix.mulVec_diagonal] at h
      -- h : μ r * y r = μ i * y r.
      have hμri : μ r ≠ μ i := fun heq => hri (hμ_inj heq)
      -- (μ r − μ i) * y r = 0, and μ r − μ i ≠ 0, so y r = 0.
      have h' : (μ r - μ i) * y r = 0 := by
        rw [sub_mul]; linear_combination h
      rcases mul_eq_zero.mp h' with h1 | h2
      · exact absurd (sub_eq_zero.mp h1) hμri
      · exact h2
    -- Step 4: diag μ *ᵥ y is supported only at i, and equals μ i · y i at i.
    -- From hstep1: W *ᵥ (diag μ *ᵥ y) = μ i • x.
    -- diag μ *ᵥ y = (μ i · y i) · e_i.
    have h_diag_y : Matrix.diagonal μ *ᵥ y =
        (μ i * y i) • (Pi.single i (1 : 𝕜) : Fin R → 𝕜) := by
      funext r
      rw [Matrix.mulVec_diagonal]
      simp only [Pi.smul_apply, smul_eq_mul]
      by_cases hri : r = i
      · subst hri; simp
      · rw [hys r hri, mul_zero, Pi.single_eq_of_ne hri, mul_zero]
    -- Step 5: substitute in hstep1.
    have hstep5 : W *ᵥ ((μ i * y i) • (Pi.single i (1 : 𝕜) : Fin R → 𝕜)) =
        μ i • x := by
      rw [← h_diag_y]; exact hstep1
    -- LHS = (μ i * y i) • (W *ᵥ e_i) = (μ i * y i) • column i of W.
    rw [Matrix.mulVec_smul] at hstep5
    -- Compute W *ᵥ e_i = column i of W.
    have hWei : W *ᵥ (Pi.single i (1 : 𝕜) : Fin R → 𝕜) = (fun row => W row i) := by
      funext row
      simp [Matrix.mulVec, dotProduct, Pi.single, Function.update]
    rw [hWei] at hstep5
    -- hstep5 : (μ i * y i) • (column i of W) = μ i • x.
    -- Rearrange: x = y i • (column i of W) (using μ i ≠ 0).
    -- (μ i * y i) • _ = μ i • x ⟹ μ i • (y i • _) = μ i • x ⟹ y i • _ = x.
    have hμi_ne : μ i ≠ 0 := hμ_ne i
    have hxeq : x = y i • (fun row => W row i) := by
      have hsmul : (μ i * y i) • (fun row : Fin N => W row i) =
          μ i • (y i • (fun row : Fin N => W row i)) := by
        rw [smul_smul]
      rw [hsmul] at hstep5
      -- hstep5 : μ i • (y i • _) = μ i • x.
      -- Cancel μ i.
      exact (smul_right_injective (Fin N → 𝕜) hμi_ne hstep5).symm
    -- y i ≠ 0: else x = 0, contradicting hx_ne.
    have hyi_ne : y i ≠ 0 := by
      intro hyi
      apply hx_ne
      rw [hxeq, hyi, zero_smul]
    exact ⟨y i, hyi_ne, hxeq⟩
  -- Λ everywhere nonzero ⟺ Da everywhere nonzero (since Db ≠ 0).
  have hΛ_ne_iff : (∀ r, Da r ≠ 0) → ∀ r, Λ r ≠ 0 := by
    intro hDa r heq
    have hΛr : Da r * (Db r)⁻¹ = 0 := heq
    rcases mul_eq_zero.mp hΛr with h1 | h2
    · exact hDa r h1
    · exact hDb r (inv_eq_zero.mp h2)
  -- (b2): apply h_abs_uniq with W := U, μ := Λ.
  have h_b2 :
      Function.Injective Λ → (∀ r, Da r ≠ 0) →
      ∀ (i : Fin R) (x : Fin I → 𝕜),
        x ≠ 0 →
        (A * B.pinv) *ᵥ x = Λ i • x →
        ∃ c : 𝕜, c ≠ 0 ∧ x = c • (fun row => U row i) := by
    intro hΛ_inj hDa i x hx_ne hx_eig
    -- Translate `(A * B.pinv) *ᵥ x = Λ i • x` to `(U·diag Λ·U.pinv) *ᵥ x = Λ i • x`.
    have hx_eig' : (U * Matrix.diagonal Λ * U.pinv) *ᵥ x = Λ i • x := by
      rw [← h_a1]; exact hx_eig
    exact h_abs_uniq U Λ hU hΛ_inj (hΛ_ne_iff hDa) i x hx_ne hx_eig'
  -- (b3): apply h_abs_uniq with W := V, μ := Λ⁻¹ pointwise = (Da⁻¹ * Db).
  have h_b3 :
      Function.Injective Λ → (∀ r, Da r ≠ 0) →
      ∀ (i : Fin R) (y : Fin J → 𝕜),
        y ≠ 0 →
        ((A.pinv * B)ᵀ) *ᵥ y = ((Da i)⁻¹ * Db i) • y →
        ∃ c : 𝕜, c ≠ 0 ∧ y = c • (fun row => V row i) := by
    intro hΛ_inj hDa i y hy_ne hy_eig
    set μ : Fin R → 𝕜 := fun r => (Da r)⁻¹ * Db r with hμ_def
    -- μ injective: μ r = μ t means (Da r)⁻¹ * Db r = (Da t)⁻¹ * Db t.
    -- Multiply by Λ r · Λ t to relate. Cleanest: μ r = (Λ r)⁻¹ pointwise.
    have hμ_eq : ∀ r, μ r = (Λ r)⁻¹ := by
      intro r
      simp [hμ_def, hΛ_def, mul_comm]
    have hμ_inj : Function.Injective μ := by
      intro r t hrt
      rw [hμ_eq r, hμ_eq t] at hrt
      have hΛr_ne : Λ r ≠ 0 := hΛ_ne_iff hDa r
      have hΛt_ne : Λ t ≠ 0 := hΛ_ne_iff hDa t
      -- (Λ r)⁻¹ = (Λ t)⁻¹ ⟹ Λ r = Λ t.
      have : Λ r = Λ t := by
        rw [← inv_inv (Λ r), ← inv_inv (Λ t), hrt]
      exact hΛ_inj this
    have hμ_ne : ∀ r, μ r ≠ 0 := by
      intro r
      rw [hμ_eq r]
      exact inv_ne_zero (hΛ_ne_iff hDa r)
    -- Translate `((A.pinv * B)ᵀ) *ᵥ y = μ i • y` to `(V · diag μ · V.pinv) *ᵥ y = μ i • y`.
    have hy_eig' : (V * Matrix.diagonal μ * V.pinv) *ᵥ y = μ i • y := by
      rw [← h_a2 hDa]; exact hy_eig
    exact h_abs_uniq V μ hV hμ_inj hμ_ne i y hy_ne hy_eig'
  -- Assemble all conjuncts.
  exact ⟨h_a1, h_a2, h_b1, h_b1V, h_b2, h_b3⟩

/-- **Constructive uniqueness via simultaneous diagonalization (Scope B
    claim 2 of Theorem `thm:simultaneous-diagonalization`).** A second
    proof of `simDiag_uniqueness`, faithfully formalizing the LRA /
    matrix-pencil construction.

    Composes axioms A1–A8 (Moore-Penrose pseudoinverse properties,
    eigendecomposition uniqueness with distinct eigenvalues, "almost
    surely" generic-contraction-vector existence) with the matrix-pencil
    structural lemma and Khatri-Rao FCR.

    This proof is "constructive" in the sense that it explicitly derives
    the matching permutation π and column scalings via the
    eigendecomposition of `A B†` and `(A† B)ᵀ`, mirroring the
    algorithm's mathematical content. The Kruskal proof
    (`simDiag_uniqueness`) is shorter but non-constructive (Kruskal's
    theorem itself is axiomatized). The two proofs cover the same
    statement (modulo the strengthened `cp'` non-degeneracy hypotheses
    `hU'`, `hV'`, which the LRA construction requires; see the note
    below).

    **Strengthening note.** This theorem requires `hU'` and `hV'` as
    hypotheses (the LRA construction needs `cp'` to be non-degenerate
    on modes 0 and 1 to apply the matrix-pencil derivation: the
    pseudoinverse manipulations rely on full-rank factorizations on
    both `cp` and `cp'`). The Kruskal version (`simDiag_uniqueness`)
    does NOT need these because Kruskal's theorem covers all `R`-term
    decompositions. The strengthened hypotheses are an honest reflection
    of what the LRA proof actually establishes.

    Reference: Leurgans, Ross, Abel, "A Decomposition for Three-Way
    Arrays", SIAM J. Matrix Anal. Appl. 14 (1993), 1064-1083. -/
theorem simDiag_uniqueness_constructive
    {I J K R : ℕ} (hr : 2 ≤ R)
    (cp : CPDecomp 𝕜 3 ![I, J, K] R)
    (hU : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => cp.factors r 0 i))
    (hV : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => cp.factors r 1 j))
    (hW : ∀ r t : Fin R, r ≠ t →
      ¬ ∃ c : 𝕜, (fun k : Fin K => cp.factors r 2 k) =
        c • (fun k : Fin K => cp.factors t 2 k))
    (cp' : CPDecomp 𝕜 3 ![I, J, K] R)
    (hU' : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => cp'.factors r 0 i))
    (hV' : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => cp'.factors r 1 j))
    (h : cp'.tensor = cp.tensor) :
    FactorEquiv₃ cp cp' := by
  classical
  -- Abbreviations.
  set U := cp.factorMatrix 0 with hU_def
  set V := cp.factorMatrix 1 with hV_def
  set W := cp.factorMatrix 2 with hW_def
  set U' := cp'.factorMatrix 0 with hU'_def
  set V' := cp'.factorMatrix 1 with hV'_def
  set W' := cp'.factorMatrix 2 with hW'_def
  -- Re-express the hypotheses on U, V, U', V' in terms of factorMatrix.
  have hU_li : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => U i r) := by
    convert hU using 0
  have hV_li : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => V j r) := by
    convert hV using 0
  have hU'_li : LinearIndependent 𝕜
      (fun r : Fin R => fun i : Fin I => U' i r) := by
    convert hU' using 0
  have hV'_li : LinearIndependent 𝕜
      (fun r : Fin R => fun j : Fin J => V' j r) := by
    convert hV' using 0
  -- ### Step A: Pick generic `a, b` via A8 applied to W.
  obtain ⟨a, b, hWa_nonzero, hWb_nonzero, h_ratios_distinct⟩ :=
    Tensor.exists_generic_contraction hr W (by intro r t hrt; exact hW r t hrt)
  -- Define diagonal entries.
  set Da : Fin R → 𝕜 := Wᵀ *ᵥ a with hDa_def
  set Db : Fin R → 𝕜 := Wᵀ *ᵥ b with hDb_def
  set D'a : Fin R → 𝕜 := (W')ᵀ *ᵥ a with hD'a_def
  set D'b : Fin R → 𝕜 := (W')ᵀ *ᵥ b with hD'b_def
  -- ### Step B: Derive matrix-pencil equalities.
  -- A_pencil := cp.tensor ×̄₃ a, B_pencil := cp.tensor ×̄₃ b, viewed as I×J matrices.
  -- The structural lemma says A_pencil entry (i,j) = (U · diag(Da) · Vᵀ) i j.
  have h_pencil_a : (U * Matrix.diagonal Da * Vᵀ : Matrix (Fin I) (Fin J) 𝕜) =
      (U' * Matrix.diagonal D'a * (V')ᵀ : Matrix (Fin I) (Fin J) 𝕜) := by
    funext i j
    have hcp : (cp.tensor.vNModeProd 2 a) (fun d => match d with
        | ⟨0, _⟩ => i | ⟨1, _⟩ => j | ⟨n+2, hn⟩ => absurd hn (by omega)) =
        (U * Matrix.diagonal Da * Vᵀ) i j :=
      cpDecomp_three_vNModeProd_two_eq_matrix_mul cp a _
    have hcp' : (cp'.tensor.vNModeProd 2 a) (fun d => match d with
        | ⟨0, _⟩ => i | ⟨1, _⟩ => j | ⟨n+2, hn⟩ => absurd hn (by omega)) =
        (U' * Matrix.diagonal D'a * (V')ᵀ) i j :=
      cpDecomp_three_vNModeProd_two_eq_matrix_mul cp' a _
    rw [← hcp, ← hcp', h]
  have h_pencil_b : (U * Matrix.diagonal Db * Vᵀ : Matrix (Fin I) (Fin J) 𝕜) =
      (U' * Matrix.diagonal D'b * (V')ᵀ : Matrix (Fin I) (Fin J) 𝕜) := by
    funext i j
    have hcp : (cp.tensor.vNModeProd 2 b) (fun d => match d with
        | ⟨0, _⟩ => i | ⟨1, _⟩ => j | ⟨n+2, hn⟩ => absurd hn (by omega)) =
        (U * Matrix.diagonal Db * Vᵀ) i j :=
      cpDecomp_three_vNModeProd_two_eq_matrix_mul cp b _
    have hcp' : (cp'.tensor.vNModeProd 2 b) (fun d => match d with
        | ⟨0, _⟩ => i | ⟨1, _⟩ => j | ⟨n+2, hn⟩ => absurd hn (by omega)) =
        (U' * Matrix.diagonal D'b * (V')ᵀ) i j :=
      cpDecomp_three_vNModeProd_two_eq_matrix_mul cp' b _
    rw [← hcp, ← hcp', h]
  -- ### Step B (cont.): Derive D'b ≠ 0 from h_pencil_b plus LI of U', V'.
  -- The key idea: U·diag(Db)·Vᵀ has rank exactly R (full rank decomposition,
  -- since U has LI columns, V has LI columns, and diag(Db) is invertible).
  -- The same matrix equals U'·diag(D'b)·(V')ᵀ, which has rank ≤ #{r : D'b r ≠ 0}.
  -- So at least R entries of D'b are nonzero — hence all R are.
  -- Concretely: assume D'b r₀ = 0 for some r₀. Then on the range side, only
  -- R-1 columns participate, so column r₀ of `U · diag(Db) · (Vᵀ ⋯)` would have
  -- a non-trivial kernel relation with the other columns — contradicting LI.
  -- We prove the cleaner pencil-quotient equality and use A7's hypotheses.
  --
  -- Direct proof: pick any r₀ with D'b r₀ = 0. Since `Db t ≠ 0` for all t,
  -- `(U · diag Db · Vᵀ) · V` has columns `Db t · U·t` (all nonzero, LI by
  -- hU_li and hd nonzero). On the cp' side, `(U' · diag D'b · (V')ᵀ) · V'`
  -- has column r₀ equal to `D'b r₀ · U'·r₀ * (V')ᵀ · V' (r₀, r₀)` which
  -- collapses to 0 if D'b r₀ = 0. But the two products only equal each other
  -- *after right-multiplying by both V's, which doesn't directly contradict.*
  --
  -- We take the even cleaner route: right-multiply h_pencil_b by `(V')ᵀ.pinv`.
  -- LHS becomes `U · diag Db · Vᵀ · (V')ᵀ.pinv`. RHS becomes `U' · diag D'b ·
  -- (V')ᵀ · (V')ᵀ.pinv = U' · diag D'b · 1 = U' · diag D'b`. Now the column
  -- r of the RHS is `D'b r · U'·r`. On the LHS, this is some I×R matrix.
  -- Apply hU'_li to LI of U' columns: if D'b r = 0 then LHS column r = 0, but
  -- LHS column r = `U · diag Db · Vᵀ · (V')ᵀ.pinv·r`. This need not be zero!
  -- So we need a different angle.
  --
  -- Alternative: apply pinv on the LEFT. (U · diag Db · Vᵀ).pinv * (cp_pencil).
  -- Or: just invoke the rank argument abstractly via A7.
  --
  -- Most direct: we use the cp' version of pencil-quotient. The pencil
  -- quotient `A * B.pinv = U · diag(Da/Db) · U.pinv` is well-defined for
  -- both cp and cp'. By transitivity, both equal each other. A7 then gives
  -- π. To compute the cp'-side quotient, we need D'b ≠ 0; we deduce this
  -- by the argument: if D'b r = 0, then `B = U' · diag D'b · (V')ᵀ` has
  -- a zero column in `diag D'b`, so its rank is ≤ R-1; but we know B
  -- (= U · diag Db · Vᵀ) has rank exactly R (since Db ≠ 0). Contradiction.
  --
  -- Formally: `(U' · diag D'b · (V')ᵀ).pinv * (U' · diag D'b · (V')ᵀ) = 1`
  -- only if `U' · diag D'b · (V')ᵀ` has full column rank `J` (or row rank).
  -- For our use, we go via the pencil-quotient direction.
  -- For a contradiction, suppose D'b r₀ = 0. Then column r₀ of
  -- `(diag D'b)` is zero. Then `U' · diag D'b · (V')ᵀ` has a particular
  -- form: at row i, col j, it's `∑ r, U' i r · D'b r · V' j r`. Since D'b r₀ = 0,
  -- the r₀ summand drops, so the matrix only depends on R-1 columns of U'
  -- and V'. We use `B = U · diag Db · Vᵀ` to derive a contradiction.
  --
  -- Easier route: Right-multiply h_pencil_b by V' on the right. RHS:
  -- `U' · diag D'b · (V')ᵀ · V'`. Now `(V')ᵀ · V'` is R×R and nonsingular
  -- (Gram of LI columns); its r-th diagonal entry is ‖V'·r‖². But that's
  -- in inner-product land — we don't need orthonormality. We need just
  -- LI which gives `(V')ᵀ · V'` nonsingular: this requires Gram nonsingular,
  -- which in turn requires inner product / `RCLike` reasoning.
  --
  -- Cleanest: invoke A2 with `V'.pinv`. (V').pinv * V' = 1 (A2 on V'),
  -- but we want V'·(something) on the right of the equation. We use
  -- A3: `(V'ᵀ).pinv = (V')ᵀ.pinv`, and right-multiply by `(V'ᵀ).pinv` to get
  --   LHS' := U · diag Db · Vᵀ · (V'ᵀ).pinv  (some I × R matrix)
  --   RHS' := U' · diag D'b
  -- This is the I × R matrix whose r-th column is D'b r · U'·r. By LI of
  -- U'·r, this is zero iff D'b r = 0. We then use the LHS' to show LHS' r ≠ 0
  -- for any r — but that needs to know LHS' has any structure we can exploit.
  --
  -- Take the cleanest possible argument: contrapositive. Suppose D'b r₀ = 0
  -- for some r₀. Then the r₀-column of `(U' · diag D'b)` is zero, so
  -- `U' · diag D'b` has a non-trivial right kernel: the vector `e_{r₀}` is
  -- in its right kernel. Since `U' · diag D'b = U · diag Db · Vᵀ · (V'ᵀ).pinv`,
  -- this same vector is in the right kernel of `U · diag Db · Vᵀ · (V'ᵀ).pinv`.
  -- That is: `(U · diag Db · Vᵀ · (V'ᵀ).pinv) · e_{r₀} = 0`, i.e. column r₀
  -- of `U · diag Db · Vᵀ · (V'ᵀ).pinv` is zero.
  -- Now, U · diag Db has LI columns (by lin_indep_mul_diagonal). So
  -- `(U · diag Db).pinv · (U · diag Db) = 1` (A2). Apply this on the left:
  -- `(U · diag Db).pinv · (U · diag Db · Vᵀ · (V'ᵀ).pinv) · e_{r₀}`
  --   = `Vᵀ · (V'ᵀ).pinv · e_{r₀}` = `(V'ᵀ).pinv · e_{r₀}` * something.
  -- Using A2 again: `Vᵀ · (V'ᵀ).pinv` is some R×R matrix.
  -- For the contradiction, we need this NOT to have e_{r₀} in its kernel.
  --
  -- This is getting tangled. Let's invoke a more abstract argument: we
  -- combine pencil_quotient_eq for cp (always works) with the corresponding
  -- equation for cp' (only works if D'b ≠ 0). We prove BOTH pencil_quotient
  -- versions equal each other → A7 gives π → use A7's surjective Λ-matching
  -- to deduce that all D'b values are matched to nonzero ratios in `Da/Db`,
  -- hence each D'b r ≠ 0 (by inverting the ratio relation).
  --
  -- This forward direction works! But it requires the pencil-quotient for
  -- cp' to be derivable assuming D'b ≠ 0, which is what we're trying to
  -- prove. Circular.
  --
  -- Resolution: avoid the issue by instead applying A7 with cp playing the
  -- role of `U' Λ' U'.pinv` and cp' similarly. We can derive the cp side's
  -- pencil quotient `U · diag(Da/Db) · U.pinv` unconditionally. For the cp'
  -- side, we *don't need* D'b nowhere zero to apply A7 — we just need *some*
  -- factorization with LI columns. We can use the equation
  -- `U · diag(Da/Db) · U.pinv = (cp' pencil quotient)` even WITHOUT having
  -- decomposed the cp' pencil quotient as `U' · diag(D'a/D'b) · U'.pinv`.
  -- Specifically, A7 only requires we *have* a factorization, not that we've
  -- algebraically derived it. So we apply pencil_quotient_eq once (for cp),
  -- and use the LHS of the original h_pencil_a / h_pencil_b equations
  -- directly to the result.
  --
  -- Wait — A7 needs `M = U Λ U.pinv = U' Λ' U'.pinv` as input. So we need
  -- to express the cp' pencil quotient as `U' · diag(Λ') · U'.pinv` for some
  -- Λ'. That means we DO need pencil_quotient_eq for cp'. Which needs D'b ≠ 0.
  --
  -- So the constructive proof structurally does need D'b ≠ 0 at this step.
  -- We derive it from h_pencil_b + LI of U', V', and Db ≠ 0:
  -- "if D'b r = 0 for some r, the cp'-pencil drops a rank, contradicting
  -- the cp-pencil's rank R."
  -- A clean formulation: project both sides onto specific (i,j)-pair entries
  -- chosen so the absurd identity is visible, OR use a rank-argument via A2/A3.
  --
  -- Concretely: assume by contradiction D'b r₀ = 0. Right-multiply h_pencil_b
  -- by `(V')ᵀ.pinv` from the right, then by `e_{r₀}` viewed as an R-vector:
  --   LHS · e_{r₀} = (U · diag Db · Vᵀ · (V'ᵀ).pinv)·e_{r₀} =: x ∈ Fin I → 𝕜
  --   RHS · e_{r₀} = (U' · diag D'b)·e_{r₀} = D'b r₀ • U'·r₀ = 0.
  -- So x = 0. Now apply (U · diag Db).pinv from the left:
  --   (U · diag Db).pinv · x = (U · diag Db).pinv · (U · diag Db) · Vᵀ · (V'ᵀ).pinv · e_{r₀}
  --                        = 1 · Vᵀ · (V'ᵀ).pinv · e_{r₀}
  --                        = Vᵀ · (V'ᵀ).pinv · e_{r₀}.
  -- And (U · diag Db).pinv · 0 = 0. So Vᵀ · (V'ᵀ).pinv · e_{r₀} = 0.
  --
  -- Now apply V on the left: V · Vᵀ · (V'ᵀ).pinv · e_{r₀} = 0. Hmm, that
  -- doesn't directly help either, because V·Vᵀ is rank R but not 1.
  --
  -- Apply (Vᵀ).pinv on the left of (Vᵀ · (V'ᵀ).pinv · e_{r₀} = 0):
  -- since Vᵀ has LI rows (= V's LI cols), A3 gives V·(Vᵀ) = 1? No wait,
  -- Vᵀ has LI rows means (Vᵀ) is full row rank, so by A3, Vᵀ * (Vᵀ).pinv = 1.
  -- That gives us a right inverse but not what we want.
  -- For a LEFT inverse of Vᵀ we'd need Vᵀ to have LI cols, i.e. V has LI rows.
  -- We don't have that (V is J × R with J ≥ R).
  --
  -- Approach via cp's perspective: V has LI cols, so V.pinv · V = 1 (A2).
  -- We have Vᵀ · (V'ᵀ).pinv · e_{r₀} = 0. Hmm.
  --
  -- Final alternative: invoke A7 directly with the *cp pencil quotient* and
  -- derive that D'b r ≠ 0 by examining what A7 forces. But A7 doesn't
  -- directly assert anything about D'b.
  --
  -- ###############################################################
  -- Resolution: invoke `rank_preservation_diag` (a *proved* lemma above).
  -- It captures the rank-preservation fact: if
  -- `U · diag d · Vᵀ = U' · diag d' · (V')ᵀ` with `U, V, U', V'` having
  -- LI columns and `d` everywhere nonzero, then `d'` is everywhere
  -- nonzero. The proof composes Mathlib's `Matrix.rank_diagonal`,
  -- `Matrix.rank_mul_le_left`/`_right`, and pinv axioms A2/A3:
  --   `U.pinv · (U · diag d · Vᵀ) · (Vᵀ).pinv = diag d`
  -- gives the lower bound `rank(diag d) ≤ rank(U · diag d · Vᵀ)`,
  -- combined with the standard upper bound to pin both sides at `R`.
  -- ###############################################################
  have hD'b_nonzero : ∀ r : Fin R, D'b r ≠ 0 :=
    rank_preservation_diag U V U' V' Db D'b hU_li hV_li hV'_li
      hWb_nonzero h_pencil_b
  -- Symmetrically, derive D'a everywhere nonzero — wait, this isn't always
  -- true. The genericity of `a` gives Da nonzero, but D'a might have zero
  -- entries — this is fine because we use the ratio (Da r)/(Db r) and only
  -- need Db, D'b nonzero for the pinv step. We do NOT need D'a ≠ 0. The
  -- ratio Λ' r = D'a r / D'b r only requires D'b r ≠ 0. Good.
  -- ### Steps C–E: Apply the eigenvector-recovery theorem
  -- `pencil_eigendecomp_unique`, which combines `pencil_quotient_eq`
  -- (twice) with axiom A7. This corresponds to Lemma 8 part 2 in the
  -- lecture.
  -- Show injectivity of `Λ_U r = Da r / Db r = Da r * (Db r)⁻¹`.
  have hΛ_U_inj : Function.Injective (fun r : Fin R => Da r * (Db r)⁻¹) := by
    intro r t hrt
    -- hrt : Da r * (Db r)⁻¹ = Da t * (Db t)⁻¹
    by_contra hne
    -- Cross-multiply: Da r * Db t = Da t * Db r.
    have h_cross : Da r * Db t = Da t * Db r := by
      have hb_r := hWb_nonzero r
      have hb_t := hWb_nonzero t
      field_simp at hrt
      linear_combination hrt
    exact h_ratios_distinct r t hne h_cross
  obtain ⟨π, hΛ_match, hcol_match⟩ :=
    pencil_eigendecomp_unique U U' V V' Da Db D'a D'b
      hU_li hV_li hU'_li hV'_li hWb_nonzero hD'b_nonzero hΛ_U_inj
      h_pencil_a h_pencil_b
  -- hΛ_match r : Da r * (Db r)⁻¹ = D'a (π r) * (D'b (π r))⁻¹
  -- hcol_match r : ∃ c ≠ 0, ∀ i, U' i (π r) = c * U i r
  -- Choose column scalars c_U r.
  choose cU hcU_ne hcU_eq using hcol_match
  -- ### Step F: Apply A7 to V via the second pencil quotient.
  -- We need to derive V's column relations. The second pencil quotient is
  -- `(A.pinv * B)ᵀ`. Its derivation:
  -- A.pinv = (Vᵀ).pinv * (diag Da).pinv * U.pinv (when Da ≠ 0).
  -- We need Da ≠ 0 — which is hWa_nonzero!
  -- A.pinv * B = (Vᵀ).pinv * (diag Da).pinv * U.pinv * U * diag(Db) * Vᵀ
  --            = (Vᵀ).pinv * diag(1/Da) * 1 * diag(Db) * Vᵀ
  --            = (Vᵀ).pinv * diag(Db/Da) * Vᵀ.
  -- Take transpose: (A.pinv * B)ᵀ = V * diag(Db/Da) * ((Vᵀ).pinv)ᵀ
  --                                = V * diag(Db/Da) * V.pinv  [via A6 backwards].
  -- Symmetric application of pencil_quotient_eq with roles of U/V swapped
  -- and Da/Db swapped. The LHS gives `(A.pinv * B)ᵀ`, which factors as
  -- `V · diag(Db/Da) · V.pinv`.
  -- Ratios Λ_V r = Db r / Da r. Compute on cp' side similarly, requiring
  -- D'a everywhere nonzero — which we DON'T have! So we need a different
  -- approach for the V columns.
  --
  -- Alternative: derive V column relations directly from h_pencil_a and the
  -- already-known U column relations. Specifically: from
  -- `U · diag Da · Vᵀ = U' · diag D'a · (V')ᵀ` and U' = U · (perm) · diag(cU),
  -- we can solve for V'.
  --
  -- Let's pursue this direct route. Apply the change of basis on the cp' side:
  -- U'·r = U·(π.symm r)·c (where c is the appropriate scalar for π.symm r).
  -- Equivalently U' = U · P · diag(cU') where P is the permutation matrix.
  -- This rewrites the cp' RHS in terms of U.
  -- After rearrangement, we get a relation `Vᵀ · M = (V')ᵀ · diag(...) · (perm) ...`.
  -- Then by `Vᵀ` having LI rows (full row rank), we can solve for V'.
  --
  -- Cleanest approach: use the equation `U · diag Da · Vᵀ = U' · diag D'a · (V')ᵀ`
  -- and "left-cancel" by `U.pinv` (using A2 since U has LI cols).
  -- After: `diag Da · Vᵀ = U.pinv · U' · diag D'a · (V')ᵀ`.
  -- Now U.pinv · U' is an R × R matrix; from U' = "permuted-scaled U", we can
  -- compute U.pinv · U' explicitly: it's a permutation matrix scaled by cU.
  -- Specifically, (U.pinv · U') r t = (col t of U.pinv · U') r = ?.
  -- Since `U' j t = cU (π.symm t) * U j (π.symm t)` (where π.symm t maps `t` to
  -- the original index r s.t. π r = t), we have `(col t of U') = cU (π.symm t) ·
  -- (col (π.symm t) of U)`. Hence `U.pinv · (col t of U') = cU (π.symm t) ·
  -- (U.pinv · col (π.symm t) of U) = cU (π.symm t) · e_{π.symm t}` (using A2).
  -- So `U.pinv · U' = (e_{π.symm t} scaled by cU(π.symm t)) at column t`,
  -- which means `(U.pinv · U') r t = cU r · [r = π.symm t] = cU r · [π r = t]`.
  -- That's the matrix `P · diag cU` where P is the permutation matrix of π.
  --
  -- Substituting back: `diag Da · Vᵀ = P · diag cU · diag D'a · (V')ᵀ`
  --                                  = `P · diag (cU · D'a∘π?) · (V')ᵀ`.
  -- (Here `cU · D'a∘?` need careful indexing, but the structure is clear.)
  --
  -- Compare row-by-row: row r of Vᵀ scaled by Da r = some scaled row of (V')ᵀ
  -- after permutation. This precisely gives V column relations:
  -- `V · t = (Da (π.symm t) / (cU (π.symm t) · D'a t)) · V' · t` ... or similar.
  --
  -- This is doable but tedious. We adopt a more pragmatic shortcut: prove the
  -- W column relations directly via the third unfolding (KR FCR), which gives
  -- the FactorEquiv₃ assembly without explicitly going through V's column
  -- relations. The triple-scaling product = 1 then determines c_V via c_W.
  --
  -- Specifically, the FactorEquiv₃ statement asks for the rank-1 outer product
  -- equality. Pointwise, this is:
  --   factors r 0 i * factors r 1 j * factors r 2 k =
  --     factors' (π r) 0 i * factors' (π r) 1 j * factors' (π r) 2 k.
  -- Substituting `factors' (π r) 0 i = cU r * factors r 0 i` (Step E), we get
  -- a relation involving only V'·(πr) and W'·(πr) (no V/W's). We then use the
  -- KR identity at mode 1 (or equivalently the original tensor equality
  -- `cp.tensor = cp'.tensor`) to derive the matching scalings on V and W.
  --
  -- The cleanest way is:
  -- (1) From `cp.tensor = cp'.tensor`, expand both sides.
  -- (2) Substitute U' columns via Step E.
  -- (3) Get pointwise:
  --     ∑ r factors r 0 i · factors r 1 j · factors r 2 k =
  --       ∑ r cU r · factors r 0 i · factors' (π r) 1 j · factors' (π r) 2 k
  -- (4) Reindex the RHS sum by π.symm:
  --     RHS = ∑ r' cU (π.symm r') · factors (π.symm r') 0 i · factors' r' 1 j · factors' r' 2 k.
  --    Let r := π.symm r' on LHS to match. So in `r`-form:
  --    cU r · factors r 0 i · factors' (π r) 1 j · factors' (π r) 2 k
  --      summed over r — but this still uses `r` indexing factors' at π r.
  -- (5) Use Khatri-Rao FCR for U, V (which gives `V ⊙ U` LI columns) to
  --     conclude that the equation `∑ r c_r · u_r ⊙ v_r · w_r(k) = 0` for all k
  --     forces each `c_r · w_r(k) = 0`, i.e., the "stack" coefficients match.
  --
  -- This last step is what we'll formalize. It directly gives us the W (and
  -- jointly V/W) relations without separately running A7 for V.
  -- ##############
  -- We formalize this directly by proving the FactorEquiv₃ goal pointwise.
  refine ⟨π, fun r => ?_⟩
  -- Goal: rank-1 outer products at index r match (LHS uses π r for cp').
  ext idx
  -- idx : (d : Fin (1 + 1 + 1)) → Fin (combinedDims ... d)
  set i : Fin I := splitLeft (splitLeft idx) 0 with hi_def
  set j : Fin J := splitRight (splitLeft idx) 0 with hj_def
  set k : Fin K := splitRight idx 0 with hk_def
  -- Reduce both sides via outerProd_apply and toTensor1_apply.
  simp only [Tensor.outerProd_apply, Tensor.toTensor1_apply]
  -- Reduce indices: splitLeft (splitLeft idx) 0 = i, etc. Already by `set`.
  -- LHS = factors r 0 i * factors r 1 j * factors r 2 k
  -- RHS = factors' (π r) 0 i * factors' (π r) 1 j * factors' (π r) 2 k
  -- Substitute U' column relation: factors' (π r) 0 i = cU r * factors r 0 i.
  have h_U' : cp'.factors (π r) 0 i = cU r * cp.factors r 0 i := by
    -- hcU_eq r i : U' i (π r) = cU r * U i r
    -- where U' i (π r) = cp'.factorMatrix 0 i (π r) = cp'.factors (π r) 0 i,
    -- and similarly for U.
    have h := hcU_eq r i
    simp only [hU_def, hU'_def, CPDecomp.factorMatrix_apply] at h
    exact h
  -- Now use `cp.tensor = cp'.tensor` plus KR FCR to get a paired V·W relation.
  -- Pointwise: ∑ r, factors r 0 i' * factors r 1 j' * factors r 2 k' =
  --            ∑ r', factors' r' 0 i' * factors' r' 1 j' * factors' r' 2 k'.
  -- Reindex RHS by π:
  --   = ∑ r, factors' (π r) 0 i' * factors' (π r) 1 j' * factors' (π r) 2 k'
  --   = ∑ r, cU r * factors r 0 i' * factors' (π r) 1 j' * factors' (π r) 2 k'.
  -- So: ∑ r, factors r 0 i' * (factors r 1 j' * factors r 2 k' -
  --             cU r * factors' (π r) 1 j' * factors' (π r) 2 k') = 0
  -- for all (i', j', k').
  -- Fix k' = k, j' = j; varying i' gives: by LI of U's columns in i',
  -- `factors r 1 j * factors r 2 k = cU r * factors' (π r) 1 j * factors' (π r) 2 k`
  -- for all r, j, k.
  -- Now fix r, k = k; varying j: by LI of V's columns... hmm, this is where it
  -- gets subtle because the cU r is a constant per r.
  -- Actually we need: for all r, ∀ j ∀ k, factors r 1 j · factors r 2 k =
  --                              cU r · factors' (π r) 1 j · factors' (π r) 2 k.
  -- This is the rank-1 outer product `(v_r ⊗ w_r) = cU r · (v'_{π r} ⊗ w'_{π r})`
  -- which gives the answer at (j, k) directly.
  -- Set i' = i, j' = j, k' = k below for the assembly.
  have h_outer_two_modes : ∀ r' : Fin R, ∀ jj : Fin J, ∀ kk : Fin K,
      cp.factors r' 1 jj * cp.factors r' 2 kk =
        cU r' * (cp'.factors (π r') 1 jj * cp'.factors (π r') 2 kk) := by
    -- Fix jj, kk; consider tensor equality at all (i', jj, kk).
    intro r' jj kk
    -- Get the pointwise tensor equation and view it as LI in i'.
    have h_pt : ∀ ii : Fin I,
        ∑ t : Fin R, cp.factors t 0 ii * cp.factors t 1 jj * cp.factors t 2 kk =
          ∑ t : Fin R, cp'.factors t 0 ii * cp'.factors t 1 jj * cp'.factors t 2 kk := by
      intro ii
      have heq := congrFun h.symm
        (fun d : Fin 3 => match d with
         | ⟨0, _⟩ => ii | ⟨1, _⟩ => jj | ⟨2, _⟩ => kk | ⟨n+3, hn⟩ => absurd hn (by omega))
      simp only [CPDecomp.tensor, Fin.prod_univ_three] at heq
      exact heq
    -- Reindex the RHS sum by π.
    have h_pt' : ∀ ii : Fin I,
        ∑ t : Fin R, cp.factors t 0 ii * cp.factors t 1 jj * cp.factors t 2 kk =
          ∑ t : Fin R, cp'.factors (π t) 0 ii * cp'.factors (π t) 1 jj
            * cp'.factors (π t) 2 kk := by
      intro ii
      rw [h_pt ii]
      exact (Equiv.sum_comp π _).symm
    -- Substitute U' relation on RHS.
    have h_pt'' : ∀ ii : Fin I,
        ∑ t : Fin R, cp.factors t 0 ii * cp.factors t 1 jj * cp.factors t 2 kk =
          ∑ t : Fin R, cp.factors t 0 ii * (cU t *
            (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk)) := by
      intro ii
      rw [h_pt' ii]
      refine Finset.sum_congr rfl fun t _ => ?_
      have hU'_t : cp'.factors (π t) 0 ii = cU t * cp.factors t 0 ii := by
        have h := hcU_eq t ii
        simp only [hU_def, hU'_def, CPDecomp.factorMatrix_apply] at h
        exact h
      rw [hU'_t]; ring
    -- Subtract: ∑ t, factors t 0 ii * (factors t 1 jj * factors t 2 kk -
    --                                  cU t * factors' (πt) 1 jj * factors' (πt) 2 kk) = 0
    have h_diff : ∀ ii : Fin I,
        ∑ t : Fin R, cp.factors t 0 ii *
          (cp.factors t 1 jj * cp.factors t 2 kk
            - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk)) = 0 := by
      intro ii
      have := h_pt'' ii
      rw [show (∑ t : Fin R, cp.factors t 0 ii *
          (cp.factors t 1 jj * cp.factors t 2 kk
            - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk)))
          = (∑ t : Fin R, cp.factors t 0 ii * cp.factors t 1 jj * cp.factors t 2 kk)
            - (∑ t : Fin R, cp.factors t 0 ii *
              (cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk))) from by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun t _ => ?_; ring]
      linear_combination this
    -- Now apply LI of U's columns: the coefficients
    -- (factors t 1 jj * factors t 2 kk - cU t * (factors' (πt) 1 jj * factors' (πt) 2 kk))
    -- as a function of t are zero.
    have h_zero_coeff : ∀ t : Fin R,
        cp.factors t 1 jj * cp.factors t 2 kk
          - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk) = 0 := by
      apply (Fintype.linearIndependent_iff.mp hU)
        (fun t => cp.factors t 1 jj * cp.factors t 2 kk
          - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk))
      funext ii
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
      have := h_diff ii
      rw [show (∑ t : Fin R,
          (cp.factors t 1 jj * cp.factors t 2 kk
            - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk))
          * cp.factors t 0 ii) = ∑ t : Fin R, cp.factors t 0 ii *
          (cp.factors t 1 jj * cp.factors t 2 kk
            - cU t * (cp'.factors (π t) 1 jj * cp'.factors (π t) 2 kk)) from by
        refine Finset.sum_congr rfl fun t _ => by ring]
      exact this
    have h_eq_at_r := h_zero_coeff r'
    linear_combination h_eq_at_r
  -- Apply h_outer_two_modes at (j, k).
  have hgoal := h_outer_two_modes r j k
  -- Now combine with h_U' to get the desired pointwise equality.
  -- LHS: cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 k
  -- RHS: cp'.factors (π r) 0 i * cp'.factors (π r) 1 j * cp'.factors (π r) 2 k
  --    = (cU r · cp.factors r 0 i) * (cp'.factors (π r) 1 j * cp'.factors (π r) 2 k)
  --    = cp.factors r 0 i · (cU r · (cp'.factors (π r) 1 j * cp'.factors (π r) 2 k))
  --    = cp.factors r 0 i · (cp.factors r 1 j · cp.factors r 2 k)  [via hgoal]
  --    = LHS.
  rw [h_U']
  have key : cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 k
       = cp.factors r 0 i *
          (cU r * (cp'.factors (π r) 1 j * cp'.factors (π r) 2 k)) := by
    rw [show cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 k
            = cp.factors r 0 i * (cp.factors r 1 j * cp.factors r 2 k) from by ring]
    rw [hgoal]
  linear_combination key

/-! ### Lemma 9: Khatri-Rao linear system for `Ŵ` recovery

The simultaneous-diagonalization algorithm's line 8 recovers `Ŵ` by solving
the linear system

  `(V̂ ⊙ Û) Z = X^⊤_{(3)}`,

then taking `Ŵ = Z^⊤`. Lemma 9 in the lecture establishes two properties:

1. **Consistency.** The system has at least one solution, justified by the
   existence of some CP decomposition `⟦Û, V̂, Ŵ⟧ = X` (the lecture proves
   this via Lemma 8, which shows each `û_r` and `v̂_r` is a scaling of some
   original factor, so a valid `ŵ_r` can be obtained by absorbing the scaling
   factors into the original `w_{π(r)}`).

2. **Soundness.** Every solution `Z` of the system yields a valid third
   factor: setting `Ŵ = Z^⊤` gives `⟦Û, V̂, Ŵ⟧ = X`.

Stated together, the existence-of-some-`Ŵ` hypothesis (consequence of
Lemma 8) implies both consistency and soundness of the linear system. -/

/-- **Cardinality identity at order 3, mode 2.** The product of `![I, J, L]`
    over `Finset.univ.erase 2` (i.e., `Tensor.prodExcept 2 ![I, J, L]`) is
    `J * I` — the row type of `V ⊙ₜ U` for `V : J × R, U : I × R`. Used to
    bridge the column type of `X.modeUnfold 2` (a
    `Fin (Tensor.prodExcept 2 ![I, J, L])`) with the Khatri-Rao row type
    `Fin (J * I)`. The expression `∏ d ∈ univ.erase 2, ![I, J, L] d` decodes
    to `![I,J,L] 1 * (![I,J,L] 0 * 1) = J * I` via `Finset.prod_pair`
    composed with the literal evaluation of `Fin 3` `Finset.univ.erase 2`. -/
private theorem prodExcept_two_three (I J L : ℕ) :
    Tensor.prodExcept (2 : Fin 3) ![I, J, L] = J * I := by
  change ∏ d ∈ (Finset.univ : Finset (Fin 3)).erase 2, ![I, J, L] d = J * I
  -- The erase of 2 in `Fin 3` is the pair `{0, 1}`.
  rw [show ((Finset.univ : Finset (Fin 3)).erase 2) = {0, 1} from by decide]
  rw [Finset.prod_insert (by decide), Finset.prod_singleton]
  change ![I, J, L] 0 * ![I, J, L] 1 = J * I
  simp [Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- **Bridge between `Tensor.modeUnfold 2` and the Khatri-Rao row index.**

    For an order-3 tensor `X : Tensor 𝕜 3 ![I, J, L]`, the mode-2 (lecture:
    mode 3) matricization lives in
    `Matrix (Fin L) (Fin (Tensor.prodExcept 2 ![I, J, L])) 𝕜`, whose column
    type `Fin (Tensor.prodExcept 2 ![I, J, L])` is the unevaluated product
    `∏ d ∈ univ.erase 2, ![I, J, L] d = J * I`. The row type of
    `V ⊙ₜ U` (with `V : J × R, U : I × R`) is `Fin (J * I)`.

    Both flattenings follow the Kolda-Bader convention (mode 0 varies
    fastest): if `c = finProdFinEquiv (j, i) : Fin (J * I)` then
    `c.val = i + I * j`, and likewise
    `(remainingModesEquiv 2 ![I, J, L] (fun ⟨k,_⟩ => idx k)).val` equals
    `i + I * j` when `idx 0 = i, idx 1 = j`. The only mismatch is at the
    cardinality level (`J * I` vs. `Tensor.prodExcept 2 ![I, J, L]`),
    bridged by the val-preserving `Fin.cast`.

    The lemma below evaluates `X.modeUnfold 2 l (Fin.cast _ c)` at
    `c = finProdFinEquiv (j, i)` and returns `X (i, j, l)`, matching the
    pointwise formula previously expressed via the now-deleted
    `modeUnfold2_KR` helper. -/
private theorem modeUnfold_two_apply_finProdFinEquiv
    {I J L : ℕ} (X : Tensor 𝕜 3 ![I, J, L])
    (l : Fin L) (j : Fin J) (i : Fin I) :
    X.modeUnfold 2 l
        (Fin.cast (prodExcept_two_three I J L).symm (finProdFinEquiv (j, i))) =
      X (fun d : Fin 3 => match d with
        | ⟨0, _⟩ => i
        | ⟨1, _⟩ => j
        | ⟨2, _⟩ => l
        | ⟨_ + 3, hbad⟩ => by omega) := by
  -- Candidate multi-index `idx : (d : Fin 3) → Fin (![I,J,L] d)` so that
  -- `idx 0 = i, idx 1 = j, idx 2 = l`.
  set idx : (d : Fin 3) → Fin (![I, J, L] d) := fun d => match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => l
    | ⟨_ + 3, hbad⟩ => by omega with hidx_def
  -- Identify `l = idx 2` and rewrite the column index as the image of `idx`
  -- under `remainingModesEquiv 2`. The val equality on column indices is
  -- the numerical identity `i + I * j = i + I * j` (both flat indices use
  -- mode 0 fastest), wrapped in a `Fin.cast`.
  have hl : l = idx 2 := rfl
  have hcol :
      Fin.cast (prodExcept_two_three I J L).symm (finProdFinEquiv (j, i)) =
        Tensor.remainingModesEquiv 2 ![I, J, L]
          (fun (d : {k : Fin 3 // k ≠ 2}) => idx d.val) := by
    apply Fin.ext
    -- Both sides have val `i.val + I * j.val`. The LHS reduces via
    -- `Fin.coe_cast` + `finProdFinEquiv` definition. The RHS reduces by
    -- the recursive definition of `remainingModesEquiv` at order 3, mode 2:
    -- `prodCast` (a `Fin.castOrderIso`) preserves val, and the inner
    -- `multiIndexEquiv` on the 2-tuple `(i, j)` flattens to `i + I * j`.
    rfl
  rw [hl, hcol]
  rw [Tensor.modeUnfold_apply]
  rw [Equiv.symm_apply_apply]
  -- The pair `(idx 2, fun ⟨k,_⟩ => idx k.val)` is the forward image of
  -- `separateModeEquiv 2 ![I,J,L]` applied to `idx`, so the symm returns `idx`.
  have h_sep :
      (Tensor.separateModeEquiv (2 : Fin 3) ![I, J, L]).symm
          (idx 2, fun (d : {k : Fin 3 // k ≠ 2}) => idx d.val) = idx :=
    (Tensor.separateModeEquiv (2 : Fin 3) ![I, J, L]).symm_apply_apply idx
  rw [h_sep]

/-- **Lemma 9 (CP recovery — third factor): Khatri-Rao system consistency
    and soundness.**

    Suppose `X : Tensor 𝕜 3 ![I, J, L]` admits some CP decomposition
    `⟦Û, V̂, Ŵ⟧ = X` (the lecture's hypothesis, established by Lemma 8 via
    the recovery of `Û, V̂` up to scaling and the absorption of the scalars
    into the original mode-3 factor). Then the linear system

      `(V̂ ⊙ₜ Û) Z = (X_{(3)})^⊤`

    is **consistent**, and **every solution** `Z` yields a valid third
    factor `Ŵ = Z^⊤` satisfying `⟦Û, V̂, Ŵ⟧ = X`.

    The right-hand side `(X_{(3)})^⊤` is the transpose of `X.modeUnfold 2`,
    with column type `Fin (Tensor.prodExcept 2 ![I, J, L])` reindexed via
    the val-preserving `Fin.cast` along the cardinality identity
    `Tensor.prodExcept 2 ![I, J, L] = J * I` (the `Fin (J * I)` row type of
    `V̂ ⊙ₜ Û`). Both column orderings follow Kolda-Bader (mode 0 varies
    fastest), so the cast is purely a type-level bridge — see
    `modeUnfold_two_apply_finProdFinEquiv` above.

    Reference: lecture4_CPDecomposition.org, Lemma 9 (lines 466–473). -/
theorem khatriRao_system_consistent_and_sound
    {I J L R : ℕ}
    (Uhat : Matrix (Fin I) (Fin R) 𝕜)
    (Vhat : Matrix (Fin J) (Fin R) 𝕜)
    (X : Tensor 𝕜 3 ![I, J, L])
    (hX : ∃ What : Matrix (Fin L) (Fin R) 𝕜,
      (⟦Uhat, Vhat, What⟧ : CPDecomp 𝕜 3 ![I, J, L] R).tensor = X) :
    (∃ Z : Matrix (Fin R) (Fin L) 𝕜,
        (Vhat ⊙ₜ Uhat) * Z =
          ((X.modeUnfold 2).submatrix
            id (Fin.cast (prodExcept_two_three I J L).symm))ᵀ) ∧
    (∀ Z : Matrix (Fin R) (Fin L) 𝕜,
        (Vhat ⊙ₜ Uhat) * Z =
            ((X.modeUnfold 2).submatrix
              id (Fin.cast (prodExcept_two_three I J L).symm))ᵀ →
        (⟦Uhat, Vhat, Zᵀ⟧ : CPDecomp 𝕜 3 ![I, J, L] R).tensor = X) := by
  -- Pointwise pivot lemma: the system equation at row `c`, column `l` reduces
  -- to `∑ r, V (c₁) r * U (c₀) r * Z r l = X (i₀, i₁, l)`, which matches
  -- (after `mul_comm`) the bracket-tensor reconstruction
  -- `⟦U, V, Zᵀ⟧.tensor (i₀, i₁, l) = ∑ r, U i₀ r * V i₁ r * Z r l`.
  -- Capture the bidirectional pointwise equivalence in a single lemma.
  have h_iff : ∀ Z : Matrix (Fin R) (Fin L) 𝕜,
      (Vhat ⊙ₜ Uhat) * Z =
        ((X.modeUnfold 2).submatrix
          id (Fin.cast (prodExcept_two_three I J L).symm))ᵀ ↔
      (⟦Uhat, Vhat, Zᵀ⟧ : CPDecomp 𝕜 3 ![I, J, L] R).tensor = X := by
    intro Z
    constructor
    · -- Forward: from the matrix equation, deduce the tensor equation pointwise.
      intro hZ
      ext idx
      -- Set `i₀ = idx 0, i₁ = idx 1, l = idx 2`.
      set i₀ : Fin I := idx 0
      set i₁ : Fin J := idx 1
      set l : Fin L := idx 2
      -- Read off the matrix equation at row `finProdFinEquiv (i₁, i₀)`, column `l`.
      have h_at := congrFun (congrFun hZ (finProdFinEquiv (i₁, i₀))) l
      -- LHS: `(V ⊙ U * Z) (finProdFinEquiv (i₁, i₀)) l`.
      rw [Matrix.mul_apply] at h_at
      -- LHS r-summand: `(V ⊙ U) (finProdFinEquiv (i₁, i₀)) r * Z r l = V i₁ r * U i₀ r * Z r l`.
      have h_sum_eq : ∀ r, (Vhat ⊙ₜ Uhat) (finProdFinEquiv (i₁, i₀)) r * Z r l =
          Vhat i₁ r * Uhat i₀ r * Z r l := by
        intro r
        rw [Matrix.khatriRao_apply_pair]
      rw [Finset.sum_congr rfl (fun r _ => h_sum_eq r)] at h_at
      -- RHS: `((X.modeUnfold 2).submatrix id (Fin.cast _))ᵀ
      --        (finProdFinEquiv (i₁, i₀)) l`
      --      = `X.modeUnfold 2 l (Fin.cast _ (finProdFinEquiv (i₁, i₀)))`.
      rw [Matrix.transpose_apply, Matrix.submatrix_apply, id] at h_at
      -- Apply the bridge lemma to read off `X (i₀, i₁, l)`.
      rw [modeUnfold_two_apply_finProdFinEquiv] at h_at
      -- The bridge returns `X (match d with | 0 => i₀ | 1 => i₁ | 2 => l)`.
      -- Show this equals `X idx`.
      have hX_idx :
          X (fun d : Fin 3 => match d with
              | ⟨0, _⟩ => i₀
              | ⟨1, _⟩ => i₁
              | ⟨2, _⟩ => l
              | ⟨_ + 3, hbad⟩ => by omega) = X idx := by
        congr 1
        funext d
        match d with
        | ⟨0, _⟩ => rfl
        | ⟨1, _⟩ => rfl
        | ⟨2, _⟩ => rfl
        | ⟨n + 3, hn⟩ => omega
      rw [hX_idx] at h_at
      -- Now expand the bracket tensor at `idx`.
      rw [bracket_tensor_apply₃]
      -- Goal: `∑ r, U (idx 0) r * V (idx 1) r * Zᵀ (idx 2) r = X idx`.
      rw [← h_at]
      refine Finset.sum_congr rfl fun r _ => ?_
      rw [Matrix.transpose_apply]
      ring
    · -- Reverse: from the tensor equation, derive the matrix equation pointwise.
      intro hT
      funext c l
      -- Both sides have row type `Fin (J * I)`. Decompose
      -- `c = finProdFinEquiv (i₁, i₀)` using `finProdFinEquiv.symm`.
      set i₁ : Fin J := (finProdFinEquiv.symm c).1 with hi₁
      set i₀ : Fin I := (finProdFinEquiv.symm c).2 with hi₀
      have hc_eq : c = finProdFinEquiv (i₁, i₀) := by
        rw [hi₁, hi₀, Prod.mk.eta]
        exact (Equiv.apply_symm_apply finProdFinEquiv c).symm
      -- LHS at `(c, l)`.
      rw [Matrix.mul_apply]
      have h_sum_eq : ∀ r, (Vhat ⊙ₜ Uhat) c r * Z r l =
          Vhat i₁ r * Uhat i₀ r * Z r l := by
        intro r
        rw [hc_eq, Matrix.khatriRao_apply_pair]
      rw [Finset.sum_congr rfl (fun r _ => h_sum_eq r)]
      -- RHS at `(c, l)`.
      rw [Matrix.transpose_apply, Matrix.submatrix_apply, id]
      -- `(X.modeUnfold 2) l (Fin.cast _ c)`. Substitute
      -- `c = finProdFinEquiv (i₁, i₀)` and apply the bridge.
      rw [hc_eq, modeUnfold_two_apply_finProdFinEquiv]
      -- Use the tensor equation `⟦U, V, Zᵀ⟧.tensor = X` at the appropriate multi-index.
      have h_idx_eq :
          X (fun d : Fin 3 => match d with
              | ⟨0, _⟩ => i₀
              | ⟨1, _⟩ => i₁
              | ⟨2, _⟩ => l
              | ⟨_ + 3, hbad⟩ => by omega) =
            (⟦Uhat, Vhat, Zᵀ⟧ : CPDecomp 𝕜 3 ![I, J, L] R).tensor
              (fun d : Fin 3 => match d with
                | ⟨0, _⟩ => i₀
                | ⟨1, _⟩ => i₁
                | ⟨2, _⟩ => l
                | ⟨_ + 3, hbad⟩ => by omega) := by
        rw [hT]
      rw [h_idx_eq, bracket_tensor_apply₃]
      -- Goal: `∑ r, V i₁ r * U i₀ r * Z r l = ∑ r, U _ r * V _ r * Zᵀ _ r`.
      refine Finset.sum_congr rfl fun r _ => ?_
      show Vhat i₁ r * Uhat i₀ r * Z r l = _ * _ * _
      rw [Matrix.transpose_apply]
      ring
  refine ⟨?_, fun Z hZ => (h_iff Z).mp hZ⟩
  -- Consistency: use the hypothesis `hX` to extract a witness `What`, take `Z = Whatᵀ`.
  obtain ⟨What, hWhat⟩ := hX
  refine ⟨Whatᵀ, ?_⟩
  -- Apply the iff in the reverse direction: need `⟦U, V, (Whatᵀ)ᵀ⟧.tensor = X`.
  apply (h_iff Whatᵀ).mpr
  rw [Matrix.transpose_transpose]
  exact hWhat

end Tensor
