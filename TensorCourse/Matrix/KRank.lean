/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.LinearIndependent.Defs
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# k-rank of a Matrix (Kruskal 1977)

The k-rank of a matrix `A`, denoted `k_A`, is the largest `k` such that
*every* `k`-tuple of distinct columns of `A` is linearly independent.
This is a stricter notion than rank: `k_A ≤ rank A` always, with
equality when the columns are in ``general position''.

The k-rank is the key quantity in **Kruskal's theorem** on CP
uniqueness (see `CPDecomposition/FactorEquiv.lean`).

## Main definitions

* `Matrix.kRank A` — the k-rank of `A`.

## Main results (bridging lemmas for Kruskal's hypothesis)

* `Matrix.kRank_le_ncols` — `k_A ≤ r` (number of columns).
* `Matrix.kRank_eq_of_lin_indep` — if `A`'s columns are linearly
  independent, then `k_A = r`.
* `Matrix.kRank_pairwise_lin_indep_ge_two` — if every pair of distinct
  columns of `A` is linearly independent (no column is a scalar
  multiple of another, in either direction), then `k_A ≥ 2`.

## References

* Kruskal, "Three-way arrays: rank and uniqueness of trilinear
  decompositions", Linear Algebra Appl. 18 (1977), 95-138.
* Bernstein, *Matrix Mathematics* (2nd ed.), Definition 16.10
  (analogous concept).
-/

noncomputable section

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The **k-rank** of a matrix: the largest `k` such that every set of
    `k` distinct columns is linearly independent. -/
def kRank {M r : ℕ} (A : Matrix (Fin M) (Fin r) 𝕜) : ℕ :=
  sSup {k : ℕ | k ≤ r ∧
    ∀ S : Finset (Fin r), S.card = k →
      LinearIndependent 𝕜 (fun s : S => fun i : Fin M => A i s.val)}

/-- The defining set of `kRank` is bounded above by `r`. -/
theorem kRank_set_bddAbove {M r : ℕ} (A : Matrix (Fin M) (Fin r) 𝕜) :
    BddAbove {k : ℕ | k ≤ r ∧
      ∀ S : Finset (Fin r), S.card = k →
        LinearIndependent 𝕜 (fun s : S => fun i : Fin M => A i s.val)} :=
  ⟨r, fun _ hk => hk.1⟩

/-- The k-rank is bounded above by `r` (the number of columns). -/
theorem kRank_le_ncols {M r : ℕ} (A : Matrix (Fin M) (Fin r) 𝕜) :
    A.kRank ≤ r := by
  unfold kRank
  by_cases hne : ({k : ℕ | k ≤ r ∧
      ∀ S : Finset (Fin r), S.card = k →
        LinearIndependent 𝕜 (fun s : S => fun i : Fin M => A i s.val)}).Nonempty
  · exact csSup_le hne (fun _ hk => hk.1)
  · have hempty : {k : ℕ | k ≤ r ∧
        ∀ S : Finset (Fin r), S.card = k →
          LinearIndependent 𝕜 (fun s : S => fun i : Fin M => A i s.val)} = ∅ :=
      Set.not_nonempty_iff_eq_empty.mp hne
    rw [hempty]; simp

/-- If `A`'s columns are linearly independent, then `A.kRank = r`. -/
theorem kRank_eq_of_lin_indep {M r : ℕ} (A : Matrix (Fin M) (Fin r) 𝕜)
    (h : LinearIndependent 𝕜 (fun s : Fin r => fun i : Fin M => A i s)) :
    A.kRank = r := by
  apply le_antisymm (kRank_le_ncols A)
  -- Show `r ∈ {k : ℕ | k ≤ r ∧ ...}`, then apply `le_csSup`.
  apply le_csSup (kRank_set_bddAbove A)
  refine ⟨le_refl r, ?_⟩
  intro S _
  -- Any subfamily of an LI family is LI.
  have hinj : Function.Injective (fun s : S => s.val) := Subtype.val_injective
  exact h.comp _ hinj

/-- If every pair of distinct columns of `A` is linearly independent
    (i.e., for any two distinct indices `s ≠ t`, neither column is a
    scalar multiple of the other) and `r ≥ 2`, then `A.kRank ≥ 2`.

    The hypothesis is stated symmetrically: for both `(s, t)` and
    `(t, s)`. The simdiag-uniqueness invocation supplies an
    `∀ s t : Fin r, s ≠ t → ¬ ∃ c, ...` hypothesis whose `(s, t)` and
    `(t, s)` instances together yield this. -/
theorem kRank_pairwise_lin_indep_ge_two {M r : ℕ}
    (A : Matrix (Fin M) (Fin r) 𝕜) (hr : 2 ≤ r)
    (h : ∀ s t : Fin r, s ≠ t →
      ¬ ∃ c : 𝕜, (fun i : Fin M => A i s) = c • (fun i : Fin M => A i t)) :
    2 ≤ A.kRank := by
  apply le_csSup (kRank_set_bddAbove A)
  refine ⟨hr, ?_⟩
  intro S hS
  rw [Fintype.linearIndependent_iff]
  intro α hα
  -- Obtain the two distinct elements of S.
  obtain ⟨s, t, hst, hSeq⟩ := Finset.card_eq_two.mp hS
  have hs_mem : s ∈ S := by rw [hSeq]; exact Finset.mem_insert_self _ _
  have ht_mem : t ∈ S := by
    rw [hSeq]; exact Finset.mem_insert_of_mem (Finset.mem_singleton_self _)
  let s' : S := ⟨s, hs_mem⟩
  let t' : S := ⟨t, ht_mem⟩
  have hst' : s' ≠ t' := by
    intro habs; exact hst (Subtype.ext_iff.mp habs)
  -- Pointwise: for every i, α s' * A i s + α t' * A i t = 0.
  have hsum : ∀ i : Fin M, α s' * A i s + α t' * A i t = 0 := by
    intro i
    have hi := congrFun hα i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at hi
    have huniv : (Finset.univ : Finset S) = {s', t'} := by
      apply Finset.eq_of_subset_of_card_le
      · intro x _
        simp only [Finset.mem_insert, Finset.mem_singleton]
        have hxval : (x : Fin r) ∈ ({s, t} : Finset (Fin r)) := hSeq ▸ x.2
        rw [Finset.mem_insert, Finset.mem_singleton] at hxval
        rcases hxval with hxs | hxt
        · left; exact Subtype.ext hxs
        · right; exact Subtype.ext hxt
      · have hcardSub : (Finset.univ : Finset S).card = 2 := by
          rw [Finset.card_univ]; simp [hS]
        have hcardPair : ({s', t'} : Finset S).card = 2 :=
          Finset.card_pair hst'
        rw [hcardSub, hcardPair]
    rw [huniv] at hi
    rw [Finset.sum_insert (by
        simp only [Finset.mem_singleton]; exact hst'),
        Finset.sum_singleton] at hi
    exact hi
  -- Conclude α s' = 0 and α t' = 0.
  -- Step 1: α s' = 0. Suppose not; then column s is a scalar multiple of column t.
  have hαs : α s' = 0 := by
    by_contra hαs_ne
    apply h s t hst
    refine ⟨-(α t' / α s'), ?_⟩
    funext i
    show A i s = (-(α t' / α s') • fun i' : Fin M => A i' t) i
    simp only [Pi.smul_apply, smul_eq_mul]
    have hi := hsum i
    -- A i s = (-α t' * A i t) / α s' = -(α t' / α s') * A i t.
    have hAis : A i s = -(α t' / α s') * A i t := by
      have hcalc : α s' * A i s = -(α t' * A i t) := by linear_combination hi
      have hAis' : A i s = -(α t' * A i t) / α s' := by
        rw [eq_div_iff hαs_ne]; linear_combination hcalc
      rw [hAis']; field_simp
    exact hAis
  -- Step 2: α t' = 0. Symmetric — use h t s (hst.symm).
  have hαt : α t' = 0 := by
    by_contra hαt_ne
    apply h t s hst.symm
    refine ⟨-(α s' / α t'), ?_⟩
    funext i
    show A i t = (-(α s' / α t') • fun i' : Fin M => A i' s) i
    simp only [Pi.smul_apply, smul_eq_mul]
    have hi := hsum i
    have hAit : A i t = -(α s' / α t') * A i s := by
      have hcalc : α t' * A i t = -(α s' * A i s) := by linear_combination hi
      have hAit' : A i t = -(α s' * A i s) / α t' := by
        rw [eq_div_iff hαt_ne]; linear_combination hcalc
      rw [hAit']; field_simp
    exact hAit
  -- Show the goal: α x = 0 for x : S. Either x = s' or x = t'.
  intro x
  have huniv : (Finset.univ : Finset S) = {s', t'} := by
    apply Finset.eq_of_subset_of_card_le
    · intro y _
      simp only [Finset.mem_insert, Finset.mem_singleton]
      have hyval : (y : Fin r) ∈ ({s, t} : Finset (Fin r)) := hSeq ▸ y.2
      rw [Finset.mem_insert, Finset.mem_singleton] at hyval
      rcases hyval with hys | hyt
      · left; exact Subtype.ext hys
      · right; exact Subtype.ext hyt
    · have hcardSub : (Finset.univ : Finset S).card = 2 := by
        rw [Finset.card_univ]; simp [hS]
      have hcardPair : ({s', t'} : Finset S).card = 2 :=
        Finset.card_pair hst'
      rw [hcardSub, hcardPair]
  have hxmem : x ∈ ({s', t'} : Finset S) := by
    rw [← huniv]; exact Finset.mem_univ x
  rw [Finset.mem_insert, Finset.mem_singleton] at hxmem
  rcases hxmem with hxs | hxt
  · rw [hxs]; exact hαs
  · rw [hxt]; exact hαt

end Matrix

end
