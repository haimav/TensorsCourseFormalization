/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import TensorCourse.Tensor.Vectorize
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# CP Decomposition

This file defines the CP (Canonical Polyadic) decomposition of tensors and
the CP rank.

## Main definitions

* `CPDecomp` — a CP decomposition with `R` rank-one terms for an order-`D` tensor
* `CPDecomp.tensor` — reconstruct the tensor from a CP decomposition
* `cpRank` — the CP rank of a tensor (minimum `R` for which a CP decomposition exists)

## Main results

* `cpRank_le_of_cpDecomp` — if a CP decomposition of rank `R` exists, then `cpRank T ≤ R`
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- A CP decomposition of an order-`D` tensor with `R` rank-one terms.
    Each term is specified by giving a vector along each mode. -/
structure CPDecomp (𝕜 : Type*) [RCLike 𝕜] (D : ℕ) (dims : Fin D → ℕ) (R : ℕ) where
  /-- `factors r d` is the vector along mode `d` for the `r`-th rank-one term. -/
  factors : Fin R → (d : Fin D) → Fin (dims d) → 𝕜

/-- Reconstruct a tensor from a CP decomposition:
    `T(idx) = ∑ᵣ ∏_d factors(r, d, idx_d)` -/
def CPDecomp.tensor (cp : CPDecomp 𝕜 D dims R) : Tensor 𝕜 D dims :=
  fun idx => ∑ r : Fin R, ∏ d : Fin D, cp.factors r d (idx d)

/-- The CP rank of a tensor is the smallest `R` for which a rank-`R`
    CP decomposition exists. -/
def cpRank (T : Tensor 𝕜 D dims) : ℕ :=
  sInf { R : ℕ | ∃ cp : CPDecomp 𝕜 D dims R, cp.tensor = T }

/-- If a CP decomposition of rank `R` reconstructs `T`, then `cpRank T ≤ R`. -/
theorem cpRank_le_of_cpDecomp {R : ℕ} (cp : CPDecomp 𝕜 D dims R)
    (h : cp.tensor = T) : cpRank T ≤ R :=
  Nat.sInf_le ⟨cp, h⟩

/-- Forward direction: if `cpRank T ≤ R`, then there exists a CP decomposition
    of rank at most `R`. We prove the converse direction which is more useful. -/
theorem cpRank_le_iff_exists {R : ℕ} {T : Tensor 𝕜 D dims} :
    (∃ cp : CPDecomp 𝕜 D dims R, cp.tensor = T) → cpRank T ≤ R :=
  fun ⟨cp, h⟩ => cpRank_le_of_cpDecomp cp h

/-- The zero tensor has CP rank 0. -/
theorem cpRank_zero : cpRank (0 : Tensor 𝕜 D dims) = 0 := by
  apply le_antisymm
  · apply cpRank_le_of_cpDecomp (⟨Fin.elim0⟩ : CPDecomp 𝕜 D dims 0)
    ext idx
    simp [CPDecomp.tensor]
  · exact Nat.zero_le _

/-- Every tensor of positive order admits *some* CP decomposition: the full
    decomposition with one rank-one term per multi-index (`∏ d, dims d`
    terms), the scalar `X idx` absorbed into the mode-0 factor. This makes
    the CP-rank infimum set nonempty (used wherever a lower bound on
    `cpRank` is taken). -/
theorem exists_cpDecomp (hpos : 0 < D) (X : Tensor 𝕜 D dims) :
    ∃ (R : ℕ) (cp : CPDecomp 𝕜 D dims R), cp.tensor = X := by
  refine ⟨∏ d, dims d,
    { factors := fun r d i =>
        if (multiIndexEquiv dims).symm r d = i then
          (if d = ⟨0, hpos⟩ then X ((multiIndexEquiv dims).symm r) else 1)
        else 0 }, ?_⟩
  ext idx
  simp only [CPDecomp.tensor]
  -- Reindex the rank sum `r : Fin (∏ dims)` to a sum over multi-indices `m`.
  rw [← Equiv.sum_comp (multiIndexEquiv dims)
    (fun r => ∏ d : Fin D, if (multiIndexEquiv dims).symm r d = idx d then
      (if d = ⟨0, hpos⟩ then X ((multiIndexEquiv dims).symm r) else 1) else 0)]
  simp only [Equiv.symm_apply_apply]
  -- Only the term `m = idx` survives (the product vanishes off the diagonal).
  rw [Finset.sum_eq_single idx]
  · -- At `m = idx`: every delta is `1`, leaving the mode-0 scalar `X idx`.
    rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
          (fun d => if idx d = idx d then
            (if d = ⟨0, hpos⟩ then X idx else 1) else 0) (Finset.mem_univ ⟨0, hpos⟩)]
    rw [if_pos rfl, if_pos rfl]
    rw [Finset.prod_congr rfl (fun d hd => by
      rw [if_pos rfl, if_neg (Finset.ne_of_mem_erase hd)])]
    rw [Finset.prod_const_one, mul_one]
  · -- At `m ≠ idx`: some coordinate differs, killing the product.
    intro m _ hm
    have : ∃ d, m d ≠ idx d := by
      by_contra hc
      push Not at hc
      exact hm (funext hc)
    obtain ⟨d, hd⟩ := this
    refine Finset.prod_eq_zero (Finset.mem_univ d) ?_
    rw [if_neg hd]
  · intro h; exact absurd (Finset.mem_univ idx) h

end Tensor
