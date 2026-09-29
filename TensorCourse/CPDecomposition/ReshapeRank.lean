/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import TensorCourse.Tensor.Reshape
import Mathlib.Logic.Equiv.Basic
import Mathlib.Data.Nat.Lattice

/-!
# Fiberwise Reindexing Preserves CP Rank

This file proves that reindexing a tensor by **per-mode equivalences** preserves
its CP rank. Specifically, given equivalences `e d : Fin (dims₁ d) ≃ Fin (dims₂ d)`
for each mode `d`, the induced reindexing via `Equiv.piCongrRight e` does not
change CP rank.

The key construction: given a CP decomposition `T idx = ∑_r ∏_d f(r,d,idx_d)`,
define new factors `g(r,d,i) = f(r,d,(e d).symm i)`. Then
`∑_r ∏_d g(r,d,idx₂_d) = ∑_r ∏_d f(r,d,(e d).symm (idx₂_d)) = T((piCongrRight e).symm idx₂)`
which equals `reindex (piCongrRight e) T idx₂`. Since the number of summands is
preserved, `rank(reindexed) ≤ rank(original)`. Applying the same argument with
`fun d => (e d).symm` gives the reverse inequality and hence equality.

**Important:** The general statement "reindexing by an arbitrary equivalence between
multi-index sets preserves CP rank" is FALSE. For example, reshaping a length-4
vector (order-1 tensor, CP rank 1) into a 2x2 identity matrix (order-2 tensor,
CP rank 2) increases rank. The result here is restricted to fiberwise (per-mode)
reindexing, which relabels indices within each mode independently without mixing
modes. Mode-grouping and mode-splitting will be handled separately.

## Main results

* `CPDecomp.reindexPiCongrRight` -- reindex a CP decomposition by per-mode equivalences
* `CPDecomp.reindexPiCongrRight_tensor` -- the reindexed decomposition reconstructs
  the reindexed tensor
* `cpRank_reindex_piCongrRight_le` -- `cpRank (reindex (piCongrRight e) T) ≤ cpRank T`
* `cpRank_reindex_piCongrRight_eq` -- `cpRank (reindex (piCongrRight e) T) = cpRank T`
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims₁ dims₂ : Fin D → ℕ}

/-- Reindex a CP decomposition by per-mode equivalences.
    Given equivalences `e d : Fin (dims₁ d) ≃ Fin (dims₂ d)` for each mode `d`,
    construct a new CP decomposition with `g(r,d,i) = f(r,d,(e d).symm i)`.
    This has the same number of rank-one terms as the original. -/
def CPDecomp.reindexPiCongrRight
    (e : (d : Fin D) → Fin (dims₁ d) ≃ Fin (dims₂ d))
    (cp : CPDecomp 𝕜 D dims₁ R) : CPDecomp 𝕜 D dims₂ R where
  factors r d i := cp.factors r d ((e d).symm i)

/-- The reindexed CP decomposition reconstructs the reindexed tensor.
    `(cp.reindexPiCongrRight e).tensor = reindex (piCongrRight e) cp.tensor`. -/
theorem CPDecomp.reindexPiCongrRight_tensor
    (e : (d : Fin D) → Fin (dims₁ d) ≃ Fin (dims₂ d))
    (cp : CPDecomp 𝕜 D dims₁ R) :
    (cp.reindexPiCongrRight e).tensor =
      Tensor.reindex (Equiv.piCongrRight e) cp.tensor := by
  ext idx₂
  simp [CPDecomp.tensor, reindex_apply, CPDecomp.reindexPiCongrRight]

/-- Reindexing by per-mode equivalences does not increase CP rank.

    Given equivalences `e d : Fin (dims₁ d) ≃ Fin (dims₂ d)` for each mode `d`,
    the induced reindexing via `Equiv.piCongrRight e` does not increase CP rank.

    Proof: For each CP decomposition of `T` with `R` terms, we construct a CP
    decomposition of `reindex (piCongrRight e) T` with the same `R` terms via
    `CPDecomp.reindexPiCongrRight`. This shows the set of achievable ranks for
    the reindexed tensor contains all achievable ranks for `T`, so its infimum
    is at most that of `T`. -/
theorem cpRank_reindex_piCongrRight_le
    (e : (d : Fin D) → Fin (dims₁ d) ≃ Fin (dims₂ d))
    (T : Tensor 𝕜 D dims₁) :
    cpRank (Tensor.reindex (Equiv.piCongrRight e) T) ≤ cpRank T := by
  unfold cpRank
  set S₁ := { R : ℕ | ∃ cp : CPDecomp 𝕜 D dims₁ R, cp.tensor = T }
  set S₂ := { R : ℕ | ∃ cp : CPDecomp 𝕜 D dims₂ R,
    cp.tensor = Tensor.reindex (Equiv.piCongrRight e) T }
  -- Every achievable rank for T is achievable for the reindexed tensor
  have hsub : S₁ ⊆ S₂ := fun R ⟨cp, hcp⟩ =>
    ⟨cp.reindexPiCongrRight e, by rw [CPDecomp.reindexPiCongrRight_tensor, hcp]⟩
  by_cases h : S₁.Nonempty
  · -- S₁ nonempty: sInf S₁ ∈ S₁ ⊆ S₂, so sInf S₂ ≤ sInf S₁
    exact Nat.sInf_le (hsub (Nat.sInf_mem h))
  · -- S₁ empty: sInf S₁ = 0; S₂ must also be empty (any decomp of the
    -- reindexed tensor can be reindexed back), so sInf S₂ = 0
    have h₁ : S₁ = ∅ := Set.not_nonempty_iff_eq_empty.mp h
    have h₂ : S₂ = ∅ := by
      by_contra h₂ne
      rw [← Ne, ← Set.nonempty_iff_ne_empty] at h₂ne
      obtain ⟨R, cp₂, hcp₂⟩ := h₂ne
      have : R ∈ S₁ := ⟨cp₂.reindexPiCongrRight (fun d => (e d).symm), by
        rw [CPDecomp.reindexPiCongrRight_tensor, hcp₂]
        ext idx
        simp only [reindex_apply]
        congr 1; ext d
        simp [Equiv.piCongrRight_symm_apply, Pi.map]⟩
      rw [h₁] at this; exact this
    rw [h₁, h₂]

/-- Reindexing by per-mode equivalences preserves CP rank exactly.

    This follows from applying `cpRank_reindex_piCongrRight_le` in both directions:
    once with `e` and once with `fun d => (e d).symm`. -/
theorem cpRank_reindex_piCongrRight_eq
    (e : (d : Fin D) → Fin (dims₁ d) ≃ Fin (dims₂ d))
    (T : Tensor 𝕜 D dims₁) :
    cpRank (Tensor.reindex (Equiv.piCongrRight e) T) = cpRank T := by
  apply le_antisymm
  · exact cpRank_reindex_piCongrRight_le e T
  · -- Reindex the reindexed tensor back by (fun d => (e d).symm), recovering T
    calc cpRank T
        = cpRank (Tensor.reindex (Equiv.piCongrRight (fun d => (e d).symm))
            (Tensor.reindex (Equiv.piCongrRight e) T)) := by
          congr 1; ext idx
          simp only [reindex_apply]
          congr 1; ext d
          simp [Equiv.piCongrRight_symm_apply, Pi.map]
      _ ≤ cpRank (Tensor.reindex (Equiv.piCongrRight e) T) :=
          cpRank_reindex_piCongrRight_le (fun d => (e d).symm) _

/-! ## Mode-Fusing Reindexing and CP Rank

This section proves that **mode-fusing** reindexing does not increase CP rank.
Mode-fusing is more general than the fiberwise reindexing above: it allows
merging (fusing) multiple source modes into a single target mode.

A mode-fusing reindex is specified by:
- `σ : Fin D₁ → Fin D₂` assigning each source mode to a target mode group
- `fe d₂ : (∀ d₁ : { d₁ // σ d₁ = d₂ }, Fin (dims₁ d₁)) ≃ Fin (dims₂ d₂)` — for
  each target mode, an equivalence from the product of source dims in the fiber
  to the target dim.

**Note:** The fiberwise case (`cpRank_reindex_piCongrRight_le`) is a special case
of mode-fusing where `σ = id` and each fiber is a singleton.
-/

section ModeFusing

variable {D₁ D₂ : ℕ}
variable {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}

/-- The overall multi-index equivalence induced by a mode-fusing map `σ` and
    per-fiber equivalences `fe`. Forward map: given a source multi-index `idx₁`,
    for each target mode `d₂`, restrict `idx₁` to the fiber `σ⁻¹(d₂)` and
    apply the fiber equivalence. -/
def fuseEquiv (σ : Fin D₁ → Fin D₂)
    (fe : (d₂ : Fin D₂) → ((d₁ : { d₁ // σ d₁ = d₂ }) → Fin (dims₁ d₁)) ≃ Fin (dims₂ d₂)) :
    ((d₁ : Fin D₁) → Fin (dims₁ d₁)) ≃ ((d₂ : Fin D₂) → Fin (dims₂ d₂)) where
  toFun idx₁ d₂ := fe d₂ (fun ⟨d₁, _⟩ => idx₁ d₁)
  invFun idx₂ d₁ := (fe (σ d₁)).symm (idx₂ (σ d₁)) ⟨d₁, rfl⟩
  left_inv idx₁ := by
    ext d₁
    -- (fe (σ d₁)).symm (fe (σ d₁) (fun ⟨d₁', _⟩ => idx₁ d₁')) ⟨d₁, rfl⟩
    -- = idx₁ d₁, because (fe _).symm ∘ (fe _) = id
    simp only [Equiv.symm_apply_apply]
  right_inv idx₂ := by
    funext d₂
    -- Beta-reduce the toFun/invFun composition
    dsimp only []
    -- Goal: (fe d₂) (fun ⟨d₁, _⟩ => (fe (σ d₁)).symm (idx₂ (σ d₁)) ⟨d₁, rfl⟩) = idx₂ d₂
    -- Suffices to show the lambda argument equals (fe d₂).symm (idx₂ d₂)
    suffices h : (fun (x : { d₁ // σ d₁ = d₂ }) =>
        (fe (σ x.val)).symm (idx₂ (σ x.val)) ⟨x.val, rfl⟩) =
        (fe d₂).symm (idx₂ d₂) by
      rw [h]; exact (fe d₂).apply_symm_apply (idx₂ d₂)
    funext ⟨d₁, h⟩
    subst h
    rfl

/-- Fuse a CP decomposition along `σ` and `fe`.
    For each rank-one term `r` and target mode `d₂`, the new factor vector at index `i` is
    the product of the original factors over all source modes in the fiber `σ⁻¹(d₂)`. -/
def CPDecomp.fuse (σ : Fin D₁ → Fin D₂)
    (fe : (d₂ : Fin D₂) → ((d₁ : { d₁ // σ d₁ = d₂ }) → Fin (dims₁ d₁)) ≃ Fin (dims₂ d₂))
    (cp : CPDecomp 𝕜 D₁ dims₁ R) : CPDecomp 𝕜 D₂ dims₂ R where
  factors r d₂ i := ∏ d₁ : { d₁ // σ d₁ = d₂ }, cp.factors r d₁.val ((fe d₂).symm i d₁)

/-- The fused CP decomposition reconstructs the fused tensor:
    `(cp.fuse σ fe).tensor = Tensor.reindex (fuseEquiv σ fe) cp.tensor`.

    The key algebraic step: for each term `r` and target multi-index `idx₂`,
    `∏ d₂, (∏ d₁ ∈ σ⁻¹(d₂), f(r, d₁, ...)) = ∏ d₁, f(r, d₁, ...)`,
    which is the product-over-fibers identity (`Fintype.prod_fiberwise`). -/
theorem CPDecomp.fuse_tensor (σ : Fin D₁ → Fin D₂)
    (fe : (d₂ : Fin D₂) → ((d₁ : { d₁ // σ d₁ = d₂ }) → Fin (dims₁ d₁)) ≃ Fin (dims₂ d₂))
    (cp : CPDecomp 𝕜 D₁ dims₁ R) :
    (cp.fuse σ fe).tensor = Tensor.reindex (fuseEquiv σ fe) cp.tensor := by
  ext idx₂
  simp only [CPDecomp.tensor, Tensor.reindex_apply, CPDecomp.fuse]
  congr 1; ext r
  -- LHS: ∏ d₂, ∏ d₁ : { d₁ // σ d₁ = d₂ }, cp.factors r d₁.val ((fe d₂).symm (idx₂ d₂) d₁)
  -- RHS: ∏ d₁, cp.factors r d₁ ((fuseEquiv σ fe).symm idx₂ d₁)
  -- Reduce (fuseEquiv σ fe).symm to the invFun definition
  simp only [fuseEquiv, Equiv.coe_fn_symm_mk]
  -- Goal: ∏ d₂, ∏ d₁ : { d₁ // σ d₁ = d₂ }, cp.factors r d₁.val ((fe d₂).symm (idx₂ d₂) d₁)
  --     = ∏ d₁, cp.factors r d₁ ((fe (σ d₁)).symm (idx₂ (σ d₁)) ⟨d₁, rfl⟩)
  -- Apply the fiberwise product identity (reversed)
  symm
  rw [← Fintype.prod_fiberwise σ
    (fun d₁ => cp.factors r d₁ ((fe (σ d₁)).symm (idx₂ (σ d₁)) ⟨d₁, rfl⟩))]
  symm
  congr 1; ext d₂; congr 1; ext ⟨d₁, h⟩
  subst h; rfl

/-- Mode-fusing reindexing does not increase CP rank.

    Given a mode-fusing map `σ` and per-fiber equivalences `fe`, the fused tensor
    has CP rank at most that of the original tensor. The proof constructs a CP
    decomposition of the fused tensor with the same number of terms via `CPDecomp.fuse`.

    **Note on the order hypothesis:** Unlike the fiberwise case where we can
    always "un-reindex" a CP decomposition (since per-mode relabeling preserves
    rank-one structure), mode-fusing can increase rank, so we cannot construct a
    CP decomposition of `T` from one of `reindex (fuseEquiv σ fe) T`. We need `T`
    to admit *some* CP decomposition to avoid the `sInf ∅ = 0` degeneracy for `ℕ`;
    for tensors of order ≥ 1 this is automatic, so we take `0 < D₁` and discharge
    the existence via `exists_cpDecomp`. -/
theorem cpRank_reindex_fuse_le (σ : Fin D₁ → Fin D₂)
    (fe : (d₂ : Fin D₂) → ((d₁ : { d₁ // σ d₁ = d₂ }) → Fin (dims₁ d₁)) ≃ Fin (dims₂ d₂))
    (T : Tensor 𝕜 D₁ dims₁)
    (hD₁ : 0 < D₁) :
    cpRank (Tensor.reindex (fuseEquiv σ fe) T) ≤ cpRank T := by
  -- Positive order ⇒ T admits a CP decomposition.
  have hT := exists_cpDecomp hD₁ T
  -- Extract an optimal decomposition at rank cpRank T
  have hne : Set.Nonempty { R : ℕ | ∃ cp : CPDecomp 𝕜 D₁ dims₁ R, cp.tensor = T } :=
    let ⟨R, cp, h⟩ := hT; ⟨R, cp, h⟩
  obtain ⟨cp, hcp⟩ := Nat.sInf_mem hne
  -- Fuse this decomposition to get one for the reindexed tensor
  exact cpRank_le_of_cpDecomp (cp.fuse σ fe)
    (by rw [CPDecomp.fuse_tensor]; exact congrArg _ hcp)

end ModeFusing

end Tensor
