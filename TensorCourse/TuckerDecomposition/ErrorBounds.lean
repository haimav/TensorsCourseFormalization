/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TuckerDecomposition.Projection
import TensorCourse.TuckerDecomposition.NRank
import TensorCourse.Matrix.SVDAxioms
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Tucker Decomposition: error bounds and HOSVD quasi-optimality

This file proves the error analysis of the (truncated) higher-order SVD,
after Vannieuwenhoven, Vandebril, Meerbergen (2012):

* `Tensor.tucker_error_bounds` (**`thm:Tucker-error-base`**, axiom-free) —
  for orthonormal-column factors `U` and `Y` the mode-projection chain
  applied in the order of a permutation `σ`,
  `maxₙ ‖Pₙ^⊥ X‖² ≤ ‖X − Y‖² ≤ ∑ₙ ‖Pₙ^⊥ X‖²`. The proof is a general-`D`
  telescoping + Pythagoras argument.
* `Tensor.IsTruncatedHOSVD` — the predicate "`Y` is a truncated HOSVD of
  `X` of target multilinear rank `R`", packaging the orthonormal factors,
  the projection chain, and the fact that each factor attains the best
  rank-`R n` Frobenius residual (the Eckart–Young content, via axiom B2).
* `Tensor.hosvd_error_singular_values` (**`cor:HOSVD-error-bounds`**) — the
  truncated-HOSVD error is bounded by the trailing singular values
  (best-rank residuals).
* `Tensor.hosvd_quasi_optimal` (**`thm:HOSVD-error`**) — the truncated
  HOSVD is quasi-optimal: `‖X − Y‖ ≤ √D · ‖X − Z‖` for every rank-`R`
  tensor `Z`.

## The projection chain

The mode-projection chain is encoded as a fold over a *list* of modes,
`Tensor.chainProj U L Y = Pₙ₀ (Pₙ₁ (⋯ (P_{last} Y)))` for `L = [n₀, n₁, …]`.
For a permutation `σ`, the order-`σ` chain `Y` is `chainProj U L X` with
`L = (List.finRange D).map σ`; since the per-mode projections on distinct
modes commute (`projMode_projMode_comm`), the resulting `Y` is independent
of the order, so any permutation gives the lecture's truncated-HOSVD
approximation.

The telescoping/orthogonality argument is run over an arbitrary `Nodup`
list of modes (which is all that the proof needs), then specialised to the
permutation list. Two facts drive it:

* the chain `chainProj U L` is itself an **orthogonal projection**
  (idempotent + self-adjoint), proved by list induction from the
  single-mode facts; hence the head telescoping summand
  `Pₙ₀^⊥ (chainProj U L X)` is orthogonal to the tail residual
  `X − chainProj U L X` (`head_summand_orthogonal`);
* each projection is norm-non-increasing, giving the per-summand bound.

## Axioms

Only `hosvd_error_singular_values` and `hosvd_quasi_optimal` depend on the
Eckart–Young axiom B2 (`Matrix.exists_best_rank_approx`). The telescoping
theorem `tucker_error_bounds` is **axiom-free** (standard axioms only).

## Transpose vs. conjugate transpose

These are adjoint-flavoured results (orthogonal projections, the inner
product `⟪·,·⟫ₜ` conjugates its first argument), so factor matrices carry
`ᴴ`; over `ℝ`, `ᴴ = ᵀ`, recovering the lecture.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ} {R : Fin D → ℕ}

/-! ### Linearity and norm bounds of the mode projections

The projection `projMode n U` is linear (it is a contraction against a
fixed matrix); we record the additive/subtractive forms, the complementary
norm bound, and that `Pₙ^⊥` does not increase the norm. -/

/-- `projMode` is additive: `Pₙ (Y + Z) = Pₙ Y + Pₙ Z`. -/
theorem projMode_add (n : Fin D) (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (Y Z : Tensor 𝕜 D dims) :
    projMode n U (Y + Z) = projMode n U Y + projMode n U Z := by
  funext idx
  simp only [projMode_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun k _ => by rw [mul_add])

/-- `projMode` respects subtraction: `Pₙ (Y − Z) = Pₙ Y − Pₙ Z`. -/
theorem projMode_sub (n : Fin D) (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (Y Z : Tensor 𝕜 D dims) :
    projMode n U (Y - Z) = projMode n U Y - projMode n U Z := by
  funext idx
  simp only [projMode_apply, Pi.sub_apply]
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl (fun k _ => by rw [mul_sub])

/-- **The complementary projection does not increase the Frobenius norm:**
    `‖Pₙ^⊥ Y‖² ≤ ‖Y‖²`. Immediate from Pythagoras and nonnegativity of
    `‖Pₙ Y‖²`. -/
theorem frobNormSq_projModePerp_le (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (hU : Matrix.HasOrthonormalCols U)
    (Y : Tensor 𝕜 D dims) :
    ‖projModePerp n U Y‖_F² ≤ ‖Y‖_F² := by
  rw [frobNormSq_projMode_add_projModePerp n U hU Y]
  have := frobNormSq_nonneg (projMode n U Y)
  linarith

/-! ### The mode-projection chain -/

/-- **The mode-projection chain.** `chainProj U L Y` applies the mode
    projections `Pₙ` for `n` in the list `L`, with the head applied
    outermost: `chainProj U [n₀, n₁, …] Y = Pₙ₀ (Pₙ₁ (⋯ Y))`. -/
def chainProj (U : (n : Fin D) → Matrix (Fin (dims n)) (Fin (R n)) 𝕜) :
    List (Fin D) → Tensor 𝕜 D dims → Tensor 𝕜 D dims
  | [], Y => Y
  | n :: L, Y => projMode n (U n) (chainProj U L Y)

@[simp] theorem chainProj_nil
    (U : (n : Fin D) → Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y : Tensor 𝕜 D dims) :
    chainProj U [] Y = Y := rfl

@[simp] theorem chainProj_cons
    (U : (n : Fin D) → Matrix (Fin (dims n)) (Fin (R n)) 𝕜)
    (n : Fin D) (L : List (Fin D)) (Y : Tensor 𝕜 D dims) :
    chainProj U (n :: L) Y = projMode n (U n) (chainProj U L Y) := rfl

variable (U : (n : Fin D) → Matrix (Fin (dims n)) (Fin (R n)) 𝕜)

/-- The chain respects subtraction. -/
theorem chainProj_sub (L : List (Fin D)) (Y Z : Tensor 𝕜 D dims) :
    chainProj U L (Y - Z) = chainProj U L Y - chainProj U L Z := by
  induction L generalizing Y Z with
  | nil => rfl
  | cons n L ih => rw [chainProj_cons, ih, projMode_sub, chainProj_cons, chainProj_cons]

/-- The chain does not increase the Frobenius norm (each step is a
    norm-non-increasing projection). -/
theorem chainProj_frobNormSq_le (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    (L : List (Fin D)) (Y : Tensor 𝕜 D dims) :
    ‖chainProj U L Y‖_F² ≤ ‖Y‖_F² := by
  induction L generalizing Y with
  | nil => rw [chainProj_nil]
  | cons n L ih => exact le_trans (frobNormSq_projMode_le n (U n) (hU n) _) (ih Y)

/-- The chain fixes `X` at any mode `n ∈ L` (the chain already contains the
    idempotent `Pₙ`). Needs `Nodup` so the leading projections commute past
    `Pₙ`. -/
theorem chainProj_fixed (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {L : List (Fin D)} (hL : L.Nodup) {n : Fin D} (hn : n ∈ L) (Y : Tensor 𝕜 D dims) :
    projMode n (U n) (chainProj U L Y) = chainProj U L Y := by
  induction L generalizing Y with
  | nil => simp at hn
  | cons m L ih =>
    rw [List.nodup_cons] at hL
    rw [chainProj_cons]
    rcases List.mem_cons.mp hn with h | h
    · subst h
      exact projMode_idempotent n (U n) (hU n) _
    · have hnm : n ≠ m := fun he => hL.1 (he ▸ h)
      rw [projMode_projMode_comm hnm (U n) (U m), ih hL.2 h Y]

/-- A single projection at a mode `n ∉ L` commutes with the whole chain. -/
theorem chainProj_projMode_comm {n : Fin D} {L : List (Fin D)} (hn : n ∉ L)
    (Un : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (Y : Tensor 𝕜 D dims) :
    chainProj U L (projMode n Un Y) = projMode n Un (chainProj U L Y) := by
  induction L generalizing Y with
  | nil => rfl
  | cons m L ih =>
    rw [List.mem_cons, not_or] at hn
    obtain ⟨hnm, hnL⟩ := hn
    rw [chainProj_cons, chainProj_cons, ih hnL Y,
      projMode_projMode_comm (Ne.symm hnm) (U m) Un]

/-- The chain is self-adjoint: `⟪chainProj U L Y, Z⟫ₜ = ⟪Y, chainProj U L Z⟫ₜ`.
    Composition of self-adjoint projections; `Nodup` lets the chain commute
    past the head projection. -/
theorem chainProj_self_adjoint {L : List (Fin D)} (hL : L.Nodup)
    (Y Z : Tensor 𝕜 D dims) :
    ⟪chainProj U L Y, Z⟫ₜ = ⟪Y, chainProj U L Z⟫ₜ := by
  induction L generalizing Y Z with
  | nil => rfl
  | cons n L ih =>
    rw [List.nodup_cons] at hL
    rw [chainProj_cons, chainProj_cons,
      projMode_self_adjoint n (U n) (chainProj U L Y) Z,
      ih hL.2 Y (projMode n (U n) Z), chainProj_projMode_comm U hL.1 (U n) Z]

/-- The chain is idempotent: `chainProj U L (chainProj U L Y) = chainProj U L Y`
    (`Nodup`). Composition of commuting idempotent projections. -/
theorem chainProj_idempotent (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {L : List (Fin D)} (hL : L.Nodup) (Y : Tensor 𝕜 D dims) :
    chainProj U L (chainProj U L Y) = chainProj U L Y := by
  induction L generalizing Y with
  | nil => rfl
  | cons n L ih =>
    rw [List.nodup_cons] at hL
    rw [chainProj_cons, chainProj_cons,
      chainProj_projMode_comm U hL.1 (U n) (chainProj U L Y), ih hL.2,
      projMode_idempotent n (U n) (hU n)]

/-- The chain is an **orthogonal projection**: it is orthogonal to its own
    complement, `⟪chainProj U L V, V − chainProj U L V⟫ₜ = 0` (`Nodup`).
    From self-adjointness and idempotence:
    `⟪P V, V − P V⟫ = ⟪V, P (V − P V)⟫ = ⟪V, P V − P V⟫ = 0`.

    *API-facing characterisation* — `tucker_error_bounds`'s proof uses the
    stronger head/tail-split `head_summand_orthogonal` directly, so this
    lemma is not on the critical path; it is kept as the natural projection
    identity to expose to users. -/
theorem chainProj_dotProd_sub_eq_zero (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {L : List (Fin D)} (hL : L.Nodup) (V : Tensor 𝕜 D dims) :
    ⟪chainProj U L V, V - chainProj U L V⟫ₜ = 0 := by
  rw [chainProj_self_adjoint U hL V (V - chainProj U L V), chainProj_sub,
    chainProj_idempotent U hU hL, sub_self, dotProd_def]
  simp

/-- The complementary projection `Pₙ^⊥` commutes with the chain when
    `n ∉ L`. -/
theorem projModePerp_chainProj_comm {n : Fin D} {L : List (Fin D)} (hn : n ∉ L)
    (Y : Tensor 𝕜 D dims) :
    projModePerp n (U n) (chainProj U L Y) = chainProj U L (projModePerp n (U n) Y) := by
  rw [projModePerp_def, projModePerp_def, chainProj_sub, chainProj_projMode_comm U hn (U n) Y]

/-! ### The telescoping orthogonality

The single nontrivial orthogonality fact behind the telescoping: the head
summand `Pₙ₀^⊥ (chainProj U L X)` (added when prepending mode `n₀`) is
orthogonal to the tail residual `X − chainProj U L X`. -/

/-- **Head summand orthogonality.** For `n₀ ∉ L` (`L` `Nodup`), the head
    telescoping summand `Pₙ₀^⊥ (chainProj U L X)` is orthogonal to the tail
    residual `X − chainProj U L X`.

    Proof: `Pₙ₀^⊥` commutes past the chain (`n₀ ∉ L`), so the summand is
    `chainProj U L V` with `V = Pₙ₀^⊥ X`. Pushing the chain across the inner
    product by self-adjointness and using idempotence,
    `⟪chainProj U L V, X − chainProj U L X⟫
       = ⟪V, chainProj U L X − chainProj U L X⟫ = 0`. -/
theorem head_summand_orthogonal (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {n₀ : Fin D} {L : List (Fin D)} (hn₀ : n₀ ∉ L) (hL : L.Nodup)
    (X : Tensor 𝕜 D dims) :
    ⟪X - chainProj U L X, projModePerp n₀ (U n₀) (chainProj U L X)⟫ₜ = 0 := by
  rw [projModePerp_chainProj_comm U hn₀ X]
  set V : Tensor 𝕜 D dims := projModePerp n₀ (U n₀) X with hV
  rw [dotProd_conj_symm (X - chainProj U L X) (chainProj U L V),
    chainProj_self_adjoint U hL V (X - chainProj U L X), chainProj_sub,
    chainProj_idempotent U hU hL, sub_self, dotProd_def]
  simp

/-! ### Error bounds for an arbitrary `Nodup` projection order -/

/-- **Upper bound (list version).** For a `Nodup` list of modes `L`,
    `‖X − chainProj U L X‖² ≤ ∑_{n ∈ L} ‖Pₙ^⊥ X‖²`.

    Telescoping induction: prepending mode `n₀`,
    `X − chainProj U (n₀::L) X = (X − chainProj U L X) + Pₙ₀^⊥ (chainProj U L X)`,
    a Pythagorean split (`head_summand_orthogonal`); the new summand is
    `≤ ‖Pₙ₀^⊥ X‖²` since `Pₙ₀^⊥ (chainProj U L X) = chainProj U L (Pₙ₀^⊥ X)`
    and the chain is norm-non-increasing. -/
theorem chainProj_error_upper (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {L : List (Fin D)} (hL : L.Nodup) (X : Tensor 𝕜 D dims) :
    ‖X - chainProj U L X‖_F² ≤ ∑ n ∈ L.toFinset, ‖projModePerp n (U n) X‖_F² := by
  induction L generalizing X with
  | nil =>
    rw [chainProj_nil, sub_self, List.toFinset_nil, Finset.sum_empty, frobNormSq_def]
    simp
  | cons n₀ L ih =>
    rw [List.nodup_cons] at hL
    obtain ⟨hn₀, hLnd⟩ := hL
    have hsplit : X - chainProj U (n₀ :: L) X =
        (X - chainProj U L X) + projModePerp n₀ (U n₀) (chainProj U L X) := by
      rw [chainProj_cons, projModePerp_def]; abel
    rw [hsplit, frobNormSq_add_of_orthogonal (head_summand_orthogonal U hU hn₀ hLnd X),
      List.toFinset_cons, Finset.sum_insert (by simp [List.mem_toFinset, hn₀])]
    have hT0 : ‖projModePerp n₀ (U n₀) (chainProj U L X)‖_F²
        ≤ ‖projModePerp n₀ (U n₀) X‖_F² := by
      rw [projModePerp_chainProj_comm U hn₀ X]
      exact chainProj_frobNormSq_le U hU L (projModePerp n₀ (U n₀) X)
    have hrest := ih hLnd X
    linarith

/-- **Lower bound (list version).** For a `Nodup` list `L` and any mode
    `n ∈ L`, `‖Pₙ^⊥ X‖² ≤ ‖X − chainProj U L X‖²`.

    Direct (no telescoping): `chainProj U L X` is fixed by `Pₙ`
    (`chainProj_fixed`), so `Pₙ^⊥ (chainProj U L X) = 0`, hence
    `Pₙ^⊥ (X − chainProj U L X) = Pₙ^⊥ X`; the complementary projection does
    not increase the norm. -/
theorem chainProj_error_lower (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    {L : List (Fin D)} (hL : L.Nodup) {n : Fin D} (hn : n ∈ L) (X : Tensor 𝕜 D dims) :
    ‖projModePerp n (U n) X‖_F² ≤ ‖X - chainProj U L X‖_F² := by
  have hfix : projMode n (U n) (chainProj U L X) = chainProj U L X :=
    chainProj_fixed U hU hL hn X
  have heq : projModePerp n (U n) (X - chainProj U L X) = projModePerp n (U n) X := by
    rw [projModePerp_def, projModePerp_def, projMode_sub, hfix]; abel
  calc ‖projModePerp n (U n) X‖_F²
      = ‖projModePerp n (U n) (X - chainProj U L X)‖_F² := by rw [heq]
    _ ≤ ‖X - chainProj U L X‖_F² := frobNormSq_projModePerp_le n (U n) (hU n) _

/-! ### The permutation list

For a permutation `σ`, the order-`σ` projection chain uses the list
`(List.finRange D).map σ`, which is `Nodup` and lists every mode exactly
once. -/

/-- `(List.finRange D).map σ` is `Nodup`. -/
theorem permList_nodup (σ : Equiv.Perm (Fin D)) : ((List.finRange D).map σ).Nodup := by
  rw [List.nodup_map_iff σ.injective]
  exact List.nodup_finRange D

/-- Every mode is in the permutation list. -/
theorem mem_permList (σ : Equiv.Perm (Fin D)) (n : Fin D) :
    n ∈ (List.finRange D).map σ := by
  rw [List.mem_map]
  exact ⟨σ.symm n, List.mem_finRange _, σ.apply_symm_apply n⟩

/-- The permutation list's `toFinset` is all of `univ`. -/
theorem permList_toFinset (σ : Equiv.Perm (Fin D)) :
    ((List.finRange D).map σ).toFinset = (Finset.univ : Finset (Fin D)) := by
  ext x
  simp only [List.mem_toFinset, Finset.mem_univ, iff_true]
  exact mem_permList σ x

/-! ### `thm:Tucker-error-base` -/

/-- **Theorem `thm:Tucker-error-base` (axiom-free).** Let `U` have
    orthonormal-column factors and let `Y = chainProj U ((finRange D).map σ) X`
    be the mode-projection chain applied in the order of the permutation `σ`.
    Then

      maxₙ ‖Pₙ^⊥ X‖²  ≤  ‖X − Y‖²  ≤  ∑ₙ ‖Pₙ^⊥ X‖².

    The lower bound (the `⨆`) is per-mode: `‖Pₙ^⊥ X‖² ≤ ‖X − Y‖²` for every
    `n`, taken to a supremum. Since the per-mode projections commute, `Y` is
    independent of `σ`, so this is the lecture's truncated-HOSVD
    approximation for any application order. -/
theorem tucker_error_bounds (hU : ∀ n, Matrix.HasOrthonormalCols (U n))
    (σ : Equiv.Perm (Fin D)) (X : Tensor 𝕜 D dims) :
    (⨆ n, ‖projModePerp n (U n) X‖_F²)
        ≤ ‖X - chainProj U ((List.finRange D).map σ) X‖_F² ∧
      ‖X - chainProj U ((List.finRange D).map σ) X‖_F²
        ≤ ∑ n, ‖projModePerp n (U n) X‖_F² := by
  have hnodup := permList_nodup σ
  refine ⟨?_, ?_⟩
  · -- Lower bound: `⨆ n, ‖Pₙ^⊥ X‖² ≤ ‖X − Y‖²`.
    rcases isEmpty_or_nonempty (Fin D) with he | hne
    · -- `D = 0`: the supremum over an empty type is `0 ≤ ‖X − Y‖²`.
      rw [Real.iSup_of_isEmpty]
      exact frobNormSq_nonneg _
    · exact ciSup_le fun n =>
        chainProj_error_lower U hU hnodup (mem_permList σ n) X
  · -- Upper bound: `‖X − Y‖² ≤ ∑ₙ ‖Pₙ^⊥ X‖²`.
    calc ‖X - chainProj U ((List.finRange D).map σ) X‖_F²
        ≤ ∑ n ∈ ((List.finRange D).map σ).toFinset, ‖projModePerp n (U n) X‖_F² :=
          chainProj_error_upper U hU hnodup X
      _ = ∑ n, ‖projModePerp n (U n) X‖_F² := by rw [permList_toFinset σ]

end Tensor

/-! ### Eckart–Young best-rank residual (axiom B2 consumption)

The matrix-level best rank-`k` Frobenius residual, defined via the
projector that axiom B2 provides, together with the lower bound it
satisfies. This is the *only* place axiom B2 is consumed. -/

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The **best rank-`k` Frobenius residual** of a matrix `M`: the squared
    Frobenius norm of `M − U Uᴴ M`, where `U` is the orthonormal-column
    projector supplied by axiom B2 (Eckart–Young). By B2 this is the
    smallest rank-`≤ k` approximation error.

    Axiom B2 only applies when `k ≤ I` (there must be room for `k`
    orthonormal columns in `𝕜^I`), so the definition is guarded: for `k > I`
    it returns `0`. This regime is never hit by the consumers — the genuine
    use site (`hosvd_quasi_optimal`) always supplies `k = (X.modeUnfold n).rank
    ≤ dims n = I`. -/
def bestRankApproxErrorSq {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) (k : ℕ) : ℝ :=
  if h : k ≤ I then
    (M - (Classical.choose (Matrix.exists_best_rank_approx M k h)) *
         (Classical.choose (Matrix.exists_best_rank_approx M k h))ᴴ * M).frobNormSq
  else 0

/-- **Axiom B2, packaged.** When `k ≤ I`, the best rank-`k` residual
    lower-bounds the Frobenius error of *every* rank-`≤ k` matrix `B`:
    `bestRankApproxErrorSq M k ≤ ‖M − B‖²`. This is exactly the Eckart–Young
    optimality of axiom B2 (`Matrix.exists_best_rank_approx`). The `k ≤ I`
    hypothesis is the same guard B2 carries: for `k > I` the residual is the
    placeholder `0` and B2 does not apply. -/
theorem bestRankApproxErrorSq_le {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) (k : ℕ)
    (hk : k ≤ I) (B : Matrix (Fin I) (Fin C) 𝕜) (hB : B.rank ≤ k) :
    bestRankApproxErrorSq M k ≤ (M - B).frobNormSq := by
  rw [bestRankApproxErrorSq, dif_pos hk]
  exact (Classical.choose_spec (Matrix.exists_best_rank_approx M k hk)).2 B hB

end Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-! ### Frobenius bridge: tensor perp residual ↔ matrix residual -/

/-- `modeUnfold` respects subtraction. -/
theorem modeUnfold_sub (n : Fin D) (T₁ T₂ : Tensor 𝕜 D dims) :
    (T₁ - T₂).modeUnfold n = T₁.modeUnfold n - T₂.modeUnfold n := by
  funext i j
  simp only [modeUnfold_apply, Pi.sub_apply, Matrix.sub_apply]

/-- **Bridge.** The squared Frobenius norm of the mode-`n` complementary
    projection of `X` equals the matrix Frobenius residual of `X`'s mode-`n`
    unfolding under the projector `U Uᴴ`:
    `‖Pₙ^⊥ X‖² = ‖X_(n) − U Uᴴ X_(n)‖²_F`. Combines `frobNormSq_modeUnfold`
    (the matricization preserves the Frobenius norm) with `modeUnfold_projMode`
    (`(Pₙ X)_(n) = U Uᴴ X_(n)`).

    *Tensor↔matrix Frobenius bridge* — `hosvd_quasi_optimal` inlines the
    bridge in its lower-bound step (via `frobNormSq_modeUnfold` +
    `Matrix.frobNormSq_def` + `modeUnfold_sub`) for proof flexibility; this
    lemma packages the same chain as a one-line equation for user-facing
    code that wants to switch between tensor and matrix Frobenius residuals
    explicitly. -/
theorem frobNormSq_projModePerp_eq_matrix {R : Fin D → ℕ} (n : Fin D)
    (U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (X : Tensor 𝕜 D dims) :
    ‖projModePerp n U X‖_F² =
      (X.modeUnfold n - U * Uᴴ * X.modeUnfold n).frobNormSq := by
  rw [frobNormSq_modeUnfold n (projModePerp n U X), Matrix.frobNormSq_def,
    projModePerp_def, modeUnfold_sub, modeUnfold_projMode]

/-! ### Truncated HOSVD predicate and the singular-value error bound -/

/-- **`Y` is a truncated HOSVD of `X` of target multilinear rank `R`.**

    This packages: orthonormal-column factor matrices `U n` (with `R n`
    columns); the reconstruction `Y` is the mode-projection chain of `X` in
    the order of some permutation `σ`; and each factor attains the **best
    rank-`R n` Frobenius residual** at its mode, i.e.
    `‖Pₙ^⊥ X‖² = bestRankApproxErrorSq X_(n) (R n)` — this is the content of
    "the columns of `U n` span the top-`R n` left singular subspace of
    `X_(n)`" (the truncated SVD), expressed through the Eckart–Young
    residual. The predicate is invariant under in-subspace rotation of the
    `U n` columns (any orthonormal basis of the top-`R n` left singular
    subspace works), which is why we phrase it via the residual rather than
    fixing the singular vectors literally. -/
def IsTruncatedHOSVD (X : Tensor 𝕜 D dims) (R : Fin D → ℕ)
    (Y : Tensor 𝕜 D dims) : Prop :=
  ∃ (U : (n : Fin D) → Matrix (Fin (dims n)) (Fin (R n)) 𝕜) (σ : Equiv.Perm (Fin D)),
    (∀ n, Matrix.HasOrthonormalCols (U n)) ∧
    (∀ n, ‖projModePerp n (U n) X‖_F²
      = Matrix.bestRankApproxErrorSq (X.modeUnfold n) (R n)) ∧
    Y = chainProj U ((List.finRange D).map σ) X

/-- **Corollary `cor:HOSVD-error-bounds` (upper bound via singular values).**
    The truncated-HOSVD error is bounded by the sum of the trailing
    singular-value energies (best rank-`R n` residuals):

      ‖X − Y‖²  ≤  ∑ₙ bestRankApproxErrorSq X_(n) (R n).

    Combines the axiom-free upper bound of `tucker_error_bounds` with the
    truncated-HOSVD identity `‖Pₙ^⊥ X‖² = bestRankApproxErrorSq X_(n) (R n)`.
    (Uses axiom B2 only through `bestRankApproxErrorSq`.) -/
theorem hosvd_error_singular_values {X : Tensor 𝕜 D dims} {R : Fin D → ℕ}
    {Y : Tensor 𝕜 D dims} (hY : IsTruncatedHOSVD X R Y) :
    (X - Y).frobNormSq ≤ ∑ n, Matrix.bestRankApproxErrorSq (X.modeUnfold n) (R n) := by
  obtain ⟨U, σ, hUorth, hUbest, hYeq⟩ := hY
  rw [hYeq]
  calc ‖X - chainProj U ((List.finRange D).map σ) X‖_F²
      ≤ ∑ n, ‖projModePerp n (U n) X‖_F² := (tucker_error_bounds U hUorth σ X).2
    _ = ∑ n, Matrix.bestRankApproxErrorSq (X.modeUnfold n) (R n) :=
        Finset.sum_congr rfl (fun n _ => hUbest n)

/-! ### `thm:HOSVD-error`: quasi-optimality -/

/-- **Theorem `thm:HOSVD-error` (quasi-optimality of the truncated HOSVD).**
    If `Y` is a truncated HOSVD of `X` of target multilinear rank `R`, then
    for every tensor `Z` of multilinear rank `R`,

      ‖X − Y‖  ≤  √D · ‖X − Z‖.

    The hypothesis on `Z` is the *inequality* form `(Z.modeUnfold n).rank
    ≤ R n` for every mode `n` (i.e. `Z`'s multilinear rank is at most `R`),
    not exact equality. This matches the lecture's "for any rank-`(R₁,…,R_D)`
    tensor `Z`" (where "rank `≤`" is what the proof actually uses) and
    strictly generalises a `Z.HasMultirank R` hypothesis (via `le_of_eq`).

    Proof: pick a mode `j` maximising `‖Pₙ^⊥ X‖²`. The upper bound
    (`tucker_error_bounds`) and `∑ₙ ≤ D · maxₙ` give
    `‖X − Y‖² ≤ D · ‖Pⱼ^⊥ X‖²`. The truncated-HOSVD identity rewrites
    `‖Pⱼ^⊥ X‖²` as the best rank-`R j` residual of `X_(j)`, which by axiom B2
    (`bestRankApproxErrorSq_le`) lower-bounds `‖X_(j) − Z_(j)‖²` because
    `Z_(j)` has rank `≤ R j` (by hypothesis); and the matricization
    preserves the Frobenius norm, so `‖X_(j) − Z_(j)‖² = ‖X − Z‖²`. Hence
    `‖X − Y‖² ≤ D · ‖X − Z‖²`; take square roots. -/
theorem hosvd_quasi_optimal {X : Tensor 𝕜 D dims} {R : Fin D → ℕ}
    {Y : Tensor 𝕜 D dims} (hY : IsTruncatedHOSVD X R Y)
    {Z : Tensor 𝕜 D dims} (hZ : ∀ n, (Z.modeUnfold n).rank ≤ R n) :
    (X - Y).frobNorm ≤ Real.sqrt D * (X - Z).frobNorm := by
  obtain ⟨U, σ, hUorth, hUbest, hYeq⟩ := hY
  -- Abbreviate the per-mode residual energies.
  set f : Fin D → ℝ := fun n => ‖projModePerp n (U n) X‖_F² with hf
  rcases isEmpty_or_nonempty (Fin D) with hDempty | hDne
  · -- `D = 0`: there are no modes, so the projection list is empty and
    -- `Y = chainProj U [] X = X`; thus `‖X − Y‖ = 0 ≤ √0 · ‖X − Z‖`.
    have hnil : (List.finRange D) = [] := by
      rw [List.eq_nil_iff_forall_not_mem]
      exact fun x => absurd (List.mem_finRange x) (fun _ => hDempty.false x)
    rw [hYeq, hnil, List.map_nil, chainProj_nil, sub_self]
    rw [frobNorm_def, frobNormSq_def]
    simp only [Pi.zero_apply, norm_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
      zero_pow, Finset.sum_const_zero, Real.sqrt_zero]
    exact mul_nonneg (Real.sqrt_nonneg _) (frobNorm_nonneg _)
  · -- `D > 0`: choose a maximising mode `j`.
    obtain ⟨j, -, hj⟩ := Finset.exists_max_image (Finset.univ : Finset (Fin D)) f
      (Finset.univ_nonempty)
    -- Upper bound: `‖X − Y‖² ≤ D · f j`.
    have hupper : (X - Y).frobNormSq ≤ (D : ℝ) * f j := by
      rw [hYeq]
      calc ‖X - chainProj U ((List.finRange D).map σ) X‖_F²
          ≤ ∑ n, f n := (tucker_error_bounds U hUorth σ X).2
        _ ≤ ∑ _n : Fin D, f j := Finset.sum_le_sum (fun n _ => hj n (Finset.mem_univ n))
        _ = (D : ℝ) * f j := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    -- Lower bound: `f j ≤ ‖X − Z‖²` (axiom B2 at mode `j`).
    have hZj : (Z.modeUnfold j).rank ≤ R j := hZ j
    -- `R j ≤ dims j` discharges B2's `k ≤ I` guard: `Z.modeUnfold j` is a
    -- `Matrix (Fin (dims j)) …` of rank ≤ R j (by `hZ j`) and ≤ dims j (by
    -- `Matrix.rank_le_height`). But we need `R j ≤ dims j` for *any* `R j`,
    -- which doesn't follow from `Z` alone. Use the truncated-HOSVD factor
    -- `U j : Matrix (Fin (dims j)) (Fin (R j)) 𝕜` with `(U j)ᴴ * (U j) = 1`:
    -- the identity matrix of size `R j` has rank `R j`, and rank cannot
    -- exceed `dims j` (number of rows of `U j` from `Matrix.rank_le_height`
    -- applied to `(U j)ᴴ * (U j)`), forcing `R j ≤ dims j`.
    have hRj : R j ≤ dims j := by
      have hUU : ((U j)ᴴ * (U j)) = (1 : Matrix (Fin (R j)) (Fin (R j)) 𝕜) := hUorth j
      have h1rank : (1 : Matrix (Fin (R j)) (Fin (R j)) 𝕜).rank = R j := by
        rw [Matrix.rank_one]; exact Fintype.card_fin _
      have hLE1 : ((U j)ᴴ * (U j)).rank ≤ (U j).rank :=
        Matrix.rank_mul_le_right _ _
      have hLE2 : (U j).rank ≤ dims j := Matrix.rank_le_height (U j)
      have hEq : R j = ((U j)ᴴ * (U j)).rank := by rw [hUU]; exact h1rank.symm
      omega
    have hlower : f j ≤ (X - Z).frobNormSq := by
      rw [hf]
      simp only
      rw [hUbest j, frobNormSq_modeUnfold j (X - Z), ← Matrix.frobNormSq_def, modeUnfold_sub]
      exact Matrix.bestRankApproxErrorSq_le (X.modeUnfold j) (R j) hRj (Z.modeUnfold j) hZj
    -- Combine: `‖X − Y‖² ≤ D · ‖X − Z‖²`, then take square roots.
    have hcomb : (X - Y).frobNormSq ≤ (D : ℝ) * (X - Z).frobNormSq := by
      calc (X - Y).frobNormSq ≤ (D : ℝ) * f j := hupper
        _ ≤ (D : ℝ) * (X - Z).frobNormSq := by
            apply mul_le_mul_of_nonneg_left hlower (Nat.cast_nonneg D)
    -- `‖X − Y‖ = √‖X−Y‖² ≤ √(D·‖X−Z‖²) = √D · ‖X−Z‖`.
    rw [frobNorm_def, frobNorm_def]
    calc Real.sqrt (X - Y).frobNormSq
        ≤ Real.sqrt ((D : ℝ) * (X - Z).frobNormSq) := Real.sqrt_le_sqrt hcomb
      _ = Real.sqrt D * Real.sqrt (X - Z).frobNormSq :=
          Real.sqrt_mul (Nat.cast_nonneg D) _

/-! ### The truncated HOSVD as an explicit witness

`IsTruncatedHOSVD` is a *predicate* — it captures every valid truncated HOSVD,
quantifying over the mode-processing order `σ` and the (non-unique) best-rank
projectors. For symmetry with the truncated TT-SVD (whose canonical output is
the named function `Tensor.ttApprox`), we expose one canonical instance:
`hosvdApprox` processes the modes in their natural order (`σ = id`) with the
Eckart–Young best-rank projector at each mode. The bound theorems above then
specialize to it as corollaries. -/

/-- The canonical per-mode best-rank-`R n` projector (axiom B2's chosen
    orthonormal-column matrix for the mode-`n` unfolding). -/
noncomputable def hosvdProj {D : ℕ} {dims : Fin D → ℕ} (X : Tensor 𝕜 D dims)
    (R : Fin D → ℕ) (hR : ∀ n, R n ≤ dims n) (n : Fin D) :
    Matrix (Fin (dims n)) (Fin (R n)) 𝕜 :=
  Classical.choose (Matrix.exists_best_rank_approx (X.modeUnfold n) (R n) (hR n))

/-- **The truncated HOSVD output** of `X` at target multilinear rank `R`, as an
    explicit function: project onto the top-`R n` left singular subspace of each
    mode-`n` unfolding, processing the modes in their natural order. The canonical
    instance of `IsTruncatedHOSVD` (mirrors `Tensor.ttApprox` for TT-SVD). -/
noncomputable def hosvdApprox {D : ℕ} {dims : Fin D → ℕ} (X : Tensor 𝕜 D dims)
    (R : Fin D → ℕ) (hR : ∀ n, R n ≤ dims n) : Tensor 𝕜 D dims :=
  chainProj (hosvdProj X R hR) (List.finRange D) X

/-- `hosvdApprox` is a truncated HOSVD: the canonical instance of the predicate
    `IsTruncatedHOSVD`. -/
theorem isTruncatedHOSVD_hosvdApprox {D : ℕ} {dims : Fin D → ℕ} (X : Tensor 𝕜 D dims)
    (R : Fin D → ℕ) (hR : ∀ n, R n ≤ dims n) :
    IsTruncatedHOSVD X R (hosvdApprox X R hR) := by
  refine ⟨hosvdProj X R hR, Equiv.refl (Fin D), ?_, ?_, ?_⟩
  · intro n
    rw [Matrix.hasOrthonormalCols_iff, hosvdProj]
    exact (Classical.choose_spec (Matrix.exists_best_rank_approx (X.modeUnfold n) (R n) (hR n))).1
  · intro n
    rw [frobNormSq_projModePerp_eq_matrix, Matrix.bestRankApproxErrorSq, dif_pos (hR n), hosvdProj]
  · rw [hosvdApprox,
      show (List.finRange D).map ⇑(Equiv.refl (Fin D)) = List.finRange D from by simp]

/-- **Quasi-optimality of the canonical truncated HOSVD (function form).** The
    explicit output `hosvdApprox X R` is within `√D` of any tensor `Z` of
    multilinear rank `≤ R`. The `hosvdApprox` instance of `hosvd_quasi_optimal`. -/
theorem hosvdApprox_quasi_optimal {D : ℕ} {dims : Fin D → ℕ} (X : Tensor 𝕜 D dims)
    (R : Fin D → ℕ) (hR : ∀ n, R n ≤ dims n)
    {Z : Tensor 𝕜 D dims} (hZ : ∀ n, (Z.modeUnfold n).rank ≤ R n) :
    (X - hosvdApprox X R hR).frobNorm ≤ Real.sqrt D * (X - Z).frobNorm :=
  hosvd_quasi_optimal (isTruncatedHOSVD_hosvdApprox X R hR) hZ

/-- **Singular-value error bound for the canonical truncated HOSVD (function
    form).** The `hosvdApprox` instance of `hosvd_error_singular_values`. -/
theorem hosvdApprox_error_singular_values {D : ℕ} {dims : Fin D → ℕ} (X : Tensor 𝕜 D dims)
    (R : Fin D → ℕ) (hR : ∀ n, R n ≤ dims n) :
    (X - hosvdApprox X R hR).frobNormSq
      ≤ ∑ n, Matrix.bestRankApproxErrorSq (X.modeUnfold n) (R n) :=
  hosvd_error_singular_values (isTruncatedHOSVD_hosvdApprox X R hR)

end Tensor

end
