/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Transpose

/-!
# Diagonal Tensors

A *diagonal* tensor has a nonzero entry only at multi-indices with all
coordinates equal. It is parameterised by a vector `v : Fin N → 𝕜`:
the entry at the "constant" multi-index `(i, i, …, i)` is `v i`.

## Main definitions

* `CubicTensor.diagonal v` — the diagonal tensor for the coefficient vector `v`.

## Main results

* `CubicTensor.diagonal_isSymmetric` — every diagonal tensor is symmetric.
-/

noncomputable section

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}

/-- The diagonal tensor with coefficient vector `v`. Its value at a
    multi-index `idx : Fin D → Fin N` is `v (idx ⟨0, _⟩)` if every
    coordinate of `idx` is the same, and `0` otherwise.

    The guarded `0 < D` handles the degenerate case `D = 0`, where
    `Fin 0` is empty and `idx 0` is ill-typed. For `D = 0` the
    "all indices equal" predicate is vacuously true but there is no
    `0 : Fin 0` to index `v` with, so we return `0`. -/
def diagonal (v : Fin N → 𝕜) : CubicTensor 𝕜 D N :=
  fun idx =>
    if ∀ d₁ d₂ : Fin D, idx d₁ = idx d₂ then
      if h : 0 < D then v (idx ⟨0, h⟩) else 0
    else 0

/-- The value of `diagonal v` at the "constant" multi-index `fun _ => i`
    (assuming `D > 0`) is `v i`. -/
theorem diagonal_const_apply (v : Fin N → 𝕜) (i : Fin N) (h : 0 < D) :
    (diagonal v : CubicTensor 𝕜 D N) (fun _ => i) = v i := by
  unfold diagonal
  rw [if_pos (fun _ _ => rfl), dif_pos h]

/-- The value of `diagonal v` at an *off-diagonal* multi-index `idx` (one
    whose coordinates are not all equal) is `0`. Complements
    `diagonal_const_apply`. -/
theorem diagonal_apply_of_not_const (v : Fin N → 𝕜) {idx : Fin D → Fin N}
    (h : ¬ ∀ d₁ d₂ : Fin D, idx d₁ = idx d₂) :
    (diagonal v : CubicTensor 𝕜 D N) idx = 0 := by
  unfold diagonal
  rw [if_neg h]

/-- For an order-0 cubic tensor, `diagonal v` is the zero tensor. -/
@[simp]
theorem diagonal_zero_order (v : Fin N → 𝕜) :
    (diagonal v : CubicTensor 𝕜 0 N) = 0 := by
  funext idx
  unfold diagonal
  simp

/-- On a cubic tensor (shape `fun _ : Fin D => N`), the `▸` cast in
    `Tensor.transpose` is along a constant motive and therefore the
    identity. So `(T.transpose σ) idx = T (fun d => idx (σ.symm d))`. -/
private theorem cubic_transpose_apply (σ : Equiv.Perm (Fin D))
    (T : CubicTensor 𝕜 D N) (idx : (d : Fin D) → Fin N) :
    T.transpose σ idx = T (fun d => idx (σ.symm d)) := by
  simp only [Tensor.transpose_apply]
  -- After unfolding, LHS = T (fun d => σ.apply_symm_apply d ▸ idx (σ.symm d)).
  -- The ▸ is a cast along a constant motive `fun _ => Fin N`, so it's the
  -- identity. Match the two tensor arguments componentwise.
  congr 1
  funext d
  -- Strip the ▸ via HEq (the motive `fun _ => Fin N` is constant).
  apply eq_of_heq
  exact eqRec_heq_iff_heq.mpr HEq.rfl

/-- Helper: `diagonal v` evaluated at any multi-index `jdx` equals
    `v (jdx ⟨0, _⟩)` if `jdx` is constant and `D > 0`, and `0` otherwise. -/
private theorem diagonal_apply (v : Fin N → 𝕜) (jdx : Fin D → Fin N) :
    (diagonal v : CubicTensor 𝕜 D N) jdx =
      if ∀ d₁ d₂ : Fin D, jdx d₁ = jdx d₂ then
        (if hD : 0 < D then v (jdx ⟨0, hD⟩) else 0)
      else 0 := rfl

/-- A diagonal tensor is symmetric — it is invariant under every permutation
    of its modes. Both sides of the equation evaluate to `v i` on the
    constant multi-index `fun _ => i`, and to `0` otherwise; the "all
    indices equal" predicate is preserved under any bijection of modes. -/
theorem diagonal_isSymmetric (v : Fin N → 𝕜) :
    IsSymmetric (diagonal v : CubicTensor 𝕜 D N) := by
  intro σ
  -- Reduce to a pointwise statement on a plain `Fin D → Fin N`.
  suffices h : ∀ idx : Fin D → Fin N,
      (diagonal v : CubicTensor 𝕜 D N) (fun d => idx (σ.symm d)) =
        (diagonal v : CubicTensor 𝕜 D N) idx by
    funext idx
    rw [cubic_transpose_apply]
    exact h idx
  intro idx
  rw [diagonal_apply v (fun d => idx (σ.symm d)), diagonal_apply v idx]
  -- `idx ∘ σ.symm` is constant iff `idx` is constant (σ.symm is a bijection).
  have hconst_iff : (∀ d₁ d₂ : Fin D, idx (σ.symm d₁) = idx (σ.symm d₂)) ↔
      (∀ d₁ d₂ : Fin D, idx d₁ = idx d₂) := by
    refine ⟨fun hall d₁ d₂ => ?_, fun hall _ _ => hall _ _⟩
    have h := hall (σ d₁) (σ d₂)
    rw [Equiv.symm_apply_apply, Equiv.symm_apply_apply] at h
    exact h
  -- Collapse both `ite`s via `ite_congr`.
  refine ite_congr (propext hconst_iff) (fun hconst => ?_) (fun _ => rfl)
  by_cases hD : 0 < D
  · rw [dif_pos hD, dif_pos hD]
    congr 1
    exact hconst _ _
  · rw [dif_neg hD, dif_neg hD]

end CubicTensor

end
