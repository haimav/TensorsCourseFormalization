/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Cubic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Data.Matrix.Basic

/-!
# Tensor Transposition and Symmetry

Transposition permutes the modes of a tensor. For a general tensor of shape
`dims`, the transposed tensor has shape `dims ∘ σ`; for a cubic tensor (all
modes of dimension `N`) the shape is preserved definitionally, so the
transpose returns a cubic tensor of the same type. This is what enables the
predicate `CubicTensor.IsSymmetric T := ∀ σ, T.transpose σ = T` to be stated
as a homogeneous equality.

## Main definitions

* `Tensor.transpose σ T` — transpose a tensor along the mode permutation `σ`.
* `CubicTensor.IsSymmetric T` — invariance under all mode permutations.

## Main results

* `Tensor.transpose_one` — `T.transpose 1 = T`.
* `Tensor.transpose_mul` — `(T.transpose σ).transpose τ = T.transpose (σ * τ)`.
* `Tensor.transpose_add`, `Tensor.transpose_smul` — linearity in `T`.
* `Tensor.transpose_swap_eq_matrix_transpose_general` — for a general order-2
  tensor (possibly non-cubic), swapping the two modes coincides with
  `Matrix.transpose`; `Tensor.transpose_swap_eq_matrix_transpose` is the cubic
  specialisation.
* `CubicTensor.isSymmetric_zero`, `CubicTensor.IsSymmetric.smul`,
  `CubicTensor.IsSymmetric.add` — symmetry is a linear subspace.
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- Transpose (permute modes) of a tensor by a permutation `σ : Perm (Fin D)`.
    The output shape is `dims ∘ σ`: the mode that used to be `σ d` is now `d`.

    The `▸` cast converts `Fin (dims (σ (σ.symm d)))` to `Fin (dims d)` via the
    propositional equality `σ (σ.symm d) = d`. -/
def transpose (σ : Equiv.Perm (Fin D))
    (T : Tensor 𝕜 D dims) : Tensor 𝕜 D (dims ∘ σ) :=
  fun idx => T (fun d => σ.apply_symm_apply d ▸ idx (σ.symm d))

-- Intentionally not `@[simp]`: the RHS carries a `▸` cast
-- (`σ.apply_symm_apply d ▸ …`), so eagerly rewriting with this would leave
-- cast-laden goals. Rewrite with it explicitly where the unfolded form is
-- wanted.
theorem transpose_apply (σ : Equiv.Perm (Fin D)) (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin ((dims ∘ σ) d)) :
    T.transpose σ idx = T (fun d => σ.apply_symm_apply d ▸ idx (σ.symm d)) := rfl

/-- Transposing by the identity permutation is the identity operation. -/
@[simp]
theorem transpose_one (T : Tensor 𝕜 D dims) : T.transpose 1 = T := by
  funext idx
  rfl

/-- Composition of transpositions. In Mathlib's convention `(σ * τ) d = σ (τ d)`,
    so transposing first by `σ` then by `τ` corresponds to transposing by `σ * τ`
    (the outer σ is applied after the inner τ).

    The output types match definitionally:
    `((dims ∘ σ) ∘ τ) d = dims (σ (τ d)) = (dims ∘ (σ * τ)) d`. -/
theorem transpose_mul (σ τ : Equiv.Perm (Fin D)) (T : Tensor 𝕜 D dims) :
    (T.transpose σ).transpose τ = T.transpose (σ * τ) := by
  funext idx
  -- Both sides evaluate `T` on a re-indexed multi-index. Establish equality
  -- of the re-indexed functions componentwise, then strip `▸` casts via HEq.
  simp only [transpose_apply]
  congr 1
  funext d
  -- `(σ * τ).symm d = τ.symm (σ.symm d)` holds definitionally. Both sides
  -- are `idx (τ.symm (σ.symm d))` wrapped in propositionally-equal
  -- `Fin`-type casts; `eqRec_heq_iff_heq` strips every `▸`.
  apply eq_of_heq
  -- The LHS has two nested ▸; strip each step explicitly.
  refine HEq.trans ?_ (heq_eqRec_iff_heq.mpr HEq.rfl)
  refine (eqRec_heq_iff_heq.mpr ?_)
  refine (eqRec_heq_iff_heq.mpr ?_)
  -- `(σ * τ).symm d` and `τ.symm (σ.symm d)` are definitionally equal.
  rfl

/-- Transpose is additive in the tensor argument. -/
theorem transpose_add (σ : Equiv.Perm (Fin D)) (T₁ T₂ : Tensor 𝕜 D dims) :
    (T₁ + T₂).transpose σ = T₁.transpose σ + T₂.transpose σ := by
  funext idx
  simp [transpose_apply]

/-- Transpose commutes with scalar multiplication. -/
theorem transpose_smul (σ : Equiv.Perm (Fin D)) (c : 𝕜) (T : Tensor 𝕜 D dims) :
    (c • T).transpose σ = c • T.transpose σ := by
  funext idx
  simp [transpose_apply]

/-- Transpose commutes with negation. -/
theorem transpose_neg (σ : Equiv.Perm (Fin D)) (T : Tensor 𝕜 D dims) :
    (-T).transpose σ = -(T.transpose σ) := by
  funext idx
  simp [transpose_apply]

/-- Transpose sends the zero tensor to the zero tensor. -/
@[simp]
theorem transpose_zero (σ : Equiv.Perm (Fin D)) :
    (0 : Tensor 𝕜 D dims).transpose σ = 0 := by
  funext idx
  simp [transpose_apply]

end Tensor

/-! ### Order-2 specialisation

For an order-2 tensor (i.e. a rectangular matrix) the transposition by the
swap permutation agrees with `Matrix.transpose`. In the general (possibly
non-cubic) case the output shape is `dims ∘ Equiv.swap 0 1`, so the equation
is between functions indexed by this reshaped dimension. The cubic case is a
direct specialisation. -/

section MatrixAgreement

variable {𝕜 : Type*} [RCLike 𝕜]

/-- An order-2 tensor is the same data as a (possibly rectangular) matrix.
    Swapping its two modes coincides with the matrix transpose.

    The equation is between functions
    `(d : Fin 2) → Fin ((dims ∘ Equiv.swap 0 1) d) → 𝕜`; after `funext` on
    the order-2 multi-index it reduces to re-evaluating the tensor on the
    coordinate pair in swapped order. -/
theorem Tensor.transpose_swap_eq_matrix_transpose_general
    {dims : Fin 2 → ℕ} (T : Tensor 𝕜 2 dims) :
    T.transpose (Equiv.swap 0 1) =
      (fun idx : (d : Fin 2) → Fin ((dims ∘ Equiv.swap 0 1) d) =>
        (Matrix.transpose (fun (i : Fin (dims 0)) (j : Fin (dims 1)) =>
          T (fun d => match d with
            | ⟨0, _⟩ => i
            | ⟨1, _⟩ => j))) (idx 0) (idx 1)) := by
  funext idx
  -- Unfold `transpose` and the matrix transpose; match the two tensors on
  -- the swapped pair `(idx 1, idx 0)`.
  simp only [Tensor.transpose_apply, Matrix.transpose_apply]
  -- After unfolding, both sides evaluate `T` at an index function that sends
  -- mode 0 to `idx 1` and mode 1 to `idx 0`. Establish equality of these
  -- index functions componentwise.
  congr 1
  funext d
  -- Case split on `d : Fin 2`. The swap is self-inverse:
  -- `(Equiv.swap 0 1).symm 0 = 1` and `(Equiv.swap 0 1).symm 1 = 0`.
  apply eq_of_heq
  simp only [eqRec_heq_iff_heq]
  match d with
  | ⟨0, _⟩ =>
    change HEq (idx ((Equiv.swap (0 : Fin 2) 1).symm 0)) (idx 1)
    rw [Equiv.symm_swap, Equiv.swap_apply_left]
  | ⟨1, _⟩ =>
    change HEq (idx ((Equiv.swap (0 : Fin 2) 1).symm 1)) (idx 0)
    rw [Equiv.symm_swap, Equiv.swap_apply_right]

/-- Cubic specialisation of `Tensor.transpose_swap_eq_matrix_transpose_general`:
    for an order-2 cubic tensor, swapping the two modes coincides with
    `Matrix.transpose`. -/
theorem Tensor.transpose_swap_eq_matrix_transpose
    {N : ℕ} (T : CubicTensor 𝕜 2 N) :
    T.transpose (Equiv.swap 0 1) =
      (fun idx : (d : Fin 2) → Fin N =>
        (Matrix.transpose (fun i j : Fin N => T (fun d => match d with
          | ⟨0, _⟩ => i
          | ⟨1, _⟩ => j))) (idx 0) (idx 1)) :=
  Tensor.transpose_swap_eq_matrix_transpose_general T

end MatrixAgreement

/-! ### Symmetry of cubic tensors -/

namespace CubicTensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D N : ℕ}

/-- A cubic tensor is *symmetric* if it is invariant under every permutation
    of its modes. Because `(fun _ : Fin D => N) ∘ σ` is definitionally
    `fun _ => N`, the equality `T.transpose σ = T` is a homogeneous equation
    in `CubicTensor 𝕜 D N`. -/
def IsSymmetric (T : CubicTensor 𝕜 D N) : Prop :=
  ∀ σ : Equiv.Perm (Fin D), T.transpose σ = T

/-- The zero tensor is symmetric. -/
theorem isSymmetric_zero : IsSymmetric (0 : CubicTensor 𝕜 D N) := by
  intro σ
  exact Tensor.transpose_zero σ

/-- Scalar multiples of symmetric tensors are symmetric. -/
theorem IsSymmetric.smul {T : CubicTensor 𝕜 D N} (hT : IsSymmetric T) (c : 𝕜) :
    IsSymmetric (c • T) := fun σ => by
  rw [Tensor.transpose_smul, hT σ]
  rfl

/-- Sums of symmetric tensors are symmetric. -/
theorem IsSymmetric.add {T₁ T₂ : CubicTensor 𝕜 D N}
    (h₁ : IsSymmetric T₁) (h₂ : IsSymmetric T₂) :
    IsSymmetric (T₁ + T₂) := fun σ => by
  rw [Tensor.transpose_add, h₁ σ, h₂ σ]
  rfl

/-- Negation preserves symmetry. -/
theorem IsSymmetric.neg {T : CubicTensor 𝕜 D N} (hT : IsSymmetric T) :
    IsSymmetric (-T) := fun σ => by
  rw [Tensor.transpose_neg, hT σ]
  rfl

end CubicTensor

end
