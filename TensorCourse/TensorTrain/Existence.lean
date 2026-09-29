/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorTrain.Defs
import TensorCourse.Matrix.ColumnSpaceProjector
import TensorCourse.Matrix.SemiOrthogonal

/-!
# Tensor Train: the split-rank factorization (`thm:tt-existence`)

This file formalizes the structural heart of the TT existence theorem
(`thm:tt-existence`): the **one-step split-rank factorization**. The lecture
proves existence by induction on the order; the inductive step factors the
mode-`j` split `X_{[j]} = U Vᵀ` (a rank-`rⱼ` factorization) and realizes it
as a contractive product `X = U ⋈ₜ V`. The lecture states: "We can now
recursively apply this idea ... This is the idea behind the proof of the
following theorem."

We formalize that step in full:

* `Tensor.exists_ctrProd_of_split` — every `X` factors as `X = A ⋈ₜ V` with
  shared bond dimension exactly the **split rank** `splitRank X`, where the
  boundary matricization `ctrLeftMat A` has **orthonormal columns**
  (the lecture's `U`, taken from the column-space orthonormal projector).

Combined with the converse bound `Tensor.splitRank_ctrProd_le` (in
`TensorTrain/Defs`), this pins down the split rank as the *minimal* shared
bond dimension of any contractive factorization of `X`:

  `splitRank X = min { S : ∃ A V, X = A ⋈ₜ V over a bond of size S }`.

The factorization is **axiom-free**: it rests only on the column-space
orthonormal-projector lemma `Matrix.exists_colspace_orthonormal_projector`
(itself `sorry`-free, no axioms), the contractive-product matrix form
`modeSplit_ctrProd`, and injectivity of the mode split.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ}

/-- **Split-rank factorization (the inductive step of `thm:tt-existence`).**
    Every tensor `X` (of shape `combinedDims da db`, split after the first
    `a` modes) factors as a contractive product

      `X = A ⋈ₜ V`

    over a shared bond of dimension `splitRank X = rank X_{[a]}`, where the
    left factor's boundary matricization `ctrLeftMat A` has orthonormal
    columns (the lecture's orthonormal `U`).

    Proof: the column-space orthonormal projector of `X_{[a]} = modeSplit X`
    gives `U` with `Uᴴ U = 1` and `U Uᴴ X_{[a]} = X_{[a]}`, with exactly
    `rank X_{[a]} = splitRank X` columns. Take `A := ctrLeftFold U` and
    `V := ctrRightFold (Uᴴ X_{[a]})`. Then `ctrLeftMat A = U` (orthonormal),
    and by the matrix form `modeSplit (A ⋈ₜ V) = U (Uᴴ X_{[a]}) = X_{[a]}`,
    so `A ⋈ₜ V = X` by injectivity of the mode split. -/
theorem exists_ctrProd_of_split (X : Tensor 𝕜 (a + b) (combinedDims da db)) :
    ∃ (A : Tensor 𝕜 (a + 1) (Fin.snoc da (splitRank X)))
      (V : Tensor 𝕜 (b + 1) (Fin.cons (splitRank X) db)),
      (ctrLeftMat A)ᴴ * (ctrLeftMat A) = 1 ∧ X = A ⋈ₜ V := by
  obtain ⟨U, hUorth, hUfix⟩ :=
    Matrix.exists_colspace_orthonormal_projector (modeSplit X)
  -- `U : Matrix (Fin (∏ da)) (Fin (modeSplit X).rank)`, and
  -- `splitRank X = (modeSplit X).rank` definitionally.
  refine ⟨ctrLeftFold U, ctrRightFold (Uᴴ * modeSplit X), ?_, ?_⟩
  · rw [ctrLeftMat_ctrLeftFold]; exact hUorth
  · apply modeSplit_injective
    rw [modeSplit_ctrProd, ctrLeftMat_ctrLeftFold, ctrRightMat_ctrRightFold]
    exact hUfix.symm.trans (Matrix.mul_assoc U Uᴴ (modeSplit X))

/-- **The split rank is the minimal contractive-bond dimension.** Packaging
    `exists_ctrProd_of_split` (a factorization over a bond of size
    `splitRank X` exists) with `splitRank_ctrProd_le` (no factorization can
    use a smaller bond): `splitRank X` is realized, and it is a lower bound
    for the shared dimension of any contractive factorization of `X`. -/
theorem splitRank_eq_min_ctrBond (X : Tensor 𝕜 (a + b) (combinedDims da db)) :
    (∃ (A : Tensor 𝕜 (a + 1) (Fin.snoc da (splitRank X)))
        (V : Tensor 𝕜 (b + 1) (Fin.cons (splitRank X) db)), X = A ⋈ₜ V) ∧
      (∀ (S : ℕ) (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
          (V : Tensor 𝕜 (b + 1) (Fin.cons S db)), X = A ⋈ₜ V → splitRank X ≤ S) := by
  refine ⟨?_, ?_⟩
  · obtain ⟨A, V, _, hX⟩ := exists_ctrProd_of_split X
    exact ⟨A, V, hX⟩
  · intro S A V hX
    rw [hX]
    exact splitRank_ctrProd_le A V

/-! ### Matrix–vector special case

The lecture remarks: "the contractive product between a matrix and vector is
the usual matrix-vector product (exercise: convince yourself)." With a
right operand of order `1` (a vector, `b = 0`), the mode split of `A ⋈ₜ v`
is exactly the matrix-vector product of the matricizations: column `j` of
`modeSplit (A ⋈ₜ v)` is `(ctrLeftMat A).mulVec` of column `j` of
`ctrRightMat v`. A direct corollary of `modeSplit_ctrProd`. -/

/-- **Matrix–vector special case.** For an order-`(a+1)` left operand `A` and
    an order-`1` (vector) right operand `v`, the contractive product realises
    matrix-vector multiplication: `(A ⋈ₜ v)_{[a]} = (ctrLeftMat A) ·ᵥ v`. -/
theorem modeSplit_ctrProd_vector {db : Fin 0 → ℕ}
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S)) (v : Tensor 𝕜 (0 + 1) (Fin.cons S db))
    (i : Fin (∏ d, da d)) (j : Fin (∏ d : Fin 0, db d)) :
    modeSplit (A ⋈ₜ v) i j = (ctrLeftMat A).mulVec (fun s => ctrRightMat v s j) i := by
  rw [modeSplit_ctrProd, Matrix.mul_apply]
  rfl

/-! ### Mode split and factorization at an arbitrary split point

The lecture's mode-`j` split `X_{[j]}` and the factorization are needed at
*every* split point for the existence induction. We generalise `modeSplit`
to an arbitrary split `p` of an arbitrary-shape tensor, and lift the
factorization to it. -/

/-- **General mode split.** Group the first `p` modes into the rows and the
    last `q` modes into the columns of an arbitrary-shape tensor — the
    lecture's `X_{[p]}` for any split point. -/
def modeSplitAt {p q : ℕ} {dims : Fin (p + q) → ℕ} (X : Tensor 𝕜 (p + q) dims) :
    Matrix (Fin (∏ i : Fin p, dims (Fin.castAdd q i)))
           (Fin (∏ i : Fin q, dims (Fin.natAdd p i))) 𝕜 :=
  Matrix.of fun ri cj =>
    X (Fin.addCases (motive := fun d => Fin (dims d))
        ((multiIndexEquiv (fun i => dims (Fin.castAdd q i))).symm ri)
        ((multiIndexEquiv (fun j => dims (Fin.natAdd p j))).symm cj))

theorem modeSplitAt_apply {p q : ℕ} {dims : Fin (p + q) → ℕ} (X : Tensor 𝕜 (p + q) dims)
    (ri : Fin (∏ i : Fin p, dims (Fin.castAdd q i)))
    (cj : Fin (∏ i : Fin q, dims (Fin.natAdd p i))) :
    modeSplitAt X ri cj =
      X (Fin.addCases (motive := fun d => Fin (dims d))
        ((multiIndexEquiv (fun i => dims (Fin.castAdd q i))).symm ri)
        ((multiIndexEquiv (fun j => dims (Fin.natAdd p j))).symm cj)) := rfl

/-- The **general split rank** `rₚ = rank X_{[p]}`. -/
def splitRankAt {p q : ℕ} {dims : Fin (p + q) → ℕ} (X : Tensor 𝕜 (p + q) dims) : ℕ :=
  (modeSplitAt X).rank

/-- Any shape factors as the `combinedDims` of its first-`p` and last-`q`
    restrictions. The transport lemma that lets the general split reuse the
    `combinedDims`-form machinery. -/
theorem combinedDims_split_eq {p q : ℕ} (dims : Fin (p + q) → ℕ) :
    combinedDims (fun i => dims (Fin.castAdd q i)) (fun j => dims (Fin.natAdd p j)) = dims := by
  funext d
  refine Fin.addCases (motive := fun d =>
    combinedDims (fun i => dims (Fin.castAdd q i)) (fun j => dims (Fin.natAdd p j)) d = dims d)
    ?_ ?_ d
  · intro i; rw [combinedDims_castAdd]
  · intro j; rw [combinedDims_natAdd]

/-- **Factorization at an arbitrary split point.** Every tensor factors as a
    contractive product `X = A ⋈ₜ V` across its mode-`p` split, over a bond of
    dimension `splitRank` (= `rank X_{[p]}`), with `A`'s boundary
    matricization orthonormal. The general-`p` form of `exists_ctrProd_of_split`
    used at each step of the TT existence construction. -/
theorem exists_ctrProd_of_splitAt {p q : ℕ} {dims : Fin (p + q) → ℕ}
    (X : Tensor 𝕜 (p + q) dims) :
    ∃ (S : ℕ)
      (A : Tensor 𝕜 (p + 1) (Fin.snoc (fun i => dims (Fin.castAdd q i)) S))
      (V : Tensor 𝕜 (q + 1) (Fin.cons S (fun j => dims (Fin.natAdd p j)))),
      (ctrLeftMat A)ᴴ * ctrLeftMat A = 1 ∧
      cast (congrArg (Tensor 𝕜 (p + q)) (combinedDims_split_eq dims).symm) X = A ⋈ₜ V := by
  obtain ⟨A, V, hAorth, hX'eq⟩ := exists_ctrProd_of_split
    (cast (congrArg (Tensor 𝕜 (p + q)) (combinedDims_split_eq dims).symm) X)
  exact ⟨splitRank _, A, V, hAorth, hX'eq⟩

end Tensor
