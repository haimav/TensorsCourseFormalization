/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.MatMulTensor.Defs
import TensorCourse.Matrix.Vectorize
import Mathlib.Data.Matrix.Mul

/-!
# Matrix Multiplication Tensor Encodes Matrix Multiplication

This file proves that the order-3 matrix multiplication tensor `matMulTensor 𝕜 M`
correctly encodes matrix multiplication: for any two `M × M` matrices `A` and `B`,

`(vec(AB))_l = ∑ i, ∑ j, (matMulTensor 𝕜 M) i j l * vec(A) i * vec(B) j`

where `vec` (`Matrix.vectorize`, defined in
`TensorCourse/Matrix/Vectorize.lean`) is the column-major flattening of a
matrix into a vector: `Matrix.vectorize A l = A (l % M) (l / M)`.

## Main results

* `matMulTensor_encodes_matMul` — the encoding correctness theorem
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- Build an order-3 multi-index from three components, for use as an input
    to the matrix multiplication tensor.

    The three components `i`, `j`, `l` in `Fin (M * M)` populate modes 0, 1, 2
    respectively. This helper avoids relying on `![...]` notation, which has
    unfolding issues with the constant dimension function `matMulDims M`.

    This is part of the public API: downstream users (e.g. `Lecture1/Theorem4`)
    need to evaluate the matrix multiplication tensor at explicit multi-indices
    built from three `Fin (M * M)` components. -/
def triIdx {M : ℕ} (i j l : Fin (M * M)) :
    (d : Fin 3) → Fin (matMulDims M d) :=
  fun d => match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => l
    | ⟨n + 3, h⟩ => absurd h (by omega)

/-- `triIdx i j l` sends mode `0` to `i`. -/
@[simp]
theorem triIdx_zero {M : ℕ} (i j l : Fin (M * M)) :
    triIdx i j l 0 = i := rfl

/-- `triIdx i j l` sends mode `1` to `j`. -/
@[simp]
theorem triIdx_one {M : ℕ} (i j l : Fin (M * M)) :
    triIdx i j l 1 = j := rfl

/-- `triIdx i j l` sends mode `2` to `l`. -/
@[simp]
theorem triIdx_two {M : ℕ} (i j l : Fin (M * M)) :
    triIdx i j l 2 = l := rfl

/-- Convenience: the value components of `finProdFinEquiv.symm (l.cast h)` and
    `finProdFinEquiv.symm l` coincide for a square shape, since `Fin.cast` is
    value-preserving. Helper used in unpacking `vec`. -/
private theorem finProdFinEquiv_symm_cast_square {M : ℕ} (l : Fin (M * M))
    (h : M * M = M * M) :
    (finProdFinEquiv.symm (l.cast h) : Fin M × Fin M) =
      finProdFinEquiv.symm l := by
  ext <;> simp

/-- **Encoding correctness of the matrix multiplication tensor.**

    For any two `M × M` matrices `A, B` and any target index `l : Fin (M*M)`:

    `∑ i, ∑ j, (matMulTensor 𝕜 M) i j l * vec(A) i * vec(B) j = vec(A * B) l`

    This is the key lemma linking the CP rank of the matrix multiplication tensor
    to the number of bilinear products needed to compute a matrix product. -/
theorem matMulTensor_encodes_matMul {M : ℕ} (A B : Matrix (Fin M) (Fin M) 𝕜)
    (l : Fin (M * M)) :
    ∑ i : Fin (M * M), ∑ j : Fin (M * M),
      matMulTensor 𝕜 M (triIdx i j l) * Matrix.vectorize A i * Matrix.vectorize B j =
        Matrix.vectorize (A * B) l := by
  -- Abbreviate the row / column of `l` under the column-major `vec` convention.
  set lRow : Fin M := (finProdFinEquiv.symm l).2 with hlRow
  set lCol : Fin M := (finProdFinEquiv.symm l).1 with hlCol
  -- Reindex the outer sum (over i : Fin (M*M)) via finProdFinEquiv.
  rw [← (finProdFinEquiv.sum_comp
    (fun i => ∑ j : Fin (M * M),
      matMulTensor 𝕜 M (triIdx i j l) * Matrix.vectorize A i * Matrix.vectorize B j))]
  -- Same for the inner sum (over j)
  simp_rw [← (finProdFinEquiv.sum_comp
    (fun j => matMulTensor 𝕜 M (triIdx _ j l) * Matrix.vectorize A _ * Matrix.vectorize B j))]
  -- Expand sums over Fin M × Fin M into iterated sums (over the pair components).
  rw [Fintype.sum_prod_type]
  simp_rw [Fintype.sum_prod_type]
  -- Simplify the tensor term and vec. After this, the summand reads
  --   (if y = lRow ∧ x' = lCol ∧ x = y' then 1 else 0) * A y x * B y' x'
  -- where `(x, y)` is the reindexed `i` and `(x', y')` is the reindexed `j`.
  simp only [matMulTensor_apply, triIdx_zero, triIdx_one, triIdx_two,
    Equiv.symm_apply_apply, Matrix.vectorize_apply, finProdFinEquiv_symm_cast_square,
    ← hlRow, ← hlCol]
  -- The RHS unpacks to `(A * B) lRow lCol = ∑ k, A lRow k * B k lCol`.
  rw [Matrix.mul_apply]
  -- Merge `(if P then 1 else 0) * A y x * B y' x'` into `if P then A y x * B y' x' else 0`.
  have h_pull : ∀ x y x' y' : Fin M,
      (if y = lRow ∧ x' = lCol ∧ x = y' then (1 : 𝕜) else 0) * A y x * B y' x'
        = if y = lRow ∧ x' = lCol ∧ x = y' then A y x * B y' x' else 0 := by
    intro x y x' y'
    split_ifs <;> simp
  simp_rw [h_pull]
  -- Decompose `ite (P ∧ Q ∧ R) a 0` into nested `ite`s.
  have h_split : ∀ x y x' y' : Fin M,
      (if y = lRow ∧ x' = lCol ∧ x = y' then A y x * B y' x' else 0)
        = (if y = lRow then
            (if x' = lCol then
              (if x = y' then A y x * B y' x' else 0) else 0) else 0) := by
    intro x y x' y'
    rw [ite_and, ite_and]
  simp_rw [h_split]
  -- Collapse the innermost `∑ y', if x = y' then _ else 0` to a single term at `y' = x`.
  have step1 :
      (∑ x : Fin M, ∑ y : Fin M, ∑ x' : Fin M, ∑ y' : Fin M,
          if y = lRow then
            (if x' = lCol then
              (if x = y' then A y x * B y' x' else 0) else 0) else 0)
      = ∑ x : Fin M, ∑ y : Fin M, ∑ x' : Fin M,
          if y = lRow then
            (if x' = lCol then A y x * B x x' else 0) else 0 := by
    apply Finset.sum_congr rfl; intro x _
    apply Finset.sum_congr rfl; intro y _
    apply Finset.sum_congr rfl; intro x' _
    -- Pull the outer two `if ... then ... else 0` out of the ∑ y'.
    rw [show (∑ y' : Fin M,
          if y = lRow then
            (if x' = lCol then
              (if x = y' then A y x * B y' x' else 0) else 0) else 0)
        = if y = lRow then
            (if x' = lCol then
              (∑ y' : Fin M, if x = y' then A y x * B y' x' else 0) else 0) else 0 from by
          by_cases hy : y = lRow <;> by_cases hx' : x' = lCol <;>
            simp [hy, hx']]
    -- Collapse ∑ y' at y' = x.
    have h_collapse :
        (∑ y' : Fin M, if x = y' then A y x * B y' x' else 0)
          = A y x * B x x' := by
      rw [Finset.sum_ite_eq Finset.univ x (fun y' => A y x * B y' x')]
      simp
    rw [h_collapse]
  rw [step1]
  -- Pull `if y = lRow` out of `∑ x'`, then collapse `∑ x'` to `x' = lCol`.
  have step2 :
      (∑ x : Fin M, ∑ y : Fin M, ∑ x' : Fin M,
          if y = lRow then
            (if x' = lCol then A y x * B x x' else 0) else 0)
      = ∑ x : Fin M, ∑ y : Fin M,
          if y = lRow then A y x * B x lCol else 0 := by
    apply Finset.sum_congr rfl; intro x _
    apply Finset.sum_congr rfl; intro y _
    -- Pull the outer `if y = lRow` out of the ∑ x'.
    rw [show (∑ x' : Fin M,
          if y = lRow then (if x' = lCol then A y x * B x x' else 0) else 0)
        = if y = lRow then
            (∑ x' : Fin M, if x' = lCol then A y x * B x x' else 0) else 0 from by
          by_cases hy : y = lRow <;> simp [hy]]
    -- Collapse the inner ∑ x' at x' = lCol.
    congr 1
    rw [Finset.sum_ite_eq' Finset.univ lCol (fun x' => A y x * B x x')]
    simp
  rw [step2]
  -- Collapse `∑ y` to `y = lRow`.
  have step3 :
      (∑ x : Fin M, ∑ y : Fin M,
          if y = lRow then A y x * B x lCol else 0)
      = ∑ x : Fin M, A lRow x * B x lCol := by
    apply Finset.sum_congr rfl; intro x _
    rw [Finset.sum_ite_eq' Finset.univ lRow (fun y => A y x * B x lCol)]
    simp
  rw [step3]

end Tensor
