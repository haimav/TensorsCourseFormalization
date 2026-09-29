/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.OuterProd

/-!
# The Contractive Product of Tensors

The **contractive product** `A ⋈ₜ B` (lecture notation `\bowtie`) contracts
the **last** mode of an order-`(a+1)` tensor `A` with the **first** mode of
an order-`(b+1)` tensor `B` whose dimensions match (both equal `S`),
producing an order-`(a+b)` tensor:

  `(A ⋈ₜ B)(i₁,…,i_a, j₁,…,j_b) = ∑ₛ A(i₁,…,i_a, s) · B(s, j₁,…,j_b)`.

This is the lecture's own notation and operation (Lecture 6, §"Tensor Train
Decomposition"), invented there for writing the TT decomposition
`X = T⁽¹⁾ ⋈ₜ ⋯ ⋈ₜ T⁽ᴰ⁾` in a tensorial way. The lecture defines it as
`A ⋈ B = Tr_{ord A, ord A + 1}(A ◯ B)` (contract the adjacent shared modes of
the outer product); we take the equivalent element-wise sum as the
definition.

## Parametrization

To make the shared (contracted) mode explicit we parametrize:

* `A : Tensor 𝕜 (a+1) (Fin.snoc da S)` — free dims `da`, last mode `S`;
* `B : Tensor 𝕜 (b+1) (Fin.cons S db)` — first mode `S`, free dims `db`;
* `A ⋈ₜ B : Tensor 𝕜 (a+b) (combinedDims da db)`.

The result shape reuses `OuterProd`'s `combinedDims` (`Fin.addCases`
concatenation) and the `splitLeft`/`splitRight` projections. Both boundary
indices are built with `Fin.cast` along `Fin.snoc_*` / `Fin.cons_*`, with the
reduction lemmas' motive pinned to `fun _ => ℕ` (the dimension level).

## Main definitions

* `Tensor.ctrProd` (`A ⋈ₜ B`) — the contractive product.

## Main results

* `Tensor.ctrProd_apply` — the defining element-wise sum.
* `Tensor.ctrProd_add_left/right`, `ctrProd_smul_left/right`,
  `zero_ctrProd`, `ctrProd_zero` — bilinearity (the lecture's
  distributivity exercise).
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ} {S : ℕ}

/-- The **contractive product** `A ⋈ₜ B`: contract the last mode of `A`
    (dimension `S`) with the first mode of `B` (dimension `S`).

    `(A ⋈ₜ B) idx = ∑ s : Fin S, A (free_a idx ++ s) · B (s ++ free_b idx)`,
    where `free_a`/`free_b` are the left/right halves of `idx` over the
    combined free dimensions `combinedDims da db`. -/
def ctrProd
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    Tensor 𝕜 (a + b) (combinedDims da db) :=
  fun idx =>
    ∑ s : Fin S,
      A (Fin.lastCases (motive := fun d => Fin ((Fin.snoc da S : Fin (a + 1) → ℕ) d))
            (Fin.cast (@Fin.snoc_last a (fun _ => ℕ) S da).symm s)
            (fun i => Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i).symm
              (splitLeft idx i))) *
      B (Fin.cons (α := fun d => Fin ((Fin.cons S db : Fin (b + 1) → ℕ) d))
            (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) S db).symm s)
            (fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i).symm (splitRight idx i)))

@[inherit_doc] scoped infixl:70 " ⋈ₜ " => ctrProd

/-- The defining element-wise sum of the contractive product. -/
theorem ctrProd_apply
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db))
    (idx : (d : Fin (a + b)) → Fin (combinedDims da db d)) :
    (A ⋈ₜ B) idx =
      ∑ s : Fin S,
        A (Fin.lastCases (motive := fun d => Fin ((Fin.snoc da S : Fin (a + 1) → ℕ) d))
              (Fin.cast (@Fin.snoc_last a (fun _ => ℕ) S da).symm s)
              (fun i => Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i).symm
                (splitLeft idx i))) *
        B (Fin.cons (α := fun d => Fin ((Fin.cons S db : Fin (b + 1) → ℕ) d))
              (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) S db).symm s)
              (fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i).symm
                (splitRight idx i))) :=
  rfl

/-! ### Bilinearity (the distributivity exercise) -/

/-- Contractive product is additive in the left argument. -/
theorem ctrProd_add_left (A₁ A₂ : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    (A₁ + A₂) ⋈ₜ B = A₁ ⋈ₜ B + A₂ ⋈ₜ B := by
  ext idx
  simp only [ctrProd_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun s _ => by ring)

/-- Contractive product is additive in the right argument (the lecture's
    distributivity exercise, `A ⋈ (B + C) = A ⋈ B + A ⋈ C`). -/
theorem ctrProd_add_right (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B₁ B₂ : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    A ⋈ₜ (B₁ + B₂) = A ⋈ₜ B₁ + A ⋈ₜ B₂ := by
  ext idx
  simp only [ctrProd_apply, Pi.add_apply]
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun s _ => by ring)

/-- Scalar multiplication pulls out of the left factor. -/
theorem ctrProd_smul_left (c : 𝕜) (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    (c • A) ⋈ₜ B = c • (A ⋈ₜ B) := by
  ext idx
  simp only [ctrProd_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun s _ => by ring)

/-- Scalar multiplication pulls out of the right factor. -/
theorem ctrProd_smul_right (c : 𝕜) (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    A ⋈ₜ (c • B) = c • (A ⋈ₜ B) := by
  ext idx
  simp only [ctrProd_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun s _ => by ring)

/-- Contractive product with the zero tensor on the left is zero. -/
@[simp]
theorem zero_ctrProd (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    (0 : Tensor 𝕜 (a + 1) (Fin.snoc da S)) ⋈ₜ B = 0 := by
  ext idx
  simp [ctrProd_apply]

/-- Contractive product with the zero tensor on the right is zero. -/
@[simp]
theorem ctrProd_zero (A : Tensor 𝕜 (a + 1) (Fin.snoc da S)) :
    A ⋈ₜ (0 : Tensor 𝕜 (b + 1) (Fin.cons S db)) = 0 := by
  ext idx
  simp [ctrProd_apply]

end Tensor
