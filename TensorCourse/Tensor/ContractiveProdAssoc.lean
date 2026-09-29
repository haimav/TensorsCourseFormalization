/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.ContractiveProd
import TensorCourse.Tensor.Reshape
import TensorCourse.Tensor.Cast

/-!
# Associativity of the Contractive Product (Lecture 6, Exercise 1)

The lecture's Exercise asks: show that the contractive product `⋈ₜ` is
associative for tensors of order `≥ 2`,

  `(A ⋈ₜ B) ⋈ₜ C = A ⋈ₜ (B ⋈ₜ C)`.

The mathematical content is a trivial Fubini exchange of the two contraction
sums. The work is entirely in the dependent types: the two parenthesisations
do not even have the same Lean type — the orders are `(a+m)+c` versus
`a+(m+c)` and the dimension functions are `combinedDims (combinedDims da dm) dc`
versus `combinedDims da (combinedDims dm dc)`. We therefore state the equality
modulo the canonical reindexings forced by the typing (`Tensor.reindex` along
the index reassociation), and the middle tensor `B` is taken of order `m+2`
(first mode the `A`-bond `S₁`, last mode the `C`-bond `S₂`) — the `≥ 2`
hypothesis of the exercise.

## Main results

* `Tensor.ctrProd_assoc` — associativity of `⋈ₜ` up to the canonical reindex.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### `combinedDims` is `Fin.append`: bridge to Mathlib's tuple lemmas -/

/-- `combinedDims` is definitionally `Fin.append`; this bridge lets us reuse
    Mathlib's `Fin.append_*` rewrite lemmas. -/
theorem combinedDims_eq_append {D₁ D₂ : ℕ} (d₁ : Fin D₁ → ℕ) (d₂ : Fin D₂ → ℕ) :
    combinedDims d₁ d₂ = Fin.append d₁ d₂ := rfl

/-- Appending a snoc on the right is a snoc of the append (no cast). -/
theorem combinedDims_snoc {D₁ D₂ : ℕ} (d₁ : Fin D₁ → ℕ) (d₂ : Fin D₂ → ℕ) (x : ℕ) :
    combinedDims d₁ (Fin.snoc d₂ x) = Fin.snoc (combinedDims d₁ d₂) x := by
  simp only [combinedDims_eq_append]; exact Fin.append_snoc d₁ d₂ x

/-- Appending a cons on the left is a cons of the append, up to the
    `add_right_comm` reindex of the position. -/
theorem combinedDims_cons {D₁ D₂ : ℕ} (x : ℕ) (d₁ : Fin D₁ → ℕ) (d₂ : Fin D₂ → ℕ) :
    combinedDims (Fin.cons x d₁) d₂
      = (Fin.cons x (combinedDims d₁ d₂)) ∘ Fin.cast (Nat.add_right_comm D₁ 1 D₂) := by
  simp only [combinedDims_eq_append]; exact Fin.append_cons x d₁ d₂

/-- Associativity of `combinedDims`, up to the `add_assoc` reindex of the
    position (this is `Fin.append_assoc`). -/
theorem combinedDims_assoc {D₁ D₂ D₃ : ℕ}
    (d₁ : Fin D₁ → ℕ) (d₂ : Fin D₂ → ℕ) (d₃ : Fin D₃ → ℕ) :
    combinedDims (combinedDims d₁ d₂) d₃
      = (combinedDims d₁ (combinedDims d₂ d₃)) ∘ Fin.cast (Nat.add_assoc D₁ D₂ D₃) := by
  simp only [combinedDims_eq_append]; exact Fin.append_assoc d₁ d₂ d₃

/-! ### A reindexing equivalence from a `Fin.cast`-compatible dim equality -/

private theorem cast_cancel {n m : ℕ} (h : n = m) (i : Fin n) :
    Fin.cast h.symm (Fin.cast h i) = i := by subst h; rfl

private theorem cast_cancel' {n m : ℕ} (h : n = m) (i : Fin m) :
    Fin.cast h (Fin.cast h.symm i) = i := by subst h; rfl

/-- The multi-index reindexing equivalence induced by a dimension equality
    `dims' i = dims (Fin.cast hD.symm i)` (the form produced by
    `combinedDims_cons`/`combinedDims_assoc`). Used to transport a tensor of
    type `Tensor 𝕜 D dims` to `Tensor 𝕜 D' dims'` via `Tensor.reindex`. -/
def castEquiv {D D' : ℕ} {dims : Fin D → ℕ} {dims' : Fin D' → ℕ}
    (hD : D = D') (hd : ∀ i : Fin D', dims' i = dims (Fin.cast hD.symm i)) :
    ((d : Fin D) → Fin (dims d)) ≃ ((d : Fin D') → Fin (dims' d)) where
  toFun f := fun i => Fin.cast (hd i).symm (f (Fin.cast hD.symm i))
  invFun g := fun d => Fin.cast (by rw [hd (Fin.cast hD d), cast_cancel]) (g (Fin.cast hD d))
  left_inv f := by
    funext d; apply Fin.ext
    simp only [Fin.val_cast]
    rw [cast_cancel]
  right_inv g := by
    funext i; apply Fin.ext
    simp only [Fin.val_cast]
    rw [cast_cancel']

@[simp]
theorem castEquiv_symm_apply {D D' : ℕ} {dims : Fin D → ℕ} {dims' : Fin D' → ℕ}
    (hD : D = D') (hd : ∀ i : Fin D', dims' i = dims (Fin.cast hD.symm i))
    (g : (d : Fin D') → Fin (dims' d)) (d : Fin D) :
    (castEquiv hD hd).symm g d
      = Fin.cast (by rw [hd (Fin.cast hD d), cast_cancel]) (g (Fin.cast hD d)) := rfl

/-! ### Associativity statement -/

variable {a m c : ℕ} {da : Fin a → ℕ} {dm : Fin m → ℕ} {dc : Fin c → ℕ} {S₁ S₂ : ℕ}

/-- The dim equality powering the `B ⋈ₜ C` order-`(m+1)+c → (m+c)+1` reindex. -/
private theorem hdR_proof (i : Fin ((m + c) + 1)) :
    (Fin.cons S₁ (combinedDims dm dc) : Fin (m + c + 1) → ℕ) i
      = combinedDims (Fin.cons S₁ dm) dc (Fin.cast (Nat.add_right_comm m 1 c).symm i) := by
  rw [combinedDims_cons]
  simp only [Function.comp_apply]
  rw [cast_cancel' (Nat.add_right_comm m 1 c) i]

/-- The dim equality powering the final `a+(m+c) → (a+m)+c` associativity reindex. -/
private theorem hdA_proof (i : Fin ((a + m) + c)) :
    combinedDims (combinedDims da dm) dc i
      = combinedDims da (combinedDims dm dc) (Fin.cast (Nat.add_assoc a m c) i) := by
  rw [combinedDims_assoc]; simp only [Function.comp_apply]

/-- **Associativity of the contractive product** (Lecture 6, Exercise 1).
    For `A` of order `a+1` (last mode `S₁`), `B` of order `m+2` (first mode
    `S₁`, last mode `S₂`), and `C` of order `c+1` (first mode `S₂`), the two
    parenthesisations of `A ⋈ₜ B ⋈ₜ C` agree, modulo the canonical reindex
    that reassociates the free modes (`Fin.append`/`combinedDims` associativity).

    The middle factor `B` has order `≥ 2`, matching the lecture's hypothesis. -/
theorem ctrProd_assoc
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S₁))
    (B : Tensor 𝕜 (m + 2) (Fin.cons S₁ (Fin.snoc dm S₂)))
    (C : Tensor 𝕜 (c + 1) (Fin.cons S₂ dc)) :
    (combinedDims_snoc da dm S₂ ▸ (A ⋈ₜ B)) ⋈ₜ C
      = reindex (castEquiv (Nat.add_assoc a m c).symm (fun i => hdA_proof i))
          (A ⋈ₜ reindex (castEquiv (Nat.add_right_comm m 1 c) (fun i => hdR_proof i))
            ((Fin.cons_snoc_eq_snoc_cons S₁ dm S₂ ▸ B) ⋈ₜ C)) := by
  funext idx
  rw [reindex_apply]
  simp only [ctrProd_apply, reindex_apply, cast_apply_dims,
    Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun s₁ _ => Finset.sum_congr rfl fun s₂ _ => ?_
  rw [mul_assoc]
  congr 1
  · -- A's argument matches
    congr 1
    funext d
    refine Fin.lastCases ?_ (fun i => ?_) d
    · apply Fin.ext; simp [Fin.val_cast]
    · apply Fin.ext
      simp only [Fin.lastCases_castSucc, splitLeft, castEquiv_symm_apply, Fin.val_cast]
      rw [show (Fin.castAdd (m + 1) i : Fin (a + m + 1))
            = Fin.castSucc (Fin.castAdd m i) from by apply Fin.ext; simp]
      simp only [Fin.lastCases_castSucc, Fin.val_cast]
      rw [show (Fin.castAdd c (Fin.castAdd m i) : Fin (a + m + c))
            = Fin.cast (Nat.add_assoc a m c).symm (Fin.castAdd (m + c) i) from by
        apply Fin.ext; simp]
  · congr 1
    · -- B's argument matches
      congr 1
      funext d
      refine Fin.cases ?_ (fun j => ?_) d
      · -- d = 0 : the S₁ bond
        apply Fin.ext
        simp only [Fin.cons_zero, Fin.val_cast]
        rw [show (0 : Fin (m + 2)) = Fin.castSucc (0 : Fin (m + 1)) from by apply Fin.ext; simp]
        simp only [Fin.lastCases_castSucc, splitLeft, castEquiv_symm_apply, Fin.val_cast]
        rw [show Fin.cast (Nat.add_right_comm m 1 c) (Fin.castAdd c (0 : Fin (m + 1)))
              = (0 : Fin (m + c + 1)) from by apply Fin.ext; simp]
        simp only [Fin.cons_zero, Fin.val_cast]
      · refine Fin.lastCases ?_ (fun j' => ?_) j
        · -- j = last m : the S₂ bond
          apply Fin.ext
          simp only [Fin.cons_succ, splitRight, Fin.val_cast]
          rw [Fin.succ_last]
          simp only [Fin.lastCases_last, Fin.val_cast]
          rw [show (Fin.natAdd a (Fin.last m) : Fin (a + m + 1)) = Fin.last (a + m) from by
            apply Fin.ext; simp]
          simp only [Fin.lastCases_last, Fin.val_cast]
        · -- j = castSucc j' : a middle mode
          apply Fin.ext
          simp only [Fin.cons_succ, splitRight, Fin.val_cast]
          rw [Fin.succ_castSucc,
            show (Fin.natAdd a (Fin.castSucc j') : Fin (a + m + 1))
              = Fin.castSucc (Fin.natAdd a j') from by apply Fin.ext; simp]
          simp only [Fin.lastCases_castSucc, splitLeft, castEquiv_symm_apply, Fin.val_cast]
          rw [show Fin.cast (Nat.add_right_comm m 1 c) (Fin.castAdd c (Fin.succ j'))
                = Fin.succ (Fin.castAdd c j') from by apply Fin.ext; simp]
          simp only [Fin.cons_succ, Fin.val_cast]
          rw [show (Fin.castAdd c (Fin.natAdd a j') : Fin (a + m + c))
                = Fin.cast (Nat.add_assoc a m c).symm (Fin.natAdd a (Fin.castAdd c j')) from by
            apply Fin.ext; simp]
    · -- C's argument matches
      congr 1
      funext d
      refine Fin.cases ?_ (fun j => ?_) d
      · apply Fin.ext; simp [Fin.cons_zero, Fin.val_cast]
      · apply Fin.ext
        simp only [Fin.cons_succ, splitRight, castEquiv_symm_apply, Fin.val_cast]
        rw [show Fin.cast (Nat.add_right_comm m 1 c) (Fin.natAdd (m + 1) j)
              = Fin.succ (Fin.natAdd m j) from by apply Fin.ext; simp; omega]
        simp only [Fin.cons_succ, Fin.val_cast]
        rw [show (Fin.natAdd (a + m) j : Fin (a + m + c))
              = Fin.cast (Nat.add_assoc a m c).symm (Fin.natAdd a (Fin.natAdd m j)) from by
          apply Fin.ext; simp; omega]

end Tensor
