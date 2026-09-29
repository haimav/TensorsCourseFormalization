/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.NModeProd
import TensorCourse.Matrix.KhatriRao
import Mathlib.LinearAlgebra.Matrix.Kronecker

/-!
# Kronecker-n-mode Product Identity (Lecture 3, step 24)

Given a chain of n-mode products
`Y = X ×₁ A⁽¹⁾ ×₂ A⁽²⁾ ⋯ ×ₙ A⁽ᴺ⁾`, the mode-`n` unfolding of `Y` factors as

    Y_(n) = A⁽ⁿ⁾ · X_(n) · (A⁽ᴺ⁾ ⊗ ⋯ ⊗ A⁽ⁿ⁺¹⁾ ⊗ A⁽ⁿ⁻¹⁾ ⊗ ⋯ ⊗ A⁽¹⁾)ᵀ

The LHS's output type involves iterated `Function.update`s; matching it to
a clean matrix product requires heavy submatrix-casts. This file instead
proves the *scalar-valued* pointwise identity — a closed-form sum-of-
products formula that captures the full mathematical content without
intrusive matrix-type mismatches.

## Main results

* `Tensor.nModeProd_chain_2_apply_pointwise` — order-2 case: every entry
  of `(T ×₁ A) ×₂ B` equals the closed-form double sum
  `∑ i₀ i₁, A (idx 0) i₀ · B (idx 1) i₁ · T ![i₀, i₁]`. This is the
  scalar-level statement of `Y_(0) = A · X_(0) · Bᵀ`.
* `Tensor.modeUnfold_zero_chain_2_entry` — corollary reading off the
  `(i, j)`-entry of `Y = (T ×₁ A) ×₂ B` at a named multi-index.
* `Tensor.nModeProd_chain_3_apply_pointwise` — order-3 case at mode 0:
  every entry of `(T ×₁ A₀) ×₂ A₁ ×₃ A₂` equals the closed-form triple
  sum. This is the scalar-level statement of
  `Y_(0) = A₀ · X_(0) · (A₂ ⊗ A₁)ᵀ`.

The fully general form with arbitrary `D` and arbitrary `n` is deferred.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Helper cast-through lemmas -/

/-- For `dims : Fin 2 → ℕ`, the dimension at mode 0 after updating both
    modes is `J₀`. -/
private theorem update2_at_0 (dims : Fin 2 → ℕ) (J₀ J₁ : ℕ) :
    (Function.update (Function.update dims 0 J₀) 1 J₁) 0 = J₀ := by
  rw [Function.update_of_ne (by decide), Function.update_self]

/-- For `dims : Fin 2 → ℕ`, the dimension at mode 1 after updating both
    modes is `J₁`. -/
private theorem update2_at_1 (dims : Fin 2 → ℕ) (J₀ J₁ : ℕ) :
    (Function.update (Function.update dims 0 J₀) 1 J₁) 1 = J₁ :=
  Function.update_self _ _ _

/-! ### Order-2 case: pointwise identity -/

/-- **Order-2 chain of n-mode products: pointwise double-sum form.**

    For `T : Tensor 𝕜 2 dims` and matrices
    `A : Matrix (Fin J₀) (Fin (dims 0)) 𝕜`,
    `B : Matrix (Fin J₁) (Fin (dims 1)) 𝕜`, the value of `(T ×₁ A) ×₂ B`
    at an output multi-index is the double sum over the input indices.

    This is the scalar-level statement of the matrix identity
    `Y_(0) = A · T_(0) · Bᵀ`: evaluating any entry of `Y = T ×₁ A ×₂ B`
    produces the same scalar as computing the same entry from the matrix
    product `A · T_(0) · Bᵀ` (expanded as a double sum). -/
theorem nModeProd_chain_2_apply_pointwise
    {dims : Fin 2 → ℕ} {J₀ J₁ : ℕ}
    (A : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (B : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (T : Tensor 𝕜 2 dims)
    (idx : (d : Fin 2) → Fin ((Function.update
      (Function.update dims 0 J₀) 1 J₁) d)) :
    (T.nModeProd 0 A).nModeProd 1
        (B.submatrix id
          (fun k : Fin ((Function.update dims 0 J₀) 1) =>
            k.cast (Function.update_of_ne (by decide) _ dims))) idx =
      ∑ i₀ : Fin (dims 0), ∑ i₁ : Fin (dims 1),
        A ((idx 0).cast (update2_at_0 dims J₀ J₁)) i₀ *
          B ((idx 1).cast (update2_at_1 dims J₀ J₁)) i₁ *
            T (fun d => match d with
              | ⟨0, _⟩ => i₀
              | ⟨1, _⟩ => i₁) := by
  -- Outer `×₂ B`: expand via `nModeProd_apply`. Sum is over `Fin (dims 1)`.
  rw [nModeProd_apply]
  simp only [Matrix.submatrix_apply, id]
  -- Inner is `(T ×₁ A) (fun d => if d = 1 then i₁ else (idx d).cast _)`.
  -- Apply `nModeProd_apply` again inside the sum. Sum is over `Fin (dims 0)`.
  simp_rw [nModeProd_apply, Finset.mul_sum]
  -- Swap summation order (Fin (dims 1)-after-update ↔ Fin (dims 0)).
  conv_lhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i₀ _ => ?_)
  -- The inner sum on LHS is over `Fin ((update dims 0 J₀) 1)` while the
  -- target sum is over `Fin (dims 1)`; bridge via `Finset.sum_bij` using
  -- `Fin.cast (Function.update_of_ne (by decide) J₀ dims)`.
  apply Finset.sum_bij
    (fun (x : Fin ((Function.update dims 0 J₀) 1)) _ =>
      x.cast (Function.update_of_ne (by decide) J₀ dims))
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨b.cast (Function.update_of_ne (by decide) J₀ dims).symm,
      Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  intro i₁ _
  -- Simplify the `if h : 0 = 1` → false branch on LHS (makes A's
  -- row-index `(idx 0).cast _`).
  simp only [show (0 : Fin 2) ≠ 1 from by decide, ↓reduceDIte]
  -- Rearrange LHS to `A' * B' * T` form matching RHS.
  rw [show ∀ (a b c : 𝕜), b * (a * c) = a * b * c from fun _ _ _ => by ring]
  -- Split into (A * B) · T factor matches; then within `A * B`, split A and B.
  have hT_eq : (T fun d => if h : d = 0 then h ▸ i₀
      else Fin.cast (Function.update_of_ne h J₀ dims)
        (if h' : d = 1 then h' ▸ i₁ else Fin.cast
          (by rw [Function.update_of_ne h', Function.update_of_ne h]) (idx d))) =
      T (fun d : Fin 2 => match d with
        | ⟨0, _⟩ => i₀
        | ⟨1, _⟩ => i₁.cast (Function.update_of_ne
            (show (1 : Fin 2) ≠ 0 by decide) J₀ dims)) := by
    congr 1
    funext d
    match d with
    | ⟨0, _⟩ => simp
    | ⟨1, _⟩ =>
        have h₁₁ : (⟨1, by decide⟩ : Fin 2) = 1 := rfl
        simp [h₁₁]
  rw [hT_eq]
  -- The residual scalar equation closes by `congr 2` — both sides match
  -- up to a composed `Fin.cast` on A's row-index, which `congr` peels off
  -- to `rfl`.
  congr 2

/-! ### Corollary: `(T ×₁ A ×₂ B).modeUnfold 0` factors as `A · T_(0) · Bᵀ`

Combining `nModeProd_chain_2_apply_pointwise` with an entry-by-entry
expansion of the `modeUnfold` on the matrix-side gives the scalar-level
statement `Y_(0) i j = (A · X_(0) · Bᵀ) i j`. The LHS entry is retrieved
by evaluating `Y` at a multi-index whose mode-0 component is `i` and
mode-1 component is `j`.
-/

/-- Entry-by-entry reading of the mode-0 unfolding of `T ×₁ A ×₂ B`
    at an output multi-index. The RHS is the closed-form double sum. -/
theorem modeUnfold_zero_chain_2_entry
    {dims : Fin 2 → ℕ} {J₀ J₁ : ℕ}
    (A : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (B : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (T : Tensor 𝕜 2 dims) (i : Fin J₀) (j : Fin J₁) :
    ((T.nModeProd 0 A).nModeProd 1
      (B.submatrix id
        (fun k : Fin ((Function.update dims 0 J₀) 1) =>
          k.cast (Function.update_of_ne (by decide) _ dims))))
        (fun d => match d with
          | ⟨0, isLt⟩ => i.cast (by
              have h0 : (⟨0, isLt⟩ : Fin 2) = 0 := Fin.ext rfl
              rw [h0, Function.update_of_ne (by decide), Function.update_self])
          | ⟨1, isLt⟩ => j.cast (by
              have h1 : (⟨1, isLt⟩ : Fin 2) = 1 := Fin.ext rfl
              rw [h1, Function.update_self])) =
      ∑ i₀ : Fin (dims 0), ∑ i₁ : Fin (dims 1),
        A i i₀ * B j i₁ *
          T (fun d => match d with
            | ⟨0, _⟩ => i₀
            | ⟨1, _⟩ => i₁) := by
  -- Directly follows from `nModeProd_chain_2_apply_pointwise` once the
  -- multi-index is applied: `idx 0 = i.cast _` and `idx 1 = j.cast _`
  -- flatten to `i` and `j` (up to cast) because the composite casts go
  -- to the same `Fin` type (`Fin J₀` and `Fin J₁`).
  exact nModeProd_chain_2_apply_pointwise A B T _

/-! ### Order-3 Kronecker-n-mode identity at mode 0

Pointwise statement for order-3 tensors. The identity's mathematical
content is the triple sum expansion, which matches
`A₀ · T_(0) · (A₂ ⊗ A₁)ᵀ` entry-for-entry.
-/

/-- Dimension-lookup lemmas for the thrice-updated `dims : Fin 3 → ℕ`. -/
private theorem update3_at_0 (dims : Fin 3 → ℕ) (J₀ J₁ J₂ : ℕ) :
    (Function.update (Function.update
      (Function.update dims 0 J₀) 1 J₁) 2 J₂) 0 = J₀ := by
  rw [Function.update_of_ne (by decide),
      Function.update_of_ne (by decide),
      Function.update_self]

private theorem update3_at_1 (dims : Fin 3 → ℕ) (J₀ J₁ J₂ : ℕ) :
    (Function.update (Function.update
      (Function.update dims 0 J₀) 1 J₁) 2 J₂) 1 = J₁ := by
  rw [Function.update_of_ne (by decide), Function.update_self]

private theorem update3_at_2 (dims : Fin 3 → ℕ) (J₀ J₁ J₂ : ℕ) :
    (Function.update (Function.update
      (Function.update dims 0 J₀) 1 J₁) 2 J₂) 2 = J₂ :=
  Function.update_self _ _ _

/-- **Order-3 chain of n-mode products: pointwise triple-sum form.**

    For `T : Tensor 𝕜 3 dims` and matrices
    `A₀ : Fin J₀ × Fin (dims 0)`,
    `A₁ : Fin J₁ × Fin (dims 1)`,
    `A₂ : Fin J₂ × Fin (dims 2)`,
    the value of `((T ×₁ A₀) ×₂ A₁) ×₃ A₂` at any output multi-index is
    the triple sum `∑ i₀ i₁ i₂, A₀ · A₁ · A₂ · T`.

    This is the scalar-level content of the matrix identity
    `Y_(0) = A₀ · T_(0) · (A₂ ⊗ A₁)ᵀ`; once the matrix product and
    Kronecker product on the right are expanded, they produce exactly this
    triple sum. -/
theorem nModeProd_chain_3_apply_pointwise
    {dims : Fin 3 → ℕ} {J₀ J₁ J₂ : ℕ}
    (A₀ : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (A₁ : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (A₂ : Matrix (Fin J₂) (Fin (dims 2)) 𝕜)
    (T : Tensor 𝕜 3 dims)
    (idx : (d : Fin 3) → Fin ((Function.update
      (Function.update (Function.update dims 0 J₀) 1 J₁) 2 J₂) d)) :
    (((T.nModeProd 0 A₀).nModeProd 1
          (A₁.submatrix id
            (fun k : Fin ((Function.update dims 0 J₀) 1) =>
              k.cast (Function.update_of_ne (by decide) _ dims)))).nModeProd 2
          (A₂.submatrix id
            (fun k : Fin ((Function.update
                (Function.update dims 0 J₀) 1 J₁) 2) =>
              k.cast (by
                rw [Function.update_of_ne (by decide) _,
                    Function.update_of_ne (by decide) _])))) idx =
      ∑ i₀ : Fin (dims 0), ∑ i₁ : Fin (dims 1), ∑ i₂ : Fin (dims 2),
        A₀ ((idx 0).cast (update3_at_0 dims J₀ J₁ J₂)) i₀ *
          A₁ ((idx 1).cast (update3_at_1 dims J₀ J₁ J₂)) i₁ *
            A₂ ((idx 2).cast (update3_at_2 dims J₀ J₁ J₂)) i₂ *
            T (fun d => match d with
              | ⟨0, _⟩ => i₀
              | ⟨1, _⟩ => i₁
              | ⟨2, _⟩ => i₂) := by
  -- Expand outer `×₃ A₂`.
  rw [nModeProd_apply]
  simp only [Matrix.submatrix_apply, id]
  -- Expand middle `×₂ A₁`.
  simp_rw [nModeProd_apply (n := 1), Matrix.submatrix_apply, id,
    Finset.mul_sum]
  -- Expand innermost `×₁ A₀`.
  simp_rw [nModeProd_apply (n := 0), Finset.mul_sum]
  -- LHS summation order: (x = ×₃, y = ×₂, i = ×₁) over
  -- (Fin ((upd²) 2), Fin ((upd¹) 1), Fin (dims 0)). Reverse to
  -- (i, y, x) over (Fin (dims 0), Fin ((upd¹) 1), Fin ((upd²) 2)), then
  -- bridge the last two via `sum_bij`.
  -- Step 1: swap outer two (x, y) → (y, x).
  conv_lhs => rw [Finset.sum_comm]
  -- Step 2: inside each `y`-summand, swap inner two (x, i) → (i, x).
  conv_lhs =>
    enter [2, y]
    rw [Finset.sum_comm]
  -- Step 3: swap outer two (y, i) → (i, y).
  conv_lhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i₀ _ => ?_)
  -- Bridge `Fin ((update dims 0 J₀) 1)` ↔ `Fin (dims 1)`.
  apply Finset.sum_bij
    (fun (x : Fin ((Function.update dims 0 J₀) 1)) _ =>
      x.cast (Function.update_of_ne (by decide) J₀ dims))
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨b.cast (Function.update_of_ne (by decide) J₀ dims).symm,
      Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  intro i₁ _
  -- Bridge `Fin ((update (update dims 0 J₀) 1 J₁) 2)` ↔ `Fin (dims 2)`.
  have hdim2 : (Function.update (Function.update dims 0 J₀) 1 J₁) 2 = dims 2 := by
    rw [Function.update_of_ne (show (2 : Fin 3) ≠ 1 by decide),
        Function.update_of_ne (show (2 : Fin 3) ≠ 0 by decide)]
  apply Finset.sum_bij
    (fun (x : Fin ((Function.update (Function.update dims 0 J₀) 1 J₁) 2)) _ =>
      x.cast hdim2)
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨b.cast hdim2.symm, Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  intro i₂ _
  -- At this point LHS has nested `dif`s on `d = 2`, `d = 1`, `d = 0`.
  -- Simplify each in turn using the decidable (dis)equalities of Fin 3
  -- constants.
  simp only [show (0 : Fin 3) ≠ 2 from by decide,
    show (0 : Fin 3) ≠ 1 from by decide,
    show (1 : Fin 3) ≠ 2 from by decide,
    ↓reduceDIte]
  -- Rewrite the T-multi-index into the match form first (as in D=2).
  have hT_eq :
      (T fun d =>
        if h : d = 0 then h ▸ i₀
        else Fin.cast (Function.update_of_ne h J₀ dims)
          (if h' : d = 1 then h' ▸ i₁ else Fin.cast
            (Function.update_of_ne h' J₁ (Function.update dims 0 J₀))
            (if h'' : d = 2 then h'' ▸ i₂
             else Fin.cast (by
               rw [Function.update_of_ne h'', Function.update_of_ne h',
                   Function.update_of_ne h]) (idx d)))) =
      T (fun d : Fin 3 => match d with
        | ⟨0, _⟩ => i₀
        | ⟨1, _⟩ => i₁.cast (Function.update_of_ne
            (show (1 : Fin 3) ≠ 0 by decide) J₀ dims)
        | ⟨2, isLt⟩ => i₂.cast (by
            have h2 : (⟨2, isLt⟩ : Fin 3) = 2 := Fin.ext rfl
            rw [h2, Function.update_of_ne (show (2 : Fin 3) ≠ 1 by decide),
                Function.update_of_ne (show (2 : Fin 3) ≠ 0 by decide)])) := by
    congr 1
    funext d
    match d with
    | ⟨0, _⟩ => simp
    | ⟨1, _⟩ =>
        have h₁₁ : (⟨1, by decide⟩ : Fin 3) = 1 := rfl
        simp [h₁₁]
    | ⟨2, isLt⟩ =>
        have h₂₂ : (⟨2, isLt⟩ : Fin 3) = 2 := Fin.ext rfl
        simp only [h₂₂, ↓reduceDIte]
        apply Fin.ext; rfl
  rw [hT_eq]
  -- Rearrange LHS factor-product: `A₂ * (A₁ * (A₀ * T))` → `A₀ * A₁ * A₂ * T`.
  rw [show ∀ (a b c t : 𝕜), c * (b * (a * t)) = a * b * c * t
      from fun _ _ _ _ => by ring]
  -- Residual: same scalar expression up to composed-cast discrepancies on
  -- A₀ and A₁ row-indices (A₂ already matches exactly).
  congr 3

/-! ### Entry-level corollaries for the order-3 chain

Three corollaries of `nModeProd_chain_3_apply_pointwise`, each specialising
the multi-index to fix one of the three modes at a named index. These
read off the triple-sum closed form at a specific entry, analogous to
`modeUnfold_zero_chain_2_entry` in the order-2 case.
-/

/-- Entry-by-entry reading of the chain `((T ×₁ A₀) ×₂ A₁) ×₃ A₂` at a
    multi-index whose mode-0 component is `i`, mode-1 component is `j`,
    and mode-2 component is `k`. The RHS is the closed-form triple sum. -/
theorem modeUnfold_zero_chain_3_entry
    {dims : Fin 3 → ℕ} {J₀ J₁ J₂ : ℕ}
    (A₀ : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (A₁ : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (A₂ : Matrix (Fin J₂) (Fin (dims 2)) 𝕜)
    (T : Tensor 𝕜 3 dims) (i : Fin J₀) (j : Fin J₁) (k : Fin J₂) :
    (((T.nModeProd 0 A₀).nModeProd 1
          (A₁.submatrix id
            (fun k : Fin ((Function.update dims 0 J₀) 1) =>
              k.cast (Function.update_of_ne (by decide) _ dims)))).nModeProd 2
          (A₂.submatrix id
            (fun k : Fin ((Function.update
                (Function.update dims 0 J₀) 1 J₁) 2) =>
              k.cast (by
                rw [Function.update_of_ne (by decide) _,
                    Function.update_of_ne (by decide) _]))))
        (fun d => match d with
          | ⟨0, isLt⟩ => i.cast (by
              have h0 : (⟨0, isLt⟩ : Fin 3) = 0 := Fin.ext rfl
              rw [h0]; exact (update3_at_0 dims J₀ J₁ J₂).symm)
          | ⟨1, isLt⟩ => j.cast (by
              have h1 : (⟨1, isLt⟩ : Fin 3) = 1 := Fin.ext rfl
              rw [h1]; exact (update3_at_1 dims J₀ J₁ J₂).symm)
          | ⟨2, isLt⟩ => k.cast (by
              have h2 : (⟨2, isLt⟩ : Fin 3) = 2 := Fin.ext rfl
              rw [h2]; exact (update3_at_2 dims J₀ J₁ J₂).symm)) =
      ∑ i₀ : Fin (dims 0), ∑ i₁ : Fin (dims 1), ∑ i₂ : Fin (dims 2),
        A₀ i i₀ * A₁ j i₁ * A₂ k i₂ *
          T (fun d => match d with
            | ⟨0, _⟩ => i₀
            | ⟨1, _⟩ => i₁
            | ⟨2, _⟩ => i₂) := by
  exact nModeProd_chain_3_apply_pointwise A₀ A₁ A₂ T _

/-! ### Matrix-level Kronecker-n-mode identity at mode 0 for D = 3

The scalar identity `nModeProd_chain_3_apply_pointwise` already captures
the full mathematical content of
`Y_(0) = A⁽⁰⁾ · X_(0) · (A⁽²⁾ ⊗ A⁽¹⁾)ᵀ` as a triple sum. To lift this to
a matrix-level equation we need to express `remainingModesEquiv 0 dims`
on an `order-3` remaining-modes tuple explicitly. The following helper
supplies that concrete numerical formula and is the key bridge between
the mode-0 unfolding of an order-3 tensor and the Kronecker product on
modes 1 and 2.
-/

/-! ### Matrix-level Kronecker-n-mode identity for D = 3, mode 0

Lifting the pointwise-triple-sum theorem to a matrix equation, we obtain a
matrix-level version of the Kronecker-n-mode identity:

    Y_(0) = A⁽⁰⁾ · X_(0) · (A⁽²⁾ ⊗ₜ A⁽¹⁾)ᵀ

The proof reduces to an entry-level equation via `ext`, evaluates both
sides into triple sums, and matches them by unfolding `separateModeEquiv`
(the column-reindexing `Equiv`) and cancelling the round trip via
`Equiv.symm_apply_apply`, rather than appealing to a separate explicit
`.val` formula for `remainingModesEquiv 0 dims`.

Both sides are expressed as `Matrix (Fin J₀) (Fin J₂ × Fin J₁) 𝕜`: the LHS
is the `submatrix`-reindexed mode-0 unfolding of the output tensor (row
reindexed via `update3_at_0`, column reindexed via the explicit column
decomposition). The RHS uses the ordinary matrix-product and Mathlib's
`Matrix.kroneckerMap`-based Kronecker, with the middle factor
`T.modeUnfold 0` submatrixed on the column side to convert its
`Fin (prodExcept 0 dims)` column type into `Fin (dims 2) × Fin (dims 1)`. -/

/-- **Matrix-level Kronecker-n-mode identity at mode 0 for D = 3.**

    For `T : Tensor 𝕜 3 dims` and matrices `A₀, A₁, A₂` operating on modes
    0, 1, 2 respectively, the mode-0 matricization of
    `Y = (T ×₁ A₀) ×₂ A₁ ×₃ A₂` equals the matrix product
    `A₀ · T_(0) · (A₂ ⊗ₜ A₁)ᵀ`, with both sides expressed as matrices of
    shape `Fin J₀ × (Fin J₂ × Fin J₁)`.

    Proof: reduce to an entry-level equation via `ext`, evaluate the LHS
    via `modeUnfold_apply` and `nModeProd_chain_3_apply_pointwise` to get
    a triple sum, evaluate the RHS via `mul_apply` and `kroneckerMap_apply`
    to get another triple sum, and match the two by a sum reordering and
    term-wise `Fin.ext` equalities for the composed cast chains. -/
theorem modeUnfold_zero_chain_3_eq_kronecker
    {dims : Fin 3 → ℕ} {J₀ J₁ J₂ : ℕ}
    (A₀ : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (A₁ : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (A₂ : Matrix (Fin J₂) (Fin (dims 2)) 𝕜)
    (T : Tensor 𝕜 3 dims) :
    Matrix.submatrix
      ((((T.nModeProd 0 A₀).nModeProd 1
          (A₁.submatrix id
            (fun k : Fin ((Function.update dims 0 J₀) 1) =>
              k.cast (Function.update_of_ne (by decide) _ dims)))).nModeProd 2
          (A₂.submatrix id
            (fun k : Fin ((Function.update
                (Function.update dims 0 J₀) 1 J₁) 2) =>
              k.cast (by
                rw [Function.update_of_ne (by decide) _,
                    Function.update_of_ne (by decide) _])))).modeUnfold 0)
          (fun i : Fin J₀ => i.cast (update3_at_0 dims J₀ J₁ J₂).symm)
          (fun c : Fin J₂ × Fin J₁ =>
            remainingModesEquiv 0 (Function.update (Function.update
                (Function.update dims 0 J₀) 1 J₁) 2 J₂)
              (fun d : {k : Fin 3 // k ≠ 0} => match d with
                | ⟨⟨0, _⟩, h⟩ => absurd rfl h
                | ⟨⟨1, _⟩, _⟩ => c.2.cast (update3_at_1 dims J₀ J₁ J₂).symm
                | ⟨⟨2, _⟩, _⟩ => c.1.cast (update3_at_2 dims J₀ J₁ J₂).symm
                | ⟨⟨n + 3, hn⟩, _⟩ => by omega)) =
      A₀ * ((T.modeUnfold 0).submatrix id
            (fun c : Fin (dims 2) × Fin (dims 1) =>
              remainingModesEquiv 0 dims
                (fun d : {k : Fin 3 // k ≠ 0} => match d with
                  | ⟨⟨0, _⟩, h⟩ => absurd rfl h
                  | ⟨⟨1, _⟩, _⟩ => c.2
                  | ⟨⟨2, _⟩, _⟩ => c.1
                  | ⟨⟨n + 3, hn⟩, _⟩ => by omega)))
        * (A₂ ⊗ₜ A₁).transpose := by
  ext i c
  -- LHS: unfold submatrix + modeUnfold_apply.
  simp only [Matrix.submatrix_apply]
  rw [modeUnfold_apply]
  -- Collapse (remainingModesEquiv).symm ∘ remainingModesEquiv = id inside LHS.
  rw [Equiv.symm_apply_apply]
  -- Apply the chain pointwise theorem to LHS tensor evaluation.
  rw [nModeProd_chain_3_apply_pointwise]
  -- Now LHS is a triple sum `∑ i₀ i₁ i₂, A₀ _ i₀ * A₁ _ i₁ * A₂ _ i₂ * T (match)`.
  -- RHS: expand matrix-product into nested sums.
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.submatrix_apply,
    Matrix.kroneckerMap_apply, modeUnfold_apply, Equiv.symm_apply_apply, id]
  -- Split the sum over `Fin (dims 2) × Fin (dims 1)` on RHS.
  simp only [Fintype.sum_prod_type]
  -- Pull the outer `(A₂ * A₁)` factor into the inner `i₀`-sum on RHS.
  simp only [Finset.sum_mul]
  -- Unfold `separateModeEquiv.symm` and kill the decidable dite branches.
  simp only [separateModeEquiv, Equiv.coe_fn_symm_mk,
    show (1 : Fin 3) ≠ 0 from by decide,
    show (2 : Fin 3) ≠ 0 from by decide, ↓reduceDIte]
  -- Reorder LHS `(i₀, i₁, i₂)` to `(i₂, i₁, i₀)` to match RHS.
  have hreorder :
      ∀ (f : Fin (dims 0) → Fin (dims 1) → Fin (dims 2) → 𝕜),
        ((∑ i₀, ∑ i₁, ∑ i₂, f i₀ i₁ i₂ : 𝕜) = ∑ i₂, ∑ i₁, ∑ i₀, f i₀ i₁ i₂) := by
    intro f
    conv_lhs => rw [Finset.sum_comm]
    conv_lhs => enter [2, i₁]; rw [Finset.sum_comm]
    conv_lhs => rw [Finset.sum_comm]
  rw [hreorder]
  refine Finset.sum_congr rfl fun i₂ _ => ?_
  refine Finset.sum_congr rfl fun i₁ _ => ?_
  refine Finset.sum_congr rfl fun i₀ _ => ?_
  -- Rearrange factors: a*b*c*t = a*t*(c*b).
  rw [show ∀ (a b c t : 𝕜), a * b * c * t = a * t * (c * b) from
    fun _ _ _ _ => by ring]
  -- Close by term-by-term equality: Fin.ext on each composed cast chain,
  -- funext + dite-reduce on the T-argument.
  congr 1
  · -- A₀ (cast (cast i)) i₀ * T(LHS) = A₀ i i₀ * T(sep.symm ...).
    refine congrArg₂ _ ?_ ?_
    · refine congrArg₂ A₀ ?_ rfl
      apply Fin.ext; rfl
    · -- T-argument equality.
      refine congrArg T ?_
      funext d
      match d with
      | ⟨0, _⟩ => simp
      | ⟨1, isLt⟩ =>
          have h1 : (⟨1, isLt⟩ : Fin 3) ≠ 0 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h1, ↓reduceDIte]
      | ⟨2, isLt⟩ =>
          have h2 : (⟨2, isLt⟩ : Fin 3) ≠ 0 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h2, ↓reduceDIte]
      | ⟨n + 3, hn⟩ => omega

/-! ### Matrix-level Kronecker-n-mode identities for D = 3, modes 1 and 2

Analogues of `modeUnfold_zero_chain_3_eq_kronecker` at the other two modes.
The structure of each proof is identical to the mode-0 proof: reduce to
entry-level via `ext`, expand LHS and RHS into triple sums, and match the
two by unfolding `separateModeEquiv` (the column-reindexing `Equiv`) and
cancelling the round trip via `Equiv.symm_apply_apply`. The Kronecker
factorization fixes the column order, so no separate explicit `.val`
formula for `remainingModesEquiv` is needed at any of the three modes.
-/

/-- **Matrix-level Kronecker-n-mode identity at mode 1 for D = 3.**

    For `T : Tensor 𝕜 3 dims` and matrices `A₀, A₁, A₂` operating on modes
    0, 1, 2 respectively, the mode-1 matricization of
    `Y = (T ×₁ A₀) ×₂ A₁ ×₃ A₂` equals the matrix product
    `A₁ · T_(1) · (A₂ ⊗ₜ A₀)ᵀ`, with both sides expressed as matrices of
    shape `Fin J₁ × (Fin J₂ × Fin J₀)`. The Kronecker factor order
    `A₂ ⊗ A₀` follows the Kolda-Bader convention: remaining modes `{0, 2}`
    listed in reverse. -/
theorem modeUnfold_one_chain_3_eq_kronecker
    {dims : Fin 3 → ℕ} {J₀ J₁ J₂ : ℕ}
    (A₀ : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (A₁ : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (A₂ : Matrix (Fin J₂) (Fin (dims 2)) 𝕜)
    (T : Tensor 𝕜 3 dims) :
    Matrix.submatrix
      ((((T.nModeProd 0 A₀).nModeProd 1
          (A₁.submatrix id
            (fun k : Fin ((Function.update dims 0 J₀) 1) =>
              k.cast (Function.update_of_ne (by decide) _ dims)))).nModeProd 2
          (A₂.submatrix id
            (fun k : Fin ((Function.update
                (Function.update dims 0 J₀) 1 J₁) 2) =>
              k.cast (by
                rw [Function.update_of_ne (by decide) _,
                    Function.update_of_ne (by decide) _])))).modeUnfold 1)
          (fun i : Fin J₁ => i.cast (update3_at_1 dims J₀ J₁ J₂).symm)
          (fun c : Fin J₂ × Fin J₀ =>
            remainingModesEquiv 1 (Function.update (Function.update
                (Function.update dims 0 J₀) 1 J₁) 2 J₂)
              (fun d : {k : Fin 3 // k ≠ 1} => match d with
                | ⟨⟨0, _⟩, _⟩ => c.2.cast (update3_at_0 dims J₀ J₁ J₂).symm
                | ⟨⟨1, _⟩, h⟩ => absurd rfl h
                | ⟨⟨2, _⟩, _⟩ => c.1.cast (update3_at_2 dims J₀ J₁ J₂).symm
                | ⟨⟨n + 3, hn⟩, _⟩ => by omega)) =
      A₁ * ((T.modeUnfold 1).submatrix id
            (fun c : Fin (dims 2) × Fin (dims 0) =>
              remainingModesEquiv 1 dims
                (fun d : {k : Fin 3 // k ≠ 1} => match d with
                  | ⟨⟨0, _⟩, _⟩ => c.2
                  | ⟨⟨1, _⟩, h⟩ => absurd rfl h
                  | ⟨⟨2, _⟩, _⟩ => c.1
                  | ⟨⟨n + 3, hn⟩, _⟩ => by omega)))
        * (A₂ ⊗ₜ A₀).transpose := by
  ext i c
  simp only [Matrix.submatrix_apply]
  rw [modeUnfold_apply]
  rw [Equiv.symm_apply_apply]
  rw [nModeProd_chain_3_apply_pointwise]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.submatrix_apply,
    Matrix.kroneckerMap_apply, modeUnfold_apply, Equiv.symm_apply_apply, id]
  simp only [Fintype.sum_prod_type]
  simp only [Finset.sum_mul]
  simp only [separateModeEquiv, Equiv.coe_fn_symm_mk,
    show (0 : Fin 3) ≠ 1 from by decide,
    show (2 : Fin 3) ≠ 1 from by decide, ↓reduceDIte]
  -- LHS order (i₀, i₁, i₂); RHS order (i₂, i₀, i₁). Cycle via two swaps.
  have hreorder :
      ∀ (f : Fin (dims 0) → Fin (dims 1) → Fin (dims 2) → 𝕜),
        ((∑ i₀, ∑ i₁, ∑ i₂, f i₀ i₁ i₂ : 𝕜) = ∑ i₂, ∑ i₀, ∑ i₁, f i₀ i₁ i₂) := by
    intro f
    conv_lhs => enter [2, i₀]; rw [Finset.sum_comm]
    conv_lhs => rw [Finset.sum_comm]
  rw [hreorder]
  refine Finset.sum_congr rfl fun i₂ _ => ?_
  refine Finset.sum_congr rfl fun i₀ _ => ?_
  refine Finset.sum_congr rfl fun i₁ _ => ?_
  -- LHS factors: A₀ c.2 i₀ * A₁ i i₁ * A₂ c.1 i₂ * T.
  -- RHS factors: A₁ i i₁ * T * (A₂ c.1 i₂ * A₀ c.2 i₀).
  rw [show ∀ (a b c t : 𝕜), a * b * c * t = b * t * (c * a) from
    fun _ _ _ _ => by ring]
  congr 1
  · refine congrArg₂ _ ?_ ?_
    · refine congrArg₂ A₁ ?_ rfl
      apply Fin.ext; rfl
    · refine congrArg T ?_
      funext d
      match d with
      | ⟨0, isLt⟩ =>
          have h0 : (⟨0, isLt⟩ : Fin 3) ≠ 1 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h0, ↓reduceDIte]
      | ⟨1, _⟩ => simp
      | ⟨2, isLt⟩ =>
          have h2 : (⟨2, isLt⟩ : Fin 3) ≠ 1 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h2, ↓reduceDIte]
      | ⟨n + 3, hn⟩ => omega

/-- **Matrix-level Kronecker-n-mode identity at mode 2 for D = 3.**

    For `T : Tensor 𝕜 3 dims` and matrices `A₀, A₁, A₂` operating on modes
    0, 1, 2 respectively, the mode-2 matricization of
    `Y = (T ×₁ A₀) ×₂ A₁ ×₃ A₂` equals the matrix product
    `A₂ · T_(2) · (A₁ ⊗ₜ A₀)ᵀ`, with both sides expressed as matrices of
    shape `Fin J₂ × (Fin J₁ × Fin J₀)`. The Kronecker factor order
    `A₁ ⊗ A₀` follows the Kolda-Bader convention: remaining modes `{0, 1}`
    listed in reverse. -/
theorem modeUnfold_two_chain_3_eq_kronecker
    {dims : Fin 3 → ℕ} {J₀ J₁ J₂ : ℕ}
    (A₀ : Matrix (Fin J₀) (Fin (dims 0)) 𝕜)
    (A₁ : Matrix (Fin J₁) (Fin (dims 1)) 𝕜)
    (A₂ : Matrix (Fin J₂) (Fin (dims 2)) 𝕜)
    (T : Tensor 𝕜 3 dims) :
    Matrix.submatrix
      ((((T.nModeProd 0 A₀).nModeProd 1
          (A₁.submatrix id
            (fun k : Fin ((Function.update dims 0 J₀) 1) =>
              k.cast (Function.update_of_ne (by decide) _ dims)))).nModeProd 2
          (A₂.submatrix id
            (fun k : Fin ((Function.update
                (Function.update dims 0 J₀) 1 J₁) 2) =>
              k.cast (by
                rw [Function.update_of_ne (by decide) _,
                    Function.update_of_ne (by decide) _])))).modeUnfold 2)
          (fun i : Fin J₂ => i.cast (update3_at_2 dims J₀ J₁ J₂).symm)
          (fun c : Fin J₁ × Fin J₀ =>
            remainingModesEquiv 2 (Function.update (Function.update
                (Function.update dims 0 J₀) 1 J₁) 2 J₂)
              (fun d : {k : Fin 3 // k ≠ 2} => match d with
                | ⟨⟨0, _⟩, _⟩ => c.2.cast (update3_at_0 dims J₀ J₁ J₂).symm
                | ⟨⟨1, _⟩, _⟩ => c.1.cast (update3_at_1 dims J₀ J₁ J₂).symm
                | ⟨⟨2, _⟩, h⟩ => absurd rfl h
                | ⟨⟨n + 3, hn⟩, _⟩ => by omega)) =
      A₂ * ((T.modeUnfold 2).submatrix id
            (fun c : Fin (dims 1) × Fin (dims 0) =>
              remainingModesEquiv 2 dims
                (fun d : {k : Fin 3 // k ≠ 2} => match d with
                  | ⟨⟨0, _⟩, _⟩ => c.2
                  | ⟨⟨1, _⟩, _⟩ => c.1
                  | ⟨⟨2, _⟩, h⟩ => absurd rfl h
                  | ⟨⟨n + 3, hn⟩, _⟩ => by omega)))
        * (A₁ ⊗ₜ A₀).transpose := by
  ext i c
  simp only [Matrix.submatrix_apply]
  rw [modeUnfold_apply]
  rw [Equiv.symm_apply_apply]
  rw [nModeProd_chain_3_apply_pointwise]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.submatrix_apply,
    Matrix.kroneckerMap_apply, modeUnfold_apply, Equiv.symm_apply_apply, id]
  simp only [Fintype.sum_prod_type]
  simp only [Finset.sum_mul]
  simp only [separateModeEquiv, Equiv.coe_fn_symm_mk,
    show (0 : Fin 3) ≠ 2 from by decide,
    show (1 : Fin 3) ≠ 2 from by decide, ↓reduceDIte]
  -- LHS order (i₀, i₁, i₂); RHS order (i₁, i₀, i₂). Swap the outer two.
  have hreorder :
      ∀ (f : Fin (dims 0) → Fin (dims 1) → Fin (dims 2) → 𝕜),
        ((∑ i₀, ∑ i₁, ∑ i₂, f i₀ i₁ i₂ : 𝕜) = ∑ i₁, ∑ i₀, ∑ i₂, f i₀ i₁ i₂) :=
    fun _ => Finset.sum_comm
  rw [hreorder]
  refine Finset.sum_congr rfl fun i₁ _ => ?_
  refine Finset.sum_congr rfl fun i₀ _ => ?_
  refine Finset.sum_congr rfl fun i₂ _ => ?_
  -- LHS factors: A₀ c.2 i₀ * A₁ c.1 i₁ * A₂ i i₂ * T.
  -- RHS factors: A₂ i i₂ * T * (A₁ c.1 i₁ * A₀ c.2 i₀).
  rw [show ∀ (a b c t : 𝕜), a * b * c * t = c * t * (b * a) from
    fun _ _ _ _ => by ring]
  congr 1
  · refine congrArg₂ _ ?_ ?_
    · refine congrArg₂ A₂ ?_ rfl
      apply Fin.ext; rfl
    · refine congrArg T ?_
      funext d
      match d with
      | ⟨0, isLt⟩ =>
          have h0 : (⟨0, isLt⟩ : Fin 3) ≠ 2 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h0, ↓reduceDIte]
      | ⟨1, isLt⟩ =>
          have h1 : (⟨1, isLt⟩ : Fin 3) ≠ 2 := fun h => by
            have := Fin.val_eq_of_eq h; simp at this
          simp only [h1, ↓reduceDIte]
      | ⟨2, _⟩ => simp
      | ⟨n + 3, hn⟩ => omega

end Tensor
