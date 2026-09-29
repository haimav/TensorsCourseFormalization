/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Fin.VecNotation

/-!
# CP Bracket Notation

This file introduces the lecture-standard bracket notation
`⟦A₁, A₂, …, A_D⟧` for CP decompositions. Each factor matrix `A⁽ᵈ⁾` has
row dimension `Iᵈ` (the tensor's `d`-th mode dimension) and a shared
column count `R` (the CP rank of the decomposition).

## Constraints

**D = 1 is excluded.** Mathlib provides a global unscoped
`notation3:arg "⟦" a "⟧" => Quotient.mk _ a` in
`Mathlib/Data/Quot.lean`, which claims the single-element form. Our
variadic elaborator only captures `⟦A₁, A₂, …⟧` with at least two
comma-separated factors, so `⟦A⟧` still resolves to `Quotient.mk _ A`.
Users with a single factor should build a `CPDecomp` directly.

## Implementation

We use a **variadic** `macro_rules` that expands
`⟦A₁, A₂, …, A_D⟧` into a `CPDecomp` literal whose `dims` field is
`![I₁, I₂, …, I_D]` (built from Mathlib's `vecCons`/`vecEmpty`) and
whose `factors` field is a `Fin.cases`-chain dispatching on the mode
index `d : Fin D` to the appropriate factor matrix. Shared `𝕜` and
`R` are inferred by unification across all factors; row dimensions
`Iᵈ` are inferred per-factor from each matrix's type.

The bracket notation is scoped to `Tensor` and activates with
`open scoped Tensor`.

## Main definitions

* Scoped notation `⟦A₁, A₂, …, A_D⟧` (for `D ≥ 2`) produces a
  `CPDecomp 𝕜 D ![I₁, …, I_D] R`.

## Main results

* `CPDecomp.bracket_tensor_apply₂` — beta-style reduction:
  `⟦A, B⟧.tensor idx = ∑ r, A (idx 0) r * B (idx 1) r`.
* `CPDecomp.bracket_tensor_apply₃` — analogous for three factors.
* `CPDecomp.bracket_tensor_apply₄` — analogous for four factors.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Variadic bracket notation

The notation `⟦A₁, A₂, …, A_D⟧` expands (for `D ≥ 2`) to a
`CPDecomp 𝕜 D ![I₁, …, I_D] R` whose factor dispatcher is a chain of
`Fin.cases`. The expansion is purely syntactic: types (including the
shared scalar `𝕜`, shared rank `R`, and each per-mode row dimension
`Iᵈ`) are recovered by unification when the term is elaborated. -/

/-- Build the `Fin.cases` chain dispatching `d : Fin D` to the
    appropriate factor column. Helper macro for `⟦…⟧`.

    Given factors `A₁, …, A_D` (as syntax) and an outer-scope scalar
    `r`, produces the term
    `Fin.cases (fun i => A₁ i r) (Fin.cases (fun i => A₂ i r) (… Fin.elim0))`
    of type `(d : Fin D) → Fin (![I₁,…,I_D] d) → 𝕜`. -/
syntax (name := cpBracketCases) "cpBracketCases%" "[" term,* "]" "at" term:max : term

macro_rules
  | `(cpBracketCases% [] at $_:term) => `(fun i => i.elim0)
  | `(cpBracketCases% [$a:term] at $r:term) =>
      `(Fin.cases (fun i => $a i $r) (fun j => j.elim0))
  | `(cpBracketCases% [$a:term, $as:term,*] at $r:term) =>
      `(Fin.cases (fun i => $a i $r) (cpBracketCases% [$as,*] at $r))

/-- CP bracket notation. `⟦A₁, A₂, …, A_D⟧ : CPDecomp 𝕜 D ![I₁,…,I_D] R`.

    Each factor `Aᵢ : Matrix (Fin Iᵢ) (Fin R) 𝕜` contributes column
    `r` at mode `d = i` as the vector `Aᵢ · r : Fin Iᵢ → 𝕜`. Requires
    `D ≥ 2`; `⟦A⟧` is intentionally left to Mathlib's global
    `Quotient.mk` notation.

    Internally expands to
    `{ factors := fun r => cpBracketCases% [A₁, …, A_D] at r }` with
    the ambient type annotation `CPDecomp _ _ _ _`; the dimensions
    `dims = ![I₁, …, I_D]` and the scalar/rank parameters are
    synthesized by unification. -/
scoped syntax (name := cpBrackets) "⟦" term,+ "⟧" : term

macro_rules
  | `(⟦$a:term, $bs:term,*⟧) =>
      `(({ factors := fun r => cpBracketCases% [$a, $bs,*] at r } :
          CPDecomp _ _ _ _))

/-! ### Beta-style reduction lemmas

Each factor-dispatch lookup through `⟦…⟧.factors` is definitionally
equal to the corresponding matrix column, so the reduction lemmas
below are `rfl`. They are marked `@[simp]` so that downstream
reasoning over `⟦A, B, …⟧.tensor idx` automatically unpacks the
`Fin.cases` chain. -/

section Arity2

variable {I₁ I₂ R : ℕ}
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)

@[simp] theorem bracket₂_factors_zero (r : Fin R) (i : Fin (![I₁, I₂] 0)) :
    (⟦A, B⟧ : CPDecomp 𝕜 2 ![I₁, I₂] R).factors r 0 i = A i r := rfl

@[simp] theorem bracket₂_factors_one (r : Fin R) (i : Fin (![I₁, I₂] 1)) :
    (⟦A, B⟧ : CPDecomp 𝕜 2 ![I₁, I₂] R).factors r 1 i = B i r := rfl

/-- The tensor reconstructed from `⟦A, B⟧` evaluates pointwise as a
    sum of pairwise products of factor columns. -/
theorem bracket_tensor_apply₂
    (idx : (d : Fin 2) → Fin (![I₁, I₂] d)) :
    (⟦A, B⟧ : CPDecomp 𝕜 2 ![I₁, I₂] R).tensor idx =
      ∑ r : Fin R, A (idx 0) r * B (idx 1) r := by
  simp only [CPDecomp.tensor]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 2)) = {0, 1} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_singleton]
  rfl

end Arity2

section Arity3

variable {I₁ I₂ I₃ R : ℕ}
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)
  (C : Matrix (Fin I₃) (Fin R) 𝕜)

@[simp] theorem bracket₃_factors_zero
    (r : Fin R) (i : Fin (![I₁, I₂, I₃] 0)) :
    (⟦A, B, C⟧ : CPDecomp 𝕜 3 ![I₁, I₂, I₃] R).factors r 0 i = A i r := rfl

@[simp] theorem bracket₃_factors_one
    (r : Fin R) (i : Fin (![I₁, I₂, I₃] 1)) :
    (⟦A, B, C⟧ : CPDecomp 𝕜 3 ![I₁, I₂, I₃] R).factors r 1 i = B i r := rfl

@[simp] theorem bracket₃_factors_two
    (r : Fin R) (i : Fin (![I₁, I₂, I₃] 2)) :
    (⟦A, B, C⟧ : CPDecomp 𝕜 3 ![I₁, I₂, I₃] R).factors r 2 i = C i r := rfl

/-- The tensor reconstructed from `⟦A, B, C⟧` evaluates pointwise as a
    sum of triple products of factor columns. -/
theorem bracket_tensor_apply₃
    (idx : (d : Fin 3) → Fin (![I₁, I₂, I₃] d)) :
    (⟦A, B, C⟧ : CPDecomp 𝕜 3 ![I₁, I₂, I₃] R).tensor idx =
      ∑ r : Fin R, A (idx 0) r * B (idx 1) r * C (idx 2) r := by
  simp only [CPDecomp.tensor]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 3)) = {0, 1, 2} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
    Finset.prod_singleton]
  change A (idx 0) r * (B (idx 1) r * C (idx 2) r) = _
  ring

end Arity3

section Arity4

variable {I₁ I₂ I₃ I₄ R : ℕ}
  (A : Matrix (Fin I₁) (Fin R) 𝕜) (B : Matrix (Fin I₂) (Fin R) 𝕜)
  (C : Matrix (Fin I₃) (Fin R) 𝕜) (D' : Matrix (Fin I₄) (Fin R) 𝕜)

@[simp] theorem bracket₄_factors_zero
    (r : Fin R) (i : Fin (![I₁, I₂, I₃, I₄] 0)) :
    (⟦A, B, C, D'⟧ : CPDecomp 𝕜 4 ![I₁, I₂, I₃, I₄] R).factors r 0 i = A i r := rfl

@[simp] theorem bracket₄_factors_one
    (r : Fin R) (i : Fin (![I₁, I₂, I₃, I₄] 1)) :
    (⟦A, B, C, D'⟧ : CPDecomp 𝕜 4 ![I₁, I₂, I₃, I₄] R).factors r 1 i = B i r := rfl

@[simp] theorem bracket₄_factors_two
    (r : Fin R) (i : Fin (![I₁, I₂, I₃, I₄] 2)) :
    (⟦A, B, C, D'⟧ : CPDecomp 𝕜 4 ![I₁, I₂, I₃, I₄] R).factors r 2 i = C i r := rfl

@[simp] theorem bracket₄_factors_three
    (r : Fin R) (i : Fin (![I₁, I₂, I₃, I₄] 3)) :
    (⟦A, B, C, D'⟧ : CPDecomp 𝕜 4 ![I₁, I₂, I₃, I₄] R).factors r 3 i = D' i r := rfl

/-- The tensor reconstructed from `⟦A, B, C, D⟧` evaluates pointwise as
    a sum of four-way products of factor columns. -/
theorem bracket_tensor_apply₄
    (idx : (d : Fin 4) → Fin (![I₁, I₂, I₃, I₄] d)) :
    (⟦A, B, C, D'⟧ : CPDecomp 𝕜 4 ![I₁, I₂, I₃, I₄] R).tensor idx =
      ∑ r : Fin R, A (idx 0) r * B (idx 1) r * C (idx 2) r * D' (idx 3) r := by
  simp only [CPDecomp.tensor]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (Finset.univ : Finset (Fin 4)) = {0, 1, 2, 3} from rfl]
  rw [Finset.prod_insert (by decide), Finset.prod_insert (by decide),
    Finset.prod_insert (by decide), Finset.prod_singleton]
  change A (idx 0) r * (B (idx 1) r * (C (idx 2) r * D' (idx 3) r)) = _
  ring

end Arity4

end Tensor
