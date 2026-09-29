/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.Weighted
import TensorCourse.Tensor.ModeUnfold
import TensorCourse.Matrix.KhatriRao

/-!
# CP Decomposition and Mode-`n` Matricization (Khatri-Rao Identity)

This file proves the central identity of Lecture 4 relating the mode-`n`
matricization of a tensor reconstructed from a CP decomposition to a
matrix product involving an iterated Khatri-Rao product.

## Main definitions

* `Tensor.khatriRaoExcept A n` — the Khatri-Rao product of all factor
  matrices `A d` except the `n`-th one. Defined directly via the
  closed-form pointwise formula:
  `(khatriRaoExcept A n) j r = ∏ d ∈ univ.erase n, A d (idx_at j d) r`,
  where `idx_at j` reads off the multi-index on the remaining modes via
  `(remainingModesEquiv n dims).symm j`.
  The row index `j : Fin (Tensor.prodExcept n dims)` is in the same
  Kolda-Bader column-major order as `Tensor.modeUnfold`'s columns,
  so the central identity below has *identity* row/column reindexes
  on both axes.

## Main results

* `Tensor.khatriRaoExcept_apply` — pointwise formula for `khatriRaoExcept`.
* `Tensor.cp_modeUnfold_eq_khatriRao` — the central identity:
  ```
  cp.tensor.modeUnfold n
    = Matrix.of (fun i r => cp.factors r n i) * (khatriRaoExcept A n)ᵀ
  ```
  (no `submatrix` needed — both axes match natively).
* `Tensor.cp_modeUnfold_eq_khatriRao_iff` — the iff-form of the central
  identity, using `modeUnfold_injective`.

## Submatrix reindexes

Because `khatriRaoExcept` is defined to land natively in
`Matrix (Fin (Tensor.prodExcept n dims)) (Fin R) 𝕜`, the central
identity needs **no `submatrix` reindex on either axis**: the row
dimension of `cp.tensor.modeUnfold n` is `Fin (dims n)` (matched
directly by the column-stacked mode-`n` factor matrix), and the
column dimension `Fin (Tensor.prodExcept n dims)` is matched directly
by `(khatriRaoExcept _ n)ᵀ`'s column index. This is the cleanest
possible statement of the identity.

## Ordering convention

The row order of `khatriRaoExcept A n`, viewed as a multi-index over
modes `d ≠ n`, follows the same Kolda-Bader column-major convention as
`Tensor.remainingModesEquiv`: the smallest mode index varies fastest,
the largest varies slowest. This matches the lecture's reverse-order
Khatri-Rao listing `A^(D-1) ⊙ ⋯ ⊙ A^(n+1) ⊙ A^(n-1) ⊙ ⋯ ⊙ A^(0)`.

## σ-form

The σ-form variant `cpWeighted_modeUnfold_eq_khatriRao` is provided
below, building on Phase 2's `Weighted.lean`. It states
```
(cp.tensorWeighted σ).modeUnfold n
  = cp.factorMatrix n * Matrix.diagonal σ
      * (khatriRaoExcept cp.factorMatrix n)ᵀ
```
and is proved by direct pointwise expansion (mirroring the plain-CP
proof but absorbing the extra `σ r` factor via `Matrix.mul_diagonal`).
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ} {R : ℕ}

/-! ### Definition of `khatriRaoExcept` -/

/-- The iterated Khatri-Rao product of factor matrices `A : (d : Fin D) →
    Matrix (Fin (dims d)) (Fin R) 𝕜`, omitting the `n`-th factor.

    The lecture writes this as a Khatri-Rao chain over modes `d ≠ n` in
    reverse mode order, `A^(D-1) ⊙ ⋯ ⊙ A^(n+1) ⊙ A^(n-1) ⊙ ⋯ ⊙ A^(0)`.
    We do *not* prove that our definition equals that explicit factor
    chain; instead we define it directly via the closed-form pointwise
    formula, and what we establish (in `cp_modeUnfold_eq_khatriRao`) is
    that its **row ordering matches `Tensor.modeUnfold`'s column ordering**
    — the Kolda-Bader column-major convention — so that the central
    identity has identity row/column reindexes on both axes.

    Concretely, the row index `j : Fin (Tensor.prodExcept n dims)` lies
    natively in the same Kolda-Bader column-major order as
    `Tensor.remainingModesEquiv`: inverting that equivalence at `j` gives a
    multi-index on `{k : Fin D // k ≠ n}`, which we then evaluate at each
    `d ≠ n`.

    For `d = n` the product term is `1` (`Finset.prod_dite` would discard it
    via the membership filter, but for type-correctness we still need a
    well-typed expression at `d = n`); we use `dite` to branch. -/
def khatriRaoExcept
    (A : (d : Fin D) → Matrix (Fin (dims d)) (Fin R) 𝕜) (n : Fin D) :
    Matrix (Fin (Tensor.prodExcept n dims)) (Fin R) 𝕜 :=
  Matrix.of fun (j : Fin (Tensor.prodExcept n dims)) (r : Fin R) =>
    ∏ d ∈ Finset.univ.erase n,
      (if h : d = n then (1 : 𝕜)
       else A d ((remainingModesEquiv n dims).symm j ⟨d, h⟩) r)

/-- Pointwise formula for `khatriRaoExcept`: the entry at row `j` and
    column `r` is the product over modes `d ∈ univ.erase n` of `A d (idx d) r`,
    where `idx = (remainingModesEquiv n dims).symm j` reads off the
    multi-index on the remaining modes from the column-major flat
    index `j`. -/
theorem khatriRaoExcept_apply
    (A : (d : Fin D) → Matrix (Fin (dims d)) (Fin R) 𝕜) (n : Fin D)
    (j : Fin (Tensor.prodExcept n dims)) (r : Fin R) :
    (khatriRaoExcept A n) j r =
      ∏ d ∈ Finset.univ.erase n,
        (if h : d = n then (1 : 𝕜)
         else A d ((remainingModesEquiv n dims).symm j ⟨d, h⟩) r) :=
  rfl


/-! ### The central CP-Khatri-Rao identity -/

/-- The mode-`n` factor matrix of a CP decomposition: rows indexed by
    `Fin (dims n)`, columns indexed by `Fin R`. -/
def CPDecomp.factorMatrix (cp : CPDecomp 𝕜 D dims R) (n : Fin D) :
    Matrix (Fin (dims n)) (Fin R) 𝕜 :=
  Matrix.of fun (i : Fin (dims n)) (r : Fin R) => cp.factors r n i

@[simp]
theorem CPDecomp.factorMatrix_apply (cp : CPDecomp 𝕜 D dims R) (n : Fin D)
    (i : Fin (dims n)) (r : Fin R) :
    cp.factorMatrix n i r = cp.factors r n i := rfl

/-- **The CP-Khatri-Rao identity (plain CP form).** For any CP
    decomposition `cp` and any mode `n`, the mode-`n` matricization of
    `cp.tensor` equals the matrix product

    `cp.factorMatrix n * (khatriRaoExcept (cp.factorMatrix) n)ᵀ`.

    Both sides are matrices of shape
    `Fin (dims n) × Fin (Tensor.prodExcept n dims)`. No `submatrix`
    reindex is required because `khatriRaoExcept` is defined to land
    natively in `Fin (Tensor.prodExcept n dims)` rows. -/
theorem cp_modeUnfold_eq_khatriRao
    (n : Fin D) (cp : CPDecomp 𝕜 D dims R) :
    cp.tensor.modeUnfold n =
      cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ := by
  funext i j
  -- LHS: unfold via `modeUnfold_apply` and `CPDecomp.tensor`.
  rw [modeUnfold_apply]
  -- LHS is now a sum over `Fin R` of products over `Fin D`:
  --   ∑ r, ∏ d, cp.factors r d (idx d)
  -- where `idx = (separateModeEquiv n dims).symm (i, (remainingModesEquiv n dims).symm j)`.
  change (∑ r : Fin R, ∏ d : Fin D, cp.factors r d _)
    = (cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ) i j
  -- RHS: expand matrix multiplication and transpose.
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [Matrix.transpose_apply, khatriRaoExcept_apply, CPDecomp.factorMatrix_apply]
  -- LHS r-summand: ∏ d : Fin D, cp.factors r d (idx d).
  -- Split off the `n`-th term using `mul_prod_erase`:
  --   ∏ d, f d = f n * ∏ d ∈ erase n, f d.
  rw [show (∏ d : Fin D, cp.factors r d
        ((separateModeEquiv n dims).symm (i, (remainingModesEquiv n dims).symm j) d))
      = cp.factors r n
          ((separateModeEquiv n dims).symm
            (i, (remainingModesEquiv n dims).symm j) n) *
        ∏ d ∈ Finset.univ.erase n, cp.factors r d
          ((separateModeEquiv n dims).symm
            (i, (remainingModesEquiv n dims).symm j) d) from
    (Finset.mul_prod_erase Finset.univ
      (fun d => cp.factors r d
        ((separateModeEquiv n dims).symm
          (i, (remainingModesEquiv n dims).symm j) d))
      (Finset.mem_univ n)).symm]
  -- The `n`-th term simplifies: at mode `n`, `separateModeEquiv.symm` returns `i`.
  have hn : ((separateModeEquiv n dims).symm
        (i, (remainingModesEquiv n dims).symm j)) n = i := by
    -- Direct from the definition of `separateModeEquiv.invFun`.
    change (if h : n = n then h ▸ i else _) = i
    simp
  rw [hn]
  -- Now match the products term-by-term over `d ∈ erase n`.
  congr 1
  refine Finset.prod_congr rfl fun d hd => ?_
  -- For `d ≠ n`, `separateModeEquiv.symm` returns `(remainingModesEquiv.symm j) ⟨d, _⟩`.
  have hd_ne : d ≠ n := Finset.ne_of_mem_erase hd
  have hsep : ((separateModeEquiv n dims).symm
        (i, (remainingModesEquiv n dims).symm j)) d =
      (remainingModesEquiv n dims).symm j ⟨d, hd_ne⟩ := by
    change (if h : d = n then h ▸ i else _) = _
    simp [hd_ne]
  rw [hsep]
  -- The RHS dite reduces to the else branch because `d ≠ n`.
  rw [dif_neg hd_ne]
  -- The remaining goal is `cp.factors r d _ = cp.factorMatrix d _ r`,
  -- which is `factorMatrix_apply` applied symmetrically.
  rfl

/-! ### The iff-form of the central identity -/

/-- **The iff-form of the CP-Khatri-Rao identity.** A tensor `T` equals
    the reconstruction `cp.tensor` of a CP decomposition `cp` if and
    only if its mode-`n` matricization equals
    `cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ`.

    The forward direction follows from `cp_modeUnfold_eq_khatriRao`.
    The reverse direction uses `modeUnfold_injective`. -/
theorem cp_modeUnfold_eq_khatriRao_iff
    (n : Fin D) (cp : CPDecomp 𝕜 D dims R) (T : Tensor 𝕜 D dims) :
    T.modeUnfold n = cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ
      ↔ T = cp.tensor := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- From mode-n matricization equality, recover tensor equality via injectivity.
    apply modeUnfold_injective n
    rw [h, cp_modeUnfold_eq_khatriRao]
  · -- The forward direction is the central identity, after substituting `T = cp.tensor`.
    subst h
    exact cp_modeUnfold_eq_khatriRao n cp

/-! ### σ-form corollary (Phase 2 link)

When `Phase 2`'s `Weighted.lean` is available, the σ-form variant of the
central identity follows by absorbing the diagonal weights `σ`. The
proof mirrors `cp_modeUnfold_eq_khatriRao` exactly, with the extra `σ r`
factor pulled into the `i,r`-position via `Matrix.mul_diagonal`. -/

/-- **σ-form CP-Khatri-Rao identity.** For a σ-form CP reconstruction
    `cp.tensorWeighted σ = ⟦σ ;; A⁽⁰⁾, …, A⁽ᴰ⁻¹⁾⟧`, the mode-`n`
    matricization is

    `cp.factorMatrix n * Matrix.diagonal σ * (khatriRaoExcept cp.factorMatrix n)ᵀ`. -/
theorem cpWeighted_modeUnfold_eq_khatriRao
    (n : Fin D) (σ : Fin R → 𝕜) (cp : CPDecomp 𝕜 D dims R) :
    (cp.tensorWeighted σ).modeUnfold n =
      cp.factorMatrix n * Matrix.diagonal σ
        * (khatriRaoExcept cp.factorMatrix n)ᵀ := by
  funext i j
  -- LHS: unfold via `modeUnfold_apply` and `tensorWeighted_apply`.
  rw [modeUnfold_apply]
  rw [CPDecomp.tensorWeighted_apply]
  -- LHS is now `∑ r, σ r * ∏ d, cp.factors r d (idx d)`.
  -- RHS: expand both matrix multiplications.
  have hRHS_eq : (cp.factorMatrix n * Matrix.diagonal σ
        * (khatriRaoExcept cp.factorMatrix n)ᵀ) i j =
      ∑ r : Fin R, (cp.factors r n i * σ r) *
        (khatriRaoExcept cp.factorMatrix n) j r := by
    rw [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Matrix.transpose_apply, Matrix.mul_diagonal, CPDecomp.factorMatrix_apply]
  rw [hRHS_eq]
  -- Match summands: σ r * ∏ d, factors r d _ = (factors r n i * σ r) * khatriRao
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [khatriRaoExcept_apply]
  -- Split off the n-th term in the product.
  rw [show (∏ d : Fin D, cp.factors r d
        ((separateModeEquiv n dims).symm (i, (remainingModesEquiv n dims).symm j) d))
      = cp.factors r n
          ((separateModeEquiv n dims).symm
            (i, (remainingModesEquiv n dims).symm j) n) *
        ∏ d ∈ Finset.univ.erase n, cp.factors r d
          ((separateModeEquiv n dims).symm
            (i, (remainingModesEquiv n dims).symm j) d) from
    (Finset.mul_prod_erase Finset.univ
      (fun d => cp.factors r d
        ((separateModeEquiv n dims).symm
          (i, (remainingModesEquiv n dims).symm j) d))
      (Finset.mem_univ n)).symm]
  -- Simplify the n-th coordinate of separateModeEquiv.symm to `i`.
  have hn : ((separateModeEquiv n dims).symm
        (i, (remainingModesEquiv n dims).symm j)) n = i := by
    change (if h : n = n then h ▸ i else _) = i
    simp
  rw [hn]
  -- Reorder: σ r * (factors r n i * ∏ ... ) = (factors r n i * σ r) * ∏ ...
  rw [show ∀ (a b c : 𝕜), a * (b * c) = (b * a) * c from fun a b c => by ring]
  -- Match the remaining product term-by-term.
  congr 1
  refine Finset.prod_congr rfl fun d hd => ?_
  have hd_ne : d ≠ n := Finset.ne_of_mem_erase hd
  have hsep : ((separateModeEquiv n dims).symm
        (i, (remainingModesEquiv n dims).symm j)) d =
      (remainingModesEquiv n dims).symm j ⟨d, hd_ne⟩ := by
    change (if h : d = n then h ▸ i else _) = _
    simp [hd_ne]
  rw [hsep, dif_neg hd_ne]
  rfl

/-- **The iff-form of the σ-form CP-Khatri-Rao identity.** -/
theorem cpWeighted_modeUnfold_eq_khatriRao_iff
    (n : Fin D) (σ : Fin R → 𝕜) (cp : CPDecomp 𝕜 D dims R)
    (T : Tensor 𝕜 D dims) :
    T.modeUnfold n = cp.factorMatrix n * Matrix.diagonal σ
        * (khatriRaoExcept cp.factorMatrix n)ᵀ
      ↔ T = cp.tensorWeighted σ := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · apply modeUnfold_injective n
    rw [h, cpWeighted_modeUnfold_eq_khatriRao]
  · subst h
    exact cpWeighted_modeUnfold_eq_khatriRao n σ cp

/-! ### Order-3 sanity check at all three modes

The plan ("stepping stone: prove the order-3 special case first") asks for
the order-3 case of the central identity at each of the three modes,
validating that the Kolda-Bader column reindex inside `khatriRaoExcept`
produces the same column ordering as `Tensor.remainingModesEquiv` at
order 3 (and hence agrees with `modeUnfold_*_chain_3_eq_kronecker` in
`Tensor/NModeKronecker.lean` up to the matrix-vs-CP setting). The
order-3 cases are obtained as direct specializations of the general-`D`
identity, since the general-`D` proof was tractable. -/

/-- Order-3 specialization of `cp_modeUnfold_eq_khatriRao` at mode 0. -/
theorem cp_modeUnfold_eq_khatriRao_three_zero
    {dims : Fin 3 → ℕ} {R : ℕ} (cp : CPDecomp 𝕜 3 dims R) :
    cp.tensor.modeUnfold 0 =
      cp.factorMatrix 0 * (khatriRaoExcept cp.factorMatrix 0)ᵀ :=
  cp_modeUnfold_eq_khatriRao 0 cp

/-- Order-3 specialization of `cp_modeUnfold_eq_khatriRao` at mode 1. -/
theorem cp_modeUnfold_eq_khatriRao_three_one
    {dims : Fin 3 → ℕ} {R : ℕ} (cp : CPDecomp 𝕜 3 dims R) :
    cp.tensor.modeUnfold 1 =
      cp.factorMatrix 1 * (khatriRaoExcept cp.factorMatrix 1)ᵀ :=
  cp_modeUnfold_eq_khatriRao 1 cp

/-- Order-3 specialization of `cp_modeUnfold_eq_khatriRao` at mode 2. -/
theorem cp_modeUnfold_eq_khatriRao_three_two
    {dims : Fin 3 → ℕ} {R : ℕ} (cp : CPDecomp 𝕜 3 dims R) :
    cp.tensor.modeUnfold 2 =
      cp.factorMatrix 2 * (khatriRaoExcept cp.factorMatrix 2)ᵀ :=
  cp_modeUnfold_eq_khatriRao 2 cp

end Tensor
