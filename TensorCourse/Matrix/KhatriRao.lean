/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.Hadamard
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Khatri-Rao, Kronecker, and Hadamard Products (notation aliases)

This file collects the three "unorthodox" matrix products that appear in
Lecture 3:

* The **Kronecker product** `A ⊗ₜ B`, a thin notation alias for
  Mathlib's `Matrix.kroneckerMap (· * ·)`.
* The **Hadamard product** `A ∗ₜ B`, a thin notation alias for
  Mathlib's `Matrix.hadamard`.
* The **Khatri-Rao product** `A ⊙ₜ B` (column-wise Kronecker), defined
  here as `Matrix.khatriRao` because Mathlib does not provide it.

The `ₜ`-subscript notations follow the project-wide convention (see
`CLAUDE.md`): every operator we introduce carries a `ₜ` subscript on
the base symbol used in the lecture. They coexist with Mathlib's
`⊗ₖ` (scoped to `Kronecker`) and `⊙` (scoped to `Matrix`), which
remain available via `open scoped Kronecker` / `open scoped Matrix`.

## Ordering convention (Kolda-Bader, row-fastest)

`Matrix.khatriRao` uses the same column-major / row-fastest convention
as `Matrix.vectorize` (see `Tensor/Vectorize.lean` and
`Matrix/Vectorize.lean`). Concretely, for `A : Fin M × Fin R` and
`B : Fin K × Fin R`, the entry of the column-`r` Khatri-Rao column at
flat row index `i = k + K * m` is

    (A ⊙ₜ B) i r = A m r * B k r

i.e. the `k`-th (row of `B`) index varies fastest within the Kronecker
block, matching the row-fastest convention in use throughout the
project. This is the same convention that makes the identity
`u ⊗ₜ v = vec(v ◯ₜ u)` hold (with `v` listed before `u` on the
right-hand side — see `Matrix/KroneckerLemmas.lean`).

## Main definitions

* `Matrix.khatriRao A B` — column-wise Kronecker product with shape
  `Fin (M * K) × Fin R`.
* Scoped notations `A ⊗ₜ B`, `A ∗ₜ B`, `A ⊙ₜ B`.

## Main results

* `Matrix.khatriRao_apply` — pointwise formula for `A ⊙ₜ B`.
* `Matrix.khatriRao_assoc` — associativity, modulo a `Fin.cast`
  through `Nat.mul_assoc` on the row dimension.
* `Matrix.khatriRao_eq_kronecker_of_one_col` — for single-column
  matrices (i.e. vectors), Khatri-Rao and (flattened) Kronecker
  coincide.
-/

noncomputable section

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Thin notation aliases for Mathlib's Kronecker and Hadamard -/

/-- `A ⊗ₜ B` denotes Mathlib's Kronecker product `Matrix.kroneckerMap (· * ·)`.
    Scoped to the `Matrix` namespace. Coexists with Mathlib's own `⊗ₖ`
    (scoped to `Kronecker`); the two are interchangeable at the level of
    the underlying definition. -/
scoped[Matrix] infixl:70 " ⊗ₜ " => Matrix.kroneckerMap (· * ·)

/-- `A ∗ₜ B` denotes Mathlib's Hadamard product `Matrix.hadamard` (elementwise
    product of two matrices with the same shape). Scoped to the `Matrix`
    namespace. Coexists with Mathlib's own scoped `⊙` for `Matrix.hadamard`. -/
scoped[Matrix] infixl:70 " ∗ₜ " => Matrix.hadamard

/-! ### Khatri-Rao product -/

/-- Khatri-Rao product of two matrices with matching column count:
    column-wise Kronecker product.

    For `A : Matrix (Fin M) (Fin R) 𝕜` and `B : Matrix (Fin K) (Fin R) 𝕜`,
    the Khatri-Rao product `A ⊙ₜ B` has shape `Fin (M * K) × Fin R`. At
    row `i : Fin (M * K)` and column `r : Fin R`, the entry is

      (A ⊙ₜ B) i r = A m r * B k r

    where `(m, k) = finProdFinEquiv.symm i`, i.e. `m = i / K` and
    `k = i % K`. This is the row-fastest convention (row of `B`
    varies fastest within the Kronecker block), matching the
    column-major Kolda-Bader convention used by `Matrix.vectorize`. -/
def khatriRao {M K R : ℕ}
    (A : Matrix (Fin M) (Fin R) 𝕜) (B : Matrix (Fin K) (Fin R) 𝕜) :
    Matrix (Fin (M * K)) (Fin R) 𝕜 :=
  fun i j =>
    A (finProdFinEquiv.symm i).1 j * B (finProdFinEquiv.symm i).2 j

@[inherit_doc]
scoped[Matrix] infixl:70 " ⊙ₜ " => Matrix.khatriRao

/-- Pointwise formula for the Khatri-Rao product. -/
@[simp]
theorem khatriRao_apply {M K R : ℕ}
    (A : Matrix (Fin M) (Fin R) 𝕜) (B : Matrix (Fin K) (Fin R) 𝕜)
    (i : Fin (M * K)) (j : Fin R) :
    (A ⊙ₜ B) i j =
      A (finProdFinEquiv.symm i).1 j * B (finProdFinEquiv.symm i).2 j :=
  rfl

/-- Convenient "pair-form" of `khatriRao_apply`: evaluate `A ⊙ₜ B` at a
    row index built via `finProdFinEquiv (m, k)`. -/
theorem khatriRao_apply_pair {M K R : ℕ}
    (A : Matrix (Fin M) (Fin R) 𝕜) (B : Matrix (Fin K) (Fin R) 𝕜)
    (m : Fin M) (k : Fin K) (j : Fin R) :
    (A ⊙ₜ B) (finProdFinEquiv (m, k)) j = A m j * B k j := by
  simp [khatriRao_apply]

/-! ### Associativity of Khatri-Rao -/

/-- Associativity of Khatri-Rao, modulo a `Fin.cast` through
    `Nat.mul_assoc` on the row dimension.

    For `A : M×R`, `B : K×R`, `C : L×R`, both sides have entries
    `A m j * B k j * C l j`, but the underlying row types are
    `Fin ((M * K) * L)` on the left and `Fin (M * (K * L))` on the
    right. The cast `Fin.cast (Nat.mul_assoc M K L).symm` reindexes
    the right-hand-side row. -/
theorem khatriRao_assoc {M K L R : ℕ}
    (A : Matrix (Fin M) (Fin R) 𝕜)
    (B : Matrix (Fin K) (Fin R) 𝕜)
    (C : Matrix (Fin L) (Fin R) 𝕜) :
    (A ⊙ₜ B) ⊙ₜ C =
      ((A ⊙ₜ (B ⊙ₜ C)).submatrix (Fin.cast (Nat.mul_assoc M K L)) id) := by
  funext i j
  simp only [khatriRao_apply, submatrix_apply, id_eq,
    finProdFinEquiv_symm_apply, Fin.divNat, Fin.modNat, Fin.val_cast]
  -- After unfolding, the LHS is
  --   A ⟨(i / L) / K, _⟩ j * B ⟨(i / L) % K, _⟩ j * C ⟨i % L, _⟩ j
  -- and the RHS is
  --   A ⟨i / (K * L), _⟩ j * B ⟨(i % (K * L)) / L, _⟩ j * C ⟨(i % (K * L)) % L, _⟩ j.
  -- The three matching nat-value equalities:
  have h1 : (i.val / L) / K = i.val / (K * L) := by
    rw [Nat.div_div_eq_div_mul]
    congr 1
    exact Nat.mul_comm _ _
  have h2 : (i.val / L) % K = (i.val % (K * L)) / L :=
    (Nat.mod_mul_left_div_self _ _ _).symm
  have h3 : i.val % L = (i.val % (K * L)) % L := by
    -- `i.val % (K * L) % L = i.val % L` by `Nat.mod_mod_of_dvd`.
    exact (Nat.mod_mod_of_dvd _ ⟨K, by ring⟩).symm
  -- Use `mul_assoc` on the ambient product, then three `Fin.ext` rewrites.
  rw [show (A ⟨(i.val / L) / K, _⟩ j * B ⟨(i.val / L) % K, _⟩ j) * C ⟨i.val % L, _⟩ j
        = A ⟨(i.val / L) / K, _⟩ j
            * (B ⟨(i.val / L) % K, _⟩ j * C ⟨i.val % L, _⟩ j) from mul_assoc _ _ _]
  congr 1
  · exact congrArg (fun x => A x j) (Fin.ext h1)
  · congr 1
    · exact congrArg (fun x => B x j) (Fin.ext h2)
    · exact congrArg (fun x => C x j) (Fin.ext h3)

/-! ### Khatri-Rao equals Kronecker for single-column matrices -/

/-- For single-column matrices (i.e. column vectors), the Khatri-Rao
    product equals the Kronecker product flattened to a single column
    via `finProdFinEquiv` on the row index and the canonical
    `Fin 1 × Fin 1 ≃ Fin 1` on the column index.

    This captures the lecture identity
    `a ⊙ b = a ⊗ b` when `a, b` are vectors. -/
theorem khatriRao_eq_kronecker_of_one_col {M K : ℕ}
    (u : Matrix (Fin M) (Fin 1) 𝕜) (v : Matrix (Fin K) (Fin 1) 𝕜) :
    u ⊙ₜ v =
      (u ⊗ₜ v).submatrix finProdFinEquiv.symm
        (finProdFinEquiv.symm ∘ Fin.cast (Nat.mul_one 1).symm) := by
  funext i j
  simp only [khatriRao_apply, submatrix_apply, Function.comp_apply,
    kroneckerMap_apply]
  -- The column factor is trivial: both `j.1` and `j.2` are `⟨0, _⟩`
  -- because they live in `Fin 1`.
  have hj1 :
      (finProdFinEquiv.symm (Fin.cast (Nat.mul_one 1).symm j)).1 = j :=
    Subsingleton.elim _ _
  have hj2 :
      (finProdFinEquiv.symm (Fin.cast (Nat.mul_one 1).symm j)).2 = j :=
    Subsingleton.elim _ _
  rw [hj1, hj2]

/-! ### Khatri-Rao Gram identity

The Gram matrix of a Khatri-Rao product factorises as the Hadamard
product of the individual Gram matrices:

    (A ⊙ₜ B)ᵀ * (A ⊙ₜ B) = (Aᵀ * A) ∗ₜ (Bᵀ * B)

This is the key identity behind CP ALS algorithms (Lecture 4).
-/

/-- Khatri-Rao Gram identity:
    `(A ⊙ₜ B)ᵀ * (A ⊙ₜ B) = (Aᵀ * A) ∗ₜ (Bᵀ * B)`,
    where `∗ₜ` is the Hadamard product.

    The LHS sums `A m p * B k p * A m q * B k q` over `(m, k) : Fin M × Fin K`.
    The double sum factorises as `(∑ m, A m p * A m q) * (∑ k, B k p * B k q)`,
    which is the Hadamard product of the two Gram matrices `Aᵀ * A` and
    `Bᵀ * B`.

    **The transpose `ᵀ` (not the conjugate transpose `ᴴ`) is intentional.**
    This is the real-form CP-ALS Gram identity used in Lecture 4
    (`plan_lecture4.md`); over `ℝ` the two coincide. Switching to `ᴴ`
    would change the semantics over `ℂ` (introducing conjugation on one
    factor of each product) and would break the `ring`-based factorisation
    proof below, which relies on the entries being plain products with no
    conjugation. -/
theorem khatriRao_transpose_mul_self {M K R : ℕ}
    (A : Matrix (Fin M) (Fin R) 𝕜) (B : Matrix (Fin K) (Fin R) 𝕜) :
    (A ⊙ₜ B)ᵀ * (A ⊙ₜ B) = (Aᵀ * A) ∗ₜ (Bᵀ * B) := by
  funext p q
  -- Expand the matrix-multiply on the LHS and reindex via `finProdFinEquiv`.
  simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.hadamard_apply,
    khatriRao_apply]
  -- Goal:
  --   ∑ i : Fin (M * K),
  --     A (finProdFinEquiv.symm i).1 p * B (finProdFinEquiv.symm i).2 p *
  --       (A (finProdFinEquiv.symm i).1 q * B (finProdFinEquiv.symm i).2 q)
  --   = (∑ m, A m p * A m q) * (∑ k, B k p * B k q)
  rw [← finProdFinEquiv.sum_comp
        (g := fun i : Fin (M * K) =>
          A (finProdFinEquiv.symm i).1 p * B (finProdFinEquiv.symm i).2 p *
            (A (finProdFinEquiv.symm i).1 q * B (finProdFinEquiv.symm i).2 q))]
  simp only [Equiv.symm_apply_apply]
  -- Now: ∑ (mk : Fin M × Fin K), A mk.1 p * B mk.2 p * (A mk.1 q * B mk.2 q)
  rw [Fintype.sum_prod_type]
  -- Goal:
  --   ∑ m, ∑ k, A m p * B k p * (A m q * B k q) = (∑ m, A m p * A m q) * ∑ k, B k p * B k q
  rw [Finset.sum_mul_sum]
  apply Finset.sum_congr rfl
  intro m _
  apply Finset.sum_congr rfl
  intro k _
  ring

end Matrix
