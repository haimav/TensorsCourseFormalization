/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import TensorCourse.Tensor.Vectorize
import TensorCourse.Matrix.Vectorize
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# Mode-n Matricization (Unfolding) of a Tensor

The mode-n unfolding of an order-D tensor rearranges it as a matrix whose
rows are indexed by mode n and whose columns are indexed by the remaining
modes (flattened into a single index).

The unfolding is packaged as an `Equiv` (`Tensor.modeUnfoldEquiv`), with
forward direction `modeUnfold` and inverse direction `fold`. The round-trip
identities then fall out of the `Equiv` structure. The `Equiv` is further
upgraded to a `LinearEquiv` (`Tensor.modeUnfoldLinearEquiv`) giving
`map_add`, `map_smul`, `map_sum` for free.

## Main definitions

* `Tensor.separateModeEquiv n` — splits a multi-index into its `n`-th
  coordinate and the remaining coordinates (represented as a tuple indexed
  by the `n`-complement).
* `Tensor.remainingModesEquiv n` — flattens the remaining-modes tuple into
  a single `Fin` index in column-major order, using the same convention as
  `multiIndexEquiv` (mode 0 fastest).
* `Tensor.modeUnfoldEquiv dims n` — the bijection between
  `Tensor 𝕜 D dims` and
  `Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜`.
* `Tensor.modeUnfold n T` and `Tensor.fold dims n M` — the forward and
  inverse operations as named wrappers.
* `Tensor.modeUnfoldLinearEquiv dims n` — the `Equiv` upgraded to a
  `𝕜`-linear equivalence.

## Main results

* `Tensor.modeUnfold_fold`, `Tensor.fold_modeUnfold` — round-trip identities.
* `Tensor.modeUnfold_injective` — injectivity of mode-n unfolding.
* `Tensor.vectorize_eq_modeUnfold_zero` — the lecture's consistency check
  at the pointwise level: every entry of a tensor's vectorization equals an
  entry of its mode-0 matricization.
* `Tensor.vectorize_modeUnfold_zero` — the same identity lifted to a
  function-level equation: `Tensor.vectorize T` equals
  `Matrix.vectorize (T.modeUnfold 0)` precomposed with a product-
  reassociation `Fin.cast`. Both sides use the Kolda-Bader column-major
  convention (row / mode-0 varies fastest), so the identity is direct,
  with no transpose needed.

## Ordering convention

The columns of the unfolding are indexed by
`Fin (Tensor.prodExcept n dims)`, and they are ordered using the
same Kolda-Bader column-major convention as `multiIndexEquiv`: the mode of
smallest index in `Finset.univ.erase n` varies fastest, the mode of largest
index varies slowest.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-! ### Separating mode `n` from the remaining modes -/

/-- Separate mode `n` from the remaining modes of a multi-index.
    A multi-index `idx : (d : Fin D) → Fin (dims d)` is sent to the pair
    `(idx n, (fun ⟨k, _⟩ => idx k))`, viewing the remaining coordinates as
    a function on the subtype `{k : Fin D // k ≠ n}`. -/
def separateModeEquiv (n : Fin D) (dims : Fin D → ℕ) :
    ((d : Fin D) → Fin (dims d)) ≃
      Fin (dims n) × ((d : {k : Fin D // k ≠ n}) → Fin (dims d.val)) where
  toFun idx := (idx n, fun ⟨k, _⟩ => idx k)
  invFun := fun ⟨iₙ, rest⟩ d =>
    if h : d = n then h ▸ iₙ else rest ⟨d, h⟩
  left_inv := by
    intro idx
    funext d
    by_cases h : d = n
    · subst h; simp
    · simp [h]
  right_inv := by
    rintro ⟨iₙ, rest⟩
    refine Prod.ext ?_ ?_
    · simp
    · funext k
      obtain ⟨k, hk⟩ := k
      simp [hk]

/-! ### Flattening the remaining modes into a single `Fin` index -/

/-- Equivalence between the subtype `{k : Fin D // k ≠ n}` and `Fin (D-1)`,
    obtained when `D = D'+1` so that `n : Fin (D'+1)` and we can use
    `finSuccAboveEquiv`. This version is stated in the form we need. -/
private def finNeEquiv {D : ℕ} (n : Fin (D + 1)) :
    Fin D ≃ {k : Fin (D + 1) // k ≠ n} := finSuccAboveEquiv n

/-- Flatten the remaining-modes multi-index into a single `Fin` index, in
    **column-major order consistent with `multiIndexEquiv`**. The ordering
    on the remaining modes is induced by the natural order on `Fin D`
    (skipping `n`), which matches the column order used throughout the file.

    Implementation: reindex the subtype-based tuple to a `Fin (D-1)`-indexed
    tuple via `finSuccAboveEquiv`, then apply `multiIndexEquiv` on the
    resulting shape `dims ∘ n.succAbove`. The overall cardinality is
    `∏ i : Fin D, dims (n.succAbove i) = Tensor.prodExcept n dims`,
    witnessed by `Fin.prod_univ_succAbove`. -/
def remainingModesEquiv : {D : ℕ} → (n : Fin D) → (dims : Fin D → ℕ) →
    ((d : {k : Fin D // k ≠ n}) → Fin (dims d.val)) ≃
      Fin (Tensor.prodExcept n dims)
  | 0, n, _ => Fin.elim0 n
  | D + 1, n, dims =>
      -- Reindex via `finSuccAboveEquiv` to get a tuple on `Fin D`.
      let reindexSubtype :
          ((d : {k : Fin (D + 1) // k ≠ n}) → Fin (dims d.val)) ≃
            ((i : Fin D) → Fin (dims (n.succAbove i))) :=
        (Equiv.piCongrLeft (fun d : {k : Fin (D + 1) // k ≠ n} => Fin (dims d.val))
          (finNeEquiv n)).symm
      -- Flatten the `Fin D`-indexed tuple.
      let flatten :
          ((i : Fin D) → Fin (dims (n.succAbove i))) ≃
            Fin (∏ i : Fin D, dims (n.succAbove i)) :=
        multiIndexEquiv (fun i => dims (n.succAbove i))
      -- Cast through the product identity
      -- `∏ i, dims (n.succAbove i) = ∏ d ∈ univ.erase n, dims d`.
      have heq : (∏ i : Fin D, dims (n.succAbove i)) =
          ∏ d ∈ Finset.univ.erase n, dims d := by
        -- Reindex the left-hand product via `n.succAbove : Fin D ↪ Fin (D+1)`.
        -- Its image is `univ.erase n`.
        have himage :
            (Finset.univ : Finset (Fin D)).image n.succAbove = Finset.univ.erase n := by
          ext x
          simp only [Finset.mem_image, Finset.mem_univ, true_and, Finset.mem_erase]
          refine ⟨?_, ?_⟩
          · rintro ⟨i, rfl⟩
            exact ⟨n.succAbove_ne i, trivial⟩
          · rintro ⟨hxn, _⟩
            exact Fin.exists_succAbove_eq hxn
        have hinj : Set.InjOn n.succAbove (Finset.univ : Finset (Fin D)) := by
          intro a _ b _ hab
          exact n.succAbove_right_injective hab
        rw [← himage, Finset.prod_image hinj]
      let prodCast :
          Fin (∏ i : Fin D, dims (n.succAbove i)) ≃
            Fin (Tensor.prodExcept n dims) :=
        (Fin.castOrderIso heq).toEquiv
      reindexSubtype.trans (flatten.trans prodCast)

/-! ### The `modeUnfoldEquiv` object -/

/-- Mode-n matricization of an order-D tensor, packaged as an `Equiv`
    between `Tensor 𝕜 D dims` and the matrix type
    `Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜`.

    Forward direction = unfolding; inverse direction = folding.
    Built by composing `separateModeEquiv`, `remainingModesEquiv`,
    and the curry equivalence between `α × β → 𝕜` and `α → β → 𝕜`. -/
def modeUnfoldEquiv (dims : Fin D → ℕ) (n : Fin D) :
    Tensor 𝕜 D dims ≃
      Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜 :=
  -- A tensor is a function out of `(d : Fin D) → Fin (dims d)`.
  -- Reindex the domain via `separateModeEquiv n` composed (on the second
  -- factor) with `remainingModesEquiv n dims` to get
  -- `Fin (dims n) × Fin (∏ d ∈ univ.erase n, dims d) → 𝕜`, then curry
  -- into a matrix.
  let reindexDomain :
      (((d : Fin D) → Fin (dims d)) → 𝕜) ≃
        (Fin (dims n) × Fin (Tensor.prodExcept n dims) → 𝕜) :=
    Equiv.arrowCongr
      ((separateModeEquiv n dims).trans
        (Equiv.prodCongr (Equiv.refl _) (remainingModesEquiv n dims)))
      (Equiv.refl 𝕜)
  let curry :
      (Fin (dims n) × Fin (Tensor.prodExcept n dims) → 𝕜) ≃
        Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜 :=
    Equiv.curry _ _ _
  reindexDomain.trans curry

/-! ### Named wrappers -/

/-- Mode-n unfolding as a plain function. -/
def modeUnfold (n : Fin D) (T : Tensor 𝕜 D dims) :
    Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜 :=
  modeUnfoldEquiv dims n T

/-- Fold: inverse of `modeUnfold`. Reshapes a matrix back into a tensor. -/
def fold (dims : Fin D → ℕ) (n : Fin D)
    (M : Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜) :
    Tensor 𝕜 D dims :=
  (modeUnfoldEquiv dims n).symm M

/-! ### Computational lemmas -/

/-- Round-trip: unfolding then folding gives back the tensor. -/
@[simp]
theorem fold_modeUnfold (n : Fin D) (T : Tensor 𝕜 D dims) :
    fold dims n (T.modeUnfold n) = T :=
  (modeUnfoldEquiv dims n).left_inv T

/-- Round-trip: folding then unfolding gives back the matrix. -/
@[simp]
theorem modeUnfold_fold (n : Fin D)
    (M : Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜) :
    (fold dims n M).modeUnfold n = M :=
  (modeUnfoldEquiv dims n).right_inv M

/-- Mode-n unfolding is injective. -/
theorem modeUnfold_injective (n : Fin D) :
    Function.Injective (modeUnfold (𝕜 := 𝕜) (dims := dims) n) :=
  (modeUnfoldEquiv dims n).injective

/-- Unfolding the zero tensor gives the zero matrix. -/
@[simp]
theorem modeUnfold_zero (n : Fin D) :
    modeUnfold (𝕜 := 𝕜) (dims := dims) n 0 = 0 := by
  funext i j
  rfl

/-- Unfolding distributes over addition. -/
theorem modeUnfold_add (n : Fin D) (T₁ T₂ : Tensor 𝕜 D dims) :
    (T₁ + T₂).modeUnfold n = T₁.modeUnfold n + T₂.modeUnfold n := by
  funext i j
  rfl

/-- Unfolding commutes with scalar multiplication. -/
theorem modeUnfold_smul (n : Fin D) (c : 𝕜) (T : Tensor 𝕜 D dims) :
    (c • T).modeUnfold n = c • T.modeUnfold n := by
  funext i j
  rfl

/-! ### Upgrade to `LinearEquiv` -/

/-- Mode-n matricization as a `𝕜`-linear equivalence.
    This bundles `modeUnfoldEquiv` with the additivity and scalar-linearity
    proofs, giving access to `map_add`, `map_smul`, `map_sum`, etc. for free. -/
def modeUnfoldLinearEquiv (dims : Fin D → ℕ) (n : Fin D) :
    Tensor 𝕜 D dims ≃ₗ[𝕜]
      Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜 :=
  { modeUnfoldEquiv dims n with
    map_add' := by intro T₁ T₂; exact modeUnfold_add n T₁ T₂
    map_smul' := by intro c T; exact modeUnfold_smul n c T }

/-! ### Consistency with vectorization -/

/-- `vectorize T` agrees with `vectorize (T.modeUnfold 0)` (viewing the
    matrix `T.modeUnfold 0` as a tensor of order 2 and vectorizing it in
    the Kolda-Bader convention). Both flatten the tensor in the same order,
    with mode 0 varying fastest.

    Stated in the "flat = flat" form used in the lecture exercise:
    the inner-product-preserving identity
    `∑ i, starRingEnd 𝕜 (vectorize T₁ i) * vectorize T₂ i` coincides with
    the Frobenius product of the mode-0 unfoldings. We prove the cleaner
    statement that each entry of the unfolding's vectorization retrieves
    the corresponding tensor entry. -/
theorem modeUnfold_apply (n : Fin D) (T : Tensor 𝕜 D dims)
    (i : Fin (dims n)) (j : Fin (Tensor.prodExcept n dims)) :
    T.modeUnfold n i j =
      T ((separateModeEquiv n dims).symm (i, (remainingModesEquiv n dims).symm j)) := by
  rfl

/-- Pointwise consistency of `vectorize` and mode-0 matricization:
    every entry of the vectorization is realized as an entry of the mode-0
    unfolding, at the image of the multi-index under `separateModeEquiv 0`
    (paired with `remainingModesEquiv 0`).

    Kept as a corollary: it expresses the lecture identity
    "vec(T) = vec(T_(1))" at the pointwise level via the same column-major
    equivs used to define vectorization and unfolding, avoiding any
    reference to matrix vectorization. -/
theorem vectorize_eq_modeUnfold_zero [NeZero D] (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims d)) :
    vectorize T (multiIndexEquiv dims idx) =
      T.modeUnfold 0 (idx 0) (remainingModesEquiv 0 dims
        (fun ⟨k, _⟩ => idx k)) := by
  rw [vectorize_multiIndexEquiv, modeUnfold_apply]
  rw [Equiv.symm_apply_apply]
  -- Now the goal is `T idx = T ((separateModeEquiv 0 dims).symm (idx 0, fun ⟨k,_⟩ => idx k))`.
  -- The pair `(idx 0, fun ⟨k,_⟩ => idx k)` is precisely `separateModeEquiv 0 dims idx`
  -- by definition of `separateModeEquiv`.
  congr 1
  exact ((separateModeEquiv 0 dims).symm_apply_apply idx).symm

/-- Product-split identity at mode 0: the full dimension product factors as
    `dims 0 * Tensor.prodExcept 0 dims`. A direct application of
    `Finset.mul_prod_erase`. -/
theorem prodDims_eq_zero {D : ℕ} (dims : Fin D → ℕ) [NeZero D] :
    dims 0 * Tensor.prodExcept 0 dims = ∏ d : Fin D, dims d :=
  Finset.mul_prod_erase Finset.univ dims (Finset.mem_univ 0)

/-- Numerical decomposition of `multiIndexEquiv` at a tensor's multi-index
    (peel-off form): the flat index equals `(idx 0) + (dims 0) * rest`,
    where `rest` is the flat index of the remaining modes as produced by
    `remainingModesEquiv 0 dims`. This is the content of the peel-off step
    in `multiIndexEquiv`, restated modulo the cast identifying
    `Fin ((∏ i, dims i.succ) * dims 0)` with
    `Fin (∏ d : Fin D, dims d)`. -/
theorem multiIndexEquiv_val_peelOff_zero [NeZero D] (dims : Fin D → ℕ)
    (idx : (d : Fin D) → Fin (dims d)) :
    (multiIndexEquiv dims idx).val =
      (idx 0).val +
        dims 0 *
          (remainingModesEquiv 0 dims (fun ⟨k, _⟩ => idx k)).val := by
  -- `NeZero D` forces `D = D' + 1` for some `D'`.
  obtain ⟨D', rfl⟩ : ∃ D', D = D' + 1 := by
    cases D with
    | zero => exact absurd rfl (NeZero.ne 0)
    | succ D' => exact ⟨D', rfl⟩
  -- Numerical reading of `multiIndexEquiv` on the `D'+1` branch:
  -- it applies `Fin.castOrderIso _` (preserves `.val`) to
  -- `finProdFinEquiv (multiIndexEquiv_rest (tail idx), idx 0)`,
  -- whose `.val` is `(idx 0).val + dims 0 * (multiIndexEquiv_rest _).val`.
  -- Numerical reading of `remainingModesEquiv 0 dims` on the `D'+1` branch:
  -- it applies `Fin.castOrderIso _` (preserves `.val`) to
  -- `multiIndexEquiv (fun i => dims ((0:Fin _).succAbove i)) (reindexed)`,
  -- which is definitionally `multiIndexEquiv (fun i => dims i.succ) (tail idx)`
  -- because `(0:Fin _).succAbove = Fin.succ` and the reindex is `rfl`-equal
  -- to `tail idx`.
  rfl

/-- **Function-level consistency identity of `Tensor.vectorize` and the mode-0
    unfolding's `Matrix.vectorize`.**

    The lecture identity "vec(T) = vec(T_(1))" lifted from
    `vectorize_eq_modeUnfold_zero` to a function-level equation.

    Both `Matrix.vectorize` and `Tensor.vectorize` use the same Kolda-Bader
    column-major convention (row / mode-0 varies fastest), so the two
    flattenings agree on the mode-0 matricization up to a product-
    reassociation `Fin.cast`. -/
theorem vectorize_modeUnfold_zero [NeZero D] (T : Tensor 𝕜 D dims) :
    Tensor.vectorize T =
      Matrix.vectorize (T.modeUnfold 0) ∘ Fin.cast (prodDims_eq_zero dims).symm := by
  funext l
  -- Set `idx := (multiIndexEquiv dims).symm l`, so `l = multiIndexEquiv dims idx`.
  set idx : (d : Fin D) → Fin (dims d) := (multiIndexEquiv dims).symm l with hidx_def
  have hl : l = multiIndexEquiv dims idx := by simp [hidx_def]
  -- Rewrite LHS via the pointwise theorem.
  conv_lhs => rw [hl]
  rw [vectorize_eq_modeUnfold_zero T idx]
  -- Unfold the RHS (function composition application).
  change _ = Matrix.vectorize (T.modeUnfold 0) (Fin.cast _ l)
  rw [Matrix.vectorize_apply]
  -- Goal: T.modeUnfold 0 (idx 0) (remainingModesEquiv 0 dims _)
  --       = T.modeUnfold 0
  --           (finProdFinEquiv.symm ((Fin.cast _ l).cast _)).2
  --           (finProdFinEquiv.symm ((Fin.cast _ l).cast _)).1
  -- Reduce to pair equality. Both components of the pair equation rely on
  -- `dims 0 > 0`, which is witnessed by `idx 0 : Fin (dims 0)` being inhabited.
  have hdims0 : 0 < dims 0 := Fin.pos (idx 0)
  congr 1
  · -- Row: `idx 0 = (finProdFinEquiv.symm ((Fin.cast _ l).cast _)).2`.
    apply Fin.ext
    simp only [finProdFinEquiv_symm_apply, Fin.val_cast, Fin.modNat,
      Fin.divNat]
    -- Goal: `(idx 0).val = l.val % (dims 0)`.
    rw [hl, multiIndexEquiv_val_peelOff_zero]
    rw [Nat.add_mul_mod_self_left]
    exact (Nat.mod_eq_of_lt (idx 0).isLt).symm
  · -- Column: `remainingModesEquiv 0 dims _
    --           = (finProdFinEquiv.symm ((Fin.cast _ l).cast _)).1`.
    apply Fin.ext
    simp only [finProdFinEquiv_symm_apply, Fin.val_cast, Fin.modNat,
      Fin.divNat]
    -- Goal: `(remainingModesEquiv 0 dims _).val = l.val / (dims 0)`.
    rw [hl, multiIndexEquiv_val_peelOff_zero]
    rw [Nat.add_mul_div_left _ _ hdims0,
        Nat.div_eq_of_lt (idx 0).isLt, Nat.zero_add]

end Tensor
