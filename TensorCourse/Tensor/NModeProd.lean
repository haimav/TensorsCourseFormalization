/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.ModeUnfold
import TensorCourse.Tensor.Cast
import Mathlib.Data.Matrix.Basic
import Mathlib.Logic.Function.Basic

/-!
# n-Mode Product of a Tensor with a Matrix

The n-mode product of a tensor `T : Tensor 𝕜 D dims` with a matrix
`U : Matrix (Fin J) (Fin (dims n)) 𝕜` is a new tensor `T ×ₙ U : Tensor 𝕜 D
(Function.update dims n J)` whose mode-n unfolding is `U * T_(n)`.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-! ### Auxiliary: `Function.update` does not change `prodExcept` -/

/-- Updating the `n`-th dimension does not affect `prodExcept n dims` because
    the product skips mode `n`. -/
theorem prodExcept_update (dims : Fin D → ℕ) (n : Fin D) (J : ℕ) :
    Tensor.prodExcept n (Function.update dims n J) = Tensor.prodExcept n dims := by
  unfold Tensor.prodExcept
  refine Finset.prod_congr rfl ?_
  intro d hd
  have hdn : d ≠ n := Finset.ne_of_mem_erase hd
  exact Function.update_of_ne hdn _ _

/-! ### Helper: pointwise evaluation of `fold` -/

/-- Pointwise evaluation of `fold`: applying `fold dims n M` at a multi-index
    retrieves the corresponding matrix entry. Counterpart of `modeUnfold_apply`. -/
theorem fold_apply (n : Fin D)
    (M : Matrix (Fin (dims n)) (Fin (Tensor.prodExcept n dims)) 𝕜)
    (idx : (d : Fin D) → Fin (dims d)) :
    fold dims n M idx =
      M (idx n) (remainingModesEquiv n dims (fun ⟨k, _⟩ => idx k)) :=
  rfl

/-! ### Cast-through lemma for `remainingModesEquiv` -/

/-- When two dimension-functions agree pointwise, `multiIndexEquiv` of a
    tuple casted between them gives the same underlying `Nat` value. -/
theorem multiIndexEquiv_val_congr {D : ℕ} {dims₁ dims₂ : Fin D → ℕ}
    (hdims : ∀ i, dims₁ i = dims₂ i)
    (f₁ : (i : Fin D) → Fin (dims₁ i)) (f₂ : (i : Fin D) → Fin (dims₂ i))
    (hval : ∀ i, (f₁ i).val = (f₂ i).val) :
    (multiIndexEquiv dims₁ f₁).val = (multiIndexEquiv dims₂ f₂).val := by
  induction D with
  | zero => rfl
  | succ D' ih =>
    -- Unfold one recursive step.
    -- `multiIndexEquiv (D+1) dims f = Fin.cast heq
    --   (finProdFinEquiv (multiIndexEquiv tail, f 0))`
    -- whose .val is `(f 0).val + dims 0 * (multiIndexEquiv tail).val`.
    -- Each side has this form; the two sides agree by `hval 0`, `hdims 0`,
    -- and the IH on the tails.
    -- Apply IH on the tails.
    have ihTail :
        (multiIndexEquiv (fun i => dims₁ i.succ) (Fin.tail f₁)).val =
          (multiIndexEquiv (fun i => dims₂ i.succ) (Fin.tail f₂)).val := by
      apply ih
      · intro i; exact hdims i.succ
      · intro i
        change (f₁ i.succ).val = (f₂ i.succ).val
        exact hval i.succ
    -- Unfold `multiIndexEquiv` one step, eliminating the `castOrderIso`
    -- (which preserves `.val`) and unfolding `finProdFinEquiv`.
    -- `multiIndexEquiv` on `D+1` decomposes as `peel ≫ (id × flattenRest) ≫
    -- swap ≫ finProdFinEquiv ≫ Fin.cast`. The last `Fin.cast` preserves
    -- `.val`, so it suffices to compute `.val` of `finProdFinEquiv (…)`.
    rw [show (multiIndexEquiv dims₁ f₁).val =
          (f₁ 0).val + dims₁ 0 *
            (multiIndexEquiv (fun i => dims₁ i.succ) (Fin.tail f₁)).val from rfl,
        show (multiIndexEquiv dims₂ f₂).val =
          (f₂ 0).val + dims₂ 0 *
            (multiIndexEquiv (fun i => dims₂ i.succ) (Fin.tail f₂)).val from rfl]
    rw [hval 0, hdims 0, ihTail]

/-- The `.val` of `remainingModesEquiv n (update dims n J) f` equals the
    `.val` of `remainingModesEquiv n dims (cast f through update_of_ne)`.
    We work at the `.val` level to avoid dependent-type cast gymnastics. -/
theorem remainingModesEquiv_update_val (n : Fin D) (J : ℕ)
    (f : (d : {k : Fin D // k ≠ n}) → Fin ((Function.update dims n J) d.val)) :
    (remainingModesEquiv n (Function.update dims n J) f).val =
      (remainingModesEquiv n dims
        (fun d => (f d).cast (by rw [Function.update_of_ne d.property]))).val := by
  cases D with
  | zero => exact Fin.elim0 n
  | succ D' =>
    simp only [remainingModesEquiv, Equiv.trans_apply]
    apply multiIndexEquiv_val_congr
    · intro i
      have : n.succAbove i ≠ n := n.succAbove_ne i
      exact Function.update_of_ne this J dims
    · intro i; rfl

/-! ### Definition -/

/-- The n-mode product of a tensor `T : Tensor 𝕜 D dims` with a matrix
    `U : Matrix (Fin J) (Fin (dims n)) 𝕜` along mode `n`.
    Defined via unfold → matrix-multiply → fold. The output is a tensor
    with the `n`-th mode's dimension replaced by `J`. -/
def nModeProd {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜)
    (T : Tensor 𝕜 D dims) :
    Tensor 𝕜 D (Function.update dims n J) :=
  Tensor.fold (Function.update dims n J) n
    ((U * T.modeUnfold n).submatrix
      (fun i => i.cast (Function.update_self n J dims))
      (fun j => j.cast (prodExcept_update dims n J)))

/-! ### Per-mode notation -/

scoped infixl:65 " ×₁ " => fun T U => Tensor.nModeProd 0 U T
scoped infixl:65 " ×₂ " => fun T U => Tensor.nModeProd 1 U T
scoped infixl:65 " ×₃ " => fun T U => Tensor.nModeProd 2 U T

/-! ### Matricization characterization -/

/-- Unfolding `T ×ₙ U` along mode `n` gives `U * T.modeUnfold n`
    (up to row/column casts). -/
theorem modeUnfold_nModeProd {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (T : Tensor 𝕜 D dims) :
    (T.nModeProd n U).modeUnfold n =
      (U * T.modeUnfold n).submatrix
        (fun i => i.cast (Function.update_self n J dims))
        (fun j => j.cast (prodExcept_update dims n J)) := by
  unfold nModeProd
  exact modeUnfold_fold n _

/-- `Fin`-level version of `remainingModesEquiv_update_val`: the two sides
    are equal after applying the `prodExcept_update` cast. -/
theorem remainingModesEquiv_update_cast (n : Fin D) (J : ℕ)
    (f : (d : {k : Fin D // k ≠ n}) → Fin ((Function.update dims n J) d.val)) :
    (remainingModesEquiv n (Function.update dims n J) f).cast
        (prodExcept_update dims n J) =
      remainingModesEquiv n dims
        (fun d => (f d).cast (by rw [Function.update_of_ne d.property])) := by
  apply Fin.ext
  simp only [Fin.val_cast]
  exact remainingModesEquiv_update_val n J f

/-- Pointwise matricization characterization. -/
theorem modeUnfold_nModeProd_apply {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (T : Tensor 𝕜 D dims)
    (i : Fin ((Function.update dims n J) n))
    (j : Fin (prodExcept n (Function.update dims n J))) :
    (T.nModeProd n U).modeUnfold n i j =
      (U * T.modeUnfold n)
        (i.cast (Function.update_self n J dims))
        (j.cast (prodExcept_update dims n J)) := by
  rw [modeUnfold_nModeProd]
  rfl

/-! ### Element-wise form -/

/-- Traditional element-wise formula for the n-mode product:
    `(T ×ₙ U) idx = ∑ iₙ, U (idx n) iₙ * T (update-idx-at-n-with-iₙ)`.

    Here `idx` ranges over multi-indices of the output shape
    `Function.update dims n J`. The cast `(idx n).cast (Function.update_self …)`
    converts `idx n : Fin J` (the output dimension) to `Fin ((update dims n J) n)`
    (which is definitionally `Fin J`, but stated as a cast for type alignment).
    On the RHS, the tensor `T` is evaluated at a multi-index of shape `dims`:
    coordinate `n` is replaced by the summation variable `iₙ`, and each other
    coordinate `idx d` is cast from `Fin ((update dims n J) d)` to `Fin (dims d)`
    using `Function.update_of_ne`. -/
@[simp] theorem nModeProd_apply {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin ((Function.update dims n J) d)) :
    (T.nModeProd n U) idx =
      ∑ iₙ : Fin (dims n),
        U ((idx n).cast (Function.update_self n J dims)) iₙ *
          T (fun d =>
            if h : d = n then
              h ▸ iₙ
            else
              (idx d).cast (by rw [Function.update_of_ne h])) := by
  -- Apply `fold_apply` to evaluate `T.nModeProd n U` at a multi-index.
  unfold nModeProd
  rw [fold_apply]
  simp only [Matrix.submatrix_apply, Matrix.mul_apply]
  -- LHS now: `∑ iₙ, U ((idx n).cast hrow) iₙ * T.modeUnfold n iₙ (j.cast hcol)`
  -- where j = remainingModesEquiv n (update dims n J) (fun ⟨k,_⟩ => idx k).
  refine Finset.sum_congr rfl ?_
  intro iₙ _
  congr 1
  -- Now the goal is
  --   T.modeUnfold n iₙ (j.cast hcol) =
  --     T (fun d => if h : d = n then h ▸ iₙ else (idx d).cast …)
  rw [modeUnfold_apply]
  congr 1
  -- Show the two multi-indices (on shape `dims`) agree.
  -- LHS: `separateModeEquiv.symm (iₙ, (remainingModesEquiv n dims).symm (j.cast hcol))`
  -- RHS: `fun d => if d = n then iₙ else (idx d).cast …`
  funext d
  unfold separateModeEquiv
  simp only [Equiv.coe_fn_symm_mk]
  by_cases h : d = n
  · subst h; simp
  · simp only [h, ↓reduceDIte]
    -- Goal: `(remainingModesEquiv n dims).symm (j.cast hcol) ⟨d, h⟩ = (idx d).cast _`
    -- where j = `remainingModesEquiv n (update dims n J) (fun ⟨k,_⟩ => idx k)`.
    -- By `remainingModesEquiv_update_cast`:
    --   `j.cast hcol = rem n dims (fun ⟨k,_⟩ => (idx k).cast _)`.
    -- Hence `(rem n dims).symm (j.cast hcol) ⟨d, h⟩ = (idx d).cast _`.
    rw [remainingModesEquiv_update_cast]
    rw [Equiv.symm_apply_apply]

/-! ### Algebraic identities

We now prove the two key algebraic identities of the n-mode product:
`(T ×ₙ A) ×ₙ B = T ×ₙ (B * A)` (same-mode composition) and
`(T ×ₘ A) ×ₙ B = (T ×ₙ B) ×ₘ A` for `m ≠ n` (different-mode commutativity).
Both have output types involving nested `Function.update` applications;
the statements use `▸` casts through `Function.update_idem` /
`Function.update_comm` to bridge those types. The same-mode case is proved
via matricization (after `modeUnfold_injective` everything reduces to
matrix algebra at concrete `Fin` types). The commutativity case is proved
element-wise via `nModeProd_apply`, rearranged with `Finset.sum_comm` and
matched with `Finset.sum_bij` to bridge the propositionally-equal `Fin`
index types. -/

/-- Transport a tensor's mode-`n` unfolding along a dimension-function
    equality: `(h ▸ T).modeUnfold n` equals `T.modeUnfold n` with row and
    column indices reindexed via `Fin.cast`. -/
private theorem modeUnfold_cast {dims dims' : Fin D → ℕ}
    (h : dims = dims') (T : Tensor 𝕜 D dims) (n : Fin D) :
    ((h ▸ T : Tensor 𝕜 D dims').modeUnfold n) =
      (T.modeUnfold n).submatrix
        (fun i : Fin (dims' n) => i.cast (by rw [h]))
        (fun j : Fin (Tensor.prodExcept n dims') => j.cast (by rw [h])) := by
  subst h
  ext i j
  rfl

/-- **Same-mode composition:** applying an n-mode product twice along the
    same mode equals a single n-mode product against the composed matrix.

    The matrix `B` has columns `Fin J`, matching `A`'s rows; to type-check
    as an argument to `(T ×ₙ A) ×ₙ _`, whose input mode-`n` dimension is
    `(Function.update dims n J) n`, `B` is reindexed via `submatrix`.

    Output types: the LHS lives in `Tensor 𝕜 D (update (update dims n J) n K)`
    while the RHS-before-cast lives in `Tensor 𝕜 D (update dims n K)`. These
    are propositionally equal by `Function.update_idem`, and the `▸` cast
    on the RHS transports the RHS to the LHS's type. -/
theorem nModeProd_nModeProd_same {J K : ℕ} (n : Fin D)
    (A : Matrix (Fin J) (Fin (dims n)) 𝕜)
    (B : Matrix (Fin K) (Fin J) 𝕜)
    (T : Tensor 𝕜 D dims) :
    (T.nModeProd n A).nModeProd n
        (B.submatrix id
          (fun j : Fin ((Function.update dims n J) n) =>
            j.cast (Function.update_self n J dims)))
      = (Function.update_idem J K dims) ▸ T.nModeProd n (B * A) := by
  -- Matricization route: unfold along mode `n`, work at matrices.
  apply modeUnfold_injective n
  rw [modeUnfold_nModeProd, modeUnfold_nModeProd, modeUnfold_cast,
      modeUnfold_nModeProd]
  ext i j
  simp only [Matrix.submatrix_apply, Matrix.mul_apply, id]
  simp only [Finset.mul_sum, Finset.sum_mul]
  -- LHS has summation order (Fin J, Fin (dims n)); swap inside `conv` to
  -- keep the `AddCommMonoid` instance resolved (a bare `Finset.sum_comm`
  -- gets stuck on typeclass inference after the `simp` rewrite).
  conv_lhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  -- Inner sum on LHS is over `Fin ((update dims n J) n)` (propositionally
  -- `Fin J`); bridge via `Finset.sum_bij` using `Fin.cast`.
  apply Finset.sum_bij
    (fun x _ => Fin.cast (Function.update_self n J dims) x)
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨Fin.cast (Function.update_self n J dims).symm b,
      Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  · intro a _
    -- After reindexing, the summand matches up to `Fin.cast` on the first
    -- `B`-index (which transports through the `update_idem` equality).
    rw [← mul_assoc]
    congr 1

/-- **Different-mode commutativity:** n-mode products along distinct modes
    commute.

    The matrices `A : Matrix (Fin Jm) (Fin (dims m)) 𝕜` and
    `B : Matrix (Fin Jn) (Fin (dims n)) 𝕜` operate on modes `m` and `n`
    respectively, with `m ≠ n`. In the LHS, `B` is reindexed via `submatrix`
    because, after `T.nModeProd m A`, the mode-`n` dimension becomes
    `(update dims m Jm) n` (definitionally `dims n` only modulo
    `Function.update_of_ne`); similarly for `A` in the RHS.

    The output types are propositionally equal by `Function.update_comm hmn`,
    bridged by the `▸` cast on the RHS. -/
theorem nModeProd_nModeProd_comm {m n : Fin D} (hmn : m ≠ n)
    {Jm Jn : ℕ}
    (A : Matrix (Fin Jm) (Fin (dims m)) 𝕜)
    (B : Matrix (Fin Jn) (Fin (dims n)) 𝕜)
    (T : Tensor 𝕜 D dims) :
    (T.nModeProd m A).nModeProd n
        (B.submatrix id
          (fun j : Fin ((Function.update dims m Jm) n) =>
            j.cast (Function.update_of_ne (Ne.symm hmn) _ dims)))
      = (Function.update_comm hmn Jm Jn dims) ▸
          ((T.nModeProd n B).nModeProd m
            (A.submatrix id
              (fun j : Fin ((Function.update dims n Jn) m) =>
                j.cast (Function.update_of_ne hmn _ dims)))) := by
  -- Element-wise route: both sides expand to the same double sum over
  -- `Fin (dims m)` and `Fin (dims n)` (modulo cast).
  funext idx
  rw [cast_apply]
  simp only [nModeProd_apply, Matrix.submatrix_apply, id]
  -- `if h : m = n` / `if h : n = m` collapses via `hmn`.
  simp only [hmn, ↓reduceDIte, hmn.symm]
  simp only [Finset.mul_sum]
  -- Swap LHS outer/inner sums (inside `conv` to keep the scalar type pinned).
  conv_lhs => rw [Finset.sum_comm]
  -- Outer `Finset.sum_bij`: bridge `Fin (dims m) ↔ Fin ((update dims n Jn) m)`.
  apply Finset.sum_bij
    (fun (iₘ : Fin (dims m)) _ =>
      Fin.cast (Function.update_of_ne hmn _ dims).symm iₘ)
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨Fin.cast (Function.update_of_ne hmn _ dims) b, Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  intro iₘ _
  -- Inner `Finset.sum_bij`: bridge `Fin ((update dims m Jm) n) ↔ Fin (dims n)`.
  apply Finset.sum_bij
    (fun (iₙ : Fin ((Function.update dims m Jm) n)) _ =>
      Fin.cast (Function.update_of_ne (Ne.symm hmn) _ dims) iₙ)
  · intros; exact Finset.mem_univ _
  · intros a _ b _ hab; exact Fin.ext (by simpa using Fin.val_eq_of_eq hab)
  · intro b _
    refine ⟨Fin.cast (Function.update_of_ne (Ne.symm hmn) _ dims).symm b,
      Finset.mem_univ _, ?_⟩
    apply Fin.ext; rfl
  intro iₙ _
  -- Reassociate and swap factors: `B * (A * T) = A * (B * T)` (scalar mul).
  rw [← mul_assoc, mul_comm (B _ _), mul_assoc]
  -- Match factor-by-factor; `Eq.rec idx` preserves `.val` via `cast_idx_val`.
  have hrecm : (Eq.rec (motive := fun (x : Fin D → ℕ) (_ : _) =>
      (d : Fin D) → Fin (x d)) idx (Function.update_comm hmn Jm Jn dims) m).val
        = (idx m).val :=
    cast_idx_val (Function.update_comm hmn Jm Jn dims) idx m
  have hrecn : (Eq.rec (motive := fun (x : Fin D → ℕ) (_ : _) =>
      (d : Fin D) → Fin (x d)) idx (Function.update_comm hmn Jm Jn dims) n).val
        = (idx n).val :=
    cast_idx_val (Function.update_comm hmn Jm Jn dims) idx n
  congr 1
  · apply congrArg₂
    · apply Fin.ext; simp [Fin.val_cast, hrecm]
    · apply Fin.ext; rfl
  congr 1
  · apply congrArg₂
    · apply Fin.ext; simp [Fin.val_cast, hrecn]
    · apply Fin.ext; rfl
  -- The `T`-argument functions agree pointwise.
  congr 1
  funext d
  by_cases hdm : d = m
  · subst hdm; simp [hmn]
  · by_cases hdn : d = n
    · subst hdn; simp [Ne.symm hmn]
    · simp only [hdm, hdn, ↓reduceDIte]
      apply Fin.ext
      simp only [Fin.val_cast]
      exact (cast_idx_val (Function.update_comm hmn Jm Jn dims) idx d).symm

/-- Bilinearity: `nModeProd` is additive in the matrix argument. -/
theorem nModeProd_add_left {J : ℕ} (n : Fin D)
    (U₁ U₂ : Matrix (Fin J) (Fin (dims n)) 𝕜) (T : Tensor 𝕜 D dims) :
    T.nModeProd n (U₁ + U₂) = T.nModeProd n U₁ + T.nModeProd n U₂ := by
  apply (modeUnfoldLinearEquiv (Function.update dims n J) n).injective
  rw [LinearEquiv.map_add]
  change (T.nModeProd n (U₁ + U₂)).modeUnfold n =
      (T.nModeProd n U₁).modeUnfold n + (T.nModeProd n U₂).modeUnfold n
  rw [modeUnfold_nModeProd, modeUnfold_nModeProd, modeUnfold_nModeProd]
  ext i j
  simp [Matrix.add_mul]

/-- Scalar-multiplication in the matrix argument. -/
theorem nModeProd_smul_left {J : ℕ} (n : Fin D)
    (c : 𝕜) (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (T : Tensor 𝕜 D dims) :
    T.nModeProd n (c • U) = c • T.nModeProd n U := by
  apply (modeUnfoldLinearEquiv (Function.update dims n J) n).injective
  rw [LinearEquiv.map_smul]
  change (T.nModeProd n (c • U)).modeUnfold n =
      c • (T.nModeProd n U).modeUnfold n
  rw [modeUnfold_nModeProd, modeUnfold_nModeProd]
  ext i j
  simp [Matrix.smul_mul]

/-- Additivity in the tensor argument. -/
theorem nModeProd_add_right {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (T₁ T₂ : Tensor 𝕜 D dims) :
    (T₁ + T₂).nModeProd n U = T₁.nModeProd n U + T₂.nModeProd n U := by
  apply (modeUnfoldLinearEquiv (Function.update dims n J) n).injective
  rw [LinearEquiv.map_add]
  change ((T₁ + T₂).nModeProd n U).modeUnfold n =
      (T₁.nModeProd n U).modeUnfold n + (T₂.nModeProd n U).modeUnfold n
  rw [modeUnfold_nModeProd, modeUnfold_nModeProd, modeUnfold_nModeProd,
      modeUnfold_add, Matrix.mul_add]
  ext i j
  simp

/-- Scalar-multiplication in the tensor argument. -/
theorem nModeProd_smul_right {J : ℕ} (n : Fin D)
    (U : Matrix (Fin J) (Fin (dims n)) 𝕜) (c : 𝕜) (T : Tensor 𝕜 D dims) :
    (c • T).nModeProd n U = c • T.nModeProd n U := by
  apply (modeUnfoldLinearEquiv (Function.update dims n J) n).injective
  rw [LinearEquiv.map_smul]
  change ((c • T).nModeProd n U).modeUnfold n =
      c • (T.nModeProd n U).modeUnfold n
  rw [modeUnfold_nModeProd, modeUnfold_nModeProd, modeUnfold_smul,
      Matrix.mul_smul]
  ext i j
  simp

/-! ### n-mode product with the identity matrix -/

/-- The n-mode product with the identity matrix is the identity
    (the output type `Function.update dims n (dims n)` is `dims` modulo
    `Function.update_eq_self`). -/
theorem nModeProd_one (n : Fin D) (Y : Tensor 𝕜 D dims) :
    Y.nModeProd n (1 : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜) =
      cast (congrArg (Tensor 𝕜 D) (Function.update_eq_self n dims).symm) Y := by
  funext idx
  rw [nModeProd_apply, cast_tensor_apply (h := (Function.update_eq_self n dims).symm)]
  -- LHS: `∑ iₙ, (1) ((idx n).cast _) iₙ * Y (update-at-n idx iₙ)`; the
  -- identity matrix collapses the sum to `iₙ = (idx n).cast _`.
  rw [Finset.sum_eq_single_of_mem
      ((idx n).cast (Function.update_self n (dims n) dims)) (Finset.mem_univ _)
      (fun b _ hb => by rw [Matrix.one_apply_ne (Ne.symm hb), zero_mul])]
  rw [Matrix.one_apply_eq, one_mul]
  -- Match the two `Y`-arguments pointwise (each coordinate has equal `.val`).
  refine congrArg Y ?_
  funext d
  apply Fin.ext
  by_cases h : d = n
  · subst h
    simp [Fin.val_cast]
  · simp only [h, ↓reduceDIte, Fin.val_cast]

end Tensor
