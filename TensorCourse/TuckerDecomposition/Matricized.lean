/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TuckerDecomposition.Defs
import TensorCourse.Tensor.ModeUnfold
import TensorCourse.Tensor.Vectorize
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Algebra.Star.BigOperators

/-!
# Tucker Decomposition: Matricized and Vectorized Forms

This file proves the matricized and vectorized forms of the Tucker
reconstruction `tuckerTensor S U`, the Tucker analogues of the
Khatri-Rao matricization identity `cp_modeUnfold_eq_khatriRao` (Lecture 4).
Where the CP identity uses a *column-wise* Khatri-Rao product, the Tucker
identity uses the *full* Kronecker product of the factor matrices.

## Main definitions

* `Tensor.kroneckerExcept U n` — the full Kronecker product of all factor
  matrices `U d` except mode `n`, in the Kolda-Bader reverse-mode order.
  Defined directly via the closed-form pointwise formula using
  `remainingModesEquiv` on **both** the row (`I`) and column (`J`) axes, so
  it lands natively in
  `Matrix (Fin (Tensor.prodExcept n I)) (Fin (Tensor.prodExcept n J)) 𝕜`
  with no `submatrix` reindex (exactly as `khatriRaoExcept` in Lecture 4).
* `Tensor.kroneckerAll U` — the full Kronecker product over *all* modes,
  closed-form via `multiIndexEquiv` on both axes, landing in
  `Matrix (Fin (∏ d, I d)) (Fin (∏ d, J d)) 𝕜`.

## Main results

* `Tensor.kroneckerExcept_apply` — closed-form pointwise simp lemma.
* `Tensor.tuckerTensor_modeUnfold` — **the matricized form**
  (eq:Tucker-Norder-matricized), general `D`, general `n`:
  ```
  (tuckerTensor S U).modeUnfold n = U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ
  ```
  No `submatrix` reindex on any axis.
* `Tensor.modeUnfold_eq_iff_tuckerTensor` — the iff-form, via
  `modeUnfold_injective`.
* `Tensor.kroneckerAll_apply` — closed-form pointwise simp lemma.
* `Tensor.vectorize_tuckerTensor` — **the vectorized form**
  (eq:Tucker-Norder-vectorized): `vec(X) = (kroneckerAll U) *ᵥ vec(S)`.
* `Tensor.kroneckerAll_orthonormalCols` — if each `(U d)ᴴ * (U d) = 1` then
  `(kroneckerAll U)ᴴ * (kroneckerAll U) = 1`.
* `Tensor.kroneckerAll_conjTranspose` —
  `kroneckerAll (fun d => (U d)ᴴ) = (kroneckerAll U)ᴴ`.

## Submatrix reindexes

As in `cp_modeUnfold_eq_khatriRao`, `kroneckerExcept` and `kroneckerAll`
are defined to land *natively* in the right matrix types, so the matricized
and vectorized forms need **no `submatrix` reindex on any axis**.

## Ordering convention

The row/column orders of `kroneckerExcept`, viewed as a multi-index over
modes `d ≠ n`, follow the Kolda-Bader column-major convention of
`remainingModesEquiv` (smallest mode index varies fastest), matching the
lecture's reverse-order Kronecker listing
`U⁽ᴰ⁻¹⁾ ⊗ ⋯ ⊗ U⁽ⁿ⁺¹⁾ ⊗ U⁽ⁿ⁻¹⁾ ⊗ ⋯ ⊗ U⁽⁰⁾`. Likewise `kroneckerAll`
follows `multiIndexEquiv`.

## Transpose vs. conjugate transpose

The matricized/vectorized identities are *pure algebra* and hold over any
`RCLike` field, so they use the plain transpose `ᵀ` (matching the lecture
and the CP Khatri-Rao file). The orthonormality lemmas
`kroneckerAll_orthonormalCols` / `kroneckerAll_conjTranspose` are
*adjoint*-flavoured (consumed by the optimal-core proof, where the inner
product `⟪·,·⟫ₜ` conjugates its first argument), so they use the conjugate
transpose `ᴴ`; over `ℝ`, `ᴴ = ᵀ`.
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {I J : Fin D → ℕ}

/-! ### Definition of `kroneckerExcept` -/

/-- The full Kronecker product of factor matrices `U : (d : Fin D) →
    Matrix (Fin (I d)) (Fin (J d)) 𝕜`, omitting the `n`-th factor.

    The lecture writes this as a Kronecker chain over modes `d ≠ n` in
    reverse mode order, `U⁽ᴰ⁻¹⁾ ⊗ ⋯ ⊗ U⁽ⁿ⁺¹⁾ ⊗ U⁽ⁿ⁻¹⁾ ⊗ ⋯ ⊗ U⁽⁰⁾`. We do
    *not* prove that our definition equals that explicit factor chain;
    instead we define it directly via the closed-form pointwise formula, and
    what we establish (in `tuckerTensor_modeUnfold`) is that its **row/column
    ordering matches `Tensor.modeUnfold`'s column ordering** — the
    Kolda-Bader column-major convention — so that the matricized identity has
    identity row/column reindexes on both axes.

    Both the row index `ri : Fin (Tensor.prodExcept n I)` and the column
    index `cj : Fin (Tensor.prodExcept n J)` are read off as multi-indices on
    `{k : Fin D // k ≠ n}` via the inverse of `remainingModesEquiv` (on the
    respective shapes `I` and `J`), then evaluated at each `d ≠ n`.

    For `d = n` the product term is `1` (the membership filter on
    `univ.erase n` discards it, but for type-correctness we still need a
    well-typed expression at `d = n`); we use `dite` to branch, exactly as
    `khatriRaoExcept` does. -/
def kroneckerExcept
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) (n : Fin D) :
    Matrix (Fin (Tensor.prodExcept n I)) (Fin (Tensor.prodExcept n J)) 𝕜 :=
  Matrix.of fun (ri : Fin (Tensor.prodExcept n I)) (cj : Fin (Tensor.prodExcept n J)) =>
    ∏ d ∈ Finset.univ.erase n,
      (if h : d = n then (1 : 𝕜)
       else U d ((remainingModesEquiv n I).symm ri ⟨d, h⟩)
              ((remainingModesEquiv n J).symm cj ⟨d, h⟩))

/-- Pointwise formula for `kroneckerExcept`: the entry at row `ri` and column
    `cj` is the product over modes `d ∈ univ.erase n` of
    `U d (rowIdx d) (colIdx d)`, where `rowIdx = (remainingModesEquiv n I).symm ri`
    and `colIdx = (remainingModesEquiv n J).symm cj` read off the multi-indices
    on the remaining modes from the column-major flat indices.

    **Why the `dite`?** The `then 1` branch (`d = n`) is unreachable because
    the product iterates over `univ.erase n`, which excludes `n`. The `dite`
    is needed only so that `h : d ≠ n` is available as the property argument
    to `(remainingModesEquiv n I).symm ri ⟨d, h⟩`. Inside any concrete proof
    the `then 1` branch can be eliminated by `Finset.prod_congr rfl (fun d hd =>
    dif_neg (Finset.ne_of_mem_erase hd))`. -/
theorem kroneckerExcept_apply
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) (n : Fin D)
    (ri : Fin (Tensor.prodExcept n I)) (cj : Fin (Tensor.prodExcept n J)) :
    (kroneckerExcept U n) ri cj =
      ∏ d ∈ Finset.univ.erase n,
        (if h : d = n then (1 : 𝕜)
         else U d ((remainingModesEquiv n I).symm ri ⟨d, h⟩)
                ((remainingModesEquiv n J).symm cj ⟨d, h⟩)) :=
  rfl

/-! ### The Tucker matricized form -/

/-- **Tucker matricized form (eq:Tucker-Norder-matricized), general `D`,
    general `n`.** The mode-`n` matricization of a Tucker reconstruction is

    `(tuckerTensor S U).modeUnfold n = U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ`.

    No `submatrix` reindex on any axis: the row dimension `Fin (I n)` is
    matched by `U n`'s rows, the inner core-row dimension `Fin (J n)` by
    `U n`'s columns and `S.modeUnfold n`'s rows, and the column dimension
    `Fin (Tensor.prodExcept n I)` / `Fin (Tensor.prodExcept n J)` by
    `(kroneckerExcept U n)ᵀ`. This is the cleanest possible statement,
    mirroring `cp_modeUnfold_eq_khatriRao`.

    **Proof (Route A, the `cp_modeUnfold_eq_khatriRao` template).** `ext i j`;
    expand the LHS via `modeUnfold_apply` + `tuckerTensor_apply` into a sum
    over core multi-indices `jdx`; expand the RHS via `Matrix.mul_apply`
    (twice), `Matrix.transpose_apply`, `kroneckerExcept_apply` into a double
    sum over a core-column index `c` and a mode-`n` core index `p`. Reindex
    the double sum into a single sum over `jdx` via the bijection
    `jdx ↦ (jdx n, remaining of jdx)` (`separateModeEquiv n J` then
    `remainingModesEquiv n J`), then match termwise: the product over modes
    splits as `(mode-n factor) · ∏_{d≠n}(…)` via `Finset.mul_prod_erase`. -/
theorem tuckerTensor_modeUnfold (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) (n : Fin D) :
    (tuckerTensor S U).modeUnfold n =
      U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ := by
  funext i j
  -- LHS: expand the mode-`n` unfolding and the Tucker closed form.
  rw [modeUnfold_apply, tuckerTensor_apply]
  -- LHS is now `∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)` where
  --   idx = (separateModeEquiv n I).symm (i, (remainingModesEquiv n I).symm j).
  -- Set `idx` for readability.
  set idxRow : (d : Fin D) → Fin (I d) :=
    (separateModeEquiv n I).symm (i, (remainingModesEquiv n I).symm j) with hidxRow
  -- At mode `n`, `idxRow n = i`.
  have hidxRow_n : idxRow n = i := by
    rw [hidxRow]
    change (if h : n = n then h ▸ i else _) = i
    simp
  -- Off mode `n`, `idxRow d = (remainingModesEquiv n I).symm j ⟨d, _⟩`.
  have hidxRow_off : ∀ d (hd : d ≠ n), idxRow d =
      (remainingModesEquiv n I).symm j ⟨d, hd⟩ := by
    intro d hd
    rw [hidxRow]
    change (if h : d = n then h ▸ i else _) = _
    simp [hd]
  -- RHS: expand the two matrix multiplications and the transpose.
  rw [Matrix.mul_apply]
  -- RHS: `∑ c, (U n * S.modeUnfold n) i c * (kroneckerExcept U n)ᵀ c j`.
  simp only [Matrix.transpose_apply, Matrix.mul_apply]
  -- RHS: `∑ c, (∑ p, U n i p * S.modeUnfold n p c) * kroneckerExcept U n j c`.
  -- Reindex the LHS sum over `jdx` into a double sum over `(c, p)` via the
  -- bijection `jdx ↦ (remaining of jdx, jdx n)`. We instead reindex the RHS
  -- backwards: rewrite each summand and combine the two RHS sums into one
  -- sum over `jdx`.
  -- First, distribute the column-sum factor into the `p`-sum on the RHS, then
  -- swap to `∑ p, ∑ c` so the flattened pair index is `(p, c)`.
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm]
  -- RHS: `∑ p, ∑ c, (U n i p * S.modeUnfold n p c) * kroneckerExcept U n j c`.
  -- Flatten `(p, c)` into a single sum over `Fin (J n) × Fin (prodExcept n J)`.
  rw [← Finset.sum_product', Finset.univ_product_univ]
  -- RHS: `∑ pc : Fin (J n) × Fin (prodExcept n J), …`.
  -- Reindex via the bijection `e jdx = (jdx n, remainingModesEquiv n J (rest jdx))`,
  -- i.e. `separateModeEquiv n J` followed by `remainingModesEquiv n J` on the
  -- second component.
  set e : ((d : Fin D) → Fin (J d)) ≃ Fin (J n) × Fin (Tensor.prodExcept n J) :=
    (separateModeEquiv n J).trans
      (Equiv.prodCongr (Equiv.refl (Fin (J n))) (remainingModesEquiv n J)) with he
  -- The forward map of `e` evaluated at any `jdx`.
  have he_apply : ∀ jdx : (d : Fin D) → Fin (J d),
      e jdx = (jdx n, remainingModesEquiv n J (fun (k : {k : Fin D // k ≠ n}) => jdx k.val)) := by
    intro jdx
    rfl
  rw [← Equiv.sum_comp e
    (fun pc : Fin (J n) × Fin (Tensor.prodExcept n J) =>
      U n i pc.1 * S.modeUnfold n pc.1 pc.2 * (kroneckerExcept U n) j pc.2)]
  -- Now both sides are sums over `jdx`. Match termwise.
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  -- Evaluate the reindexing equiv at `jdx`.
  rw [he_apply]
  -- The RHS summand is now
  --   `U n i (jdx n) * S.modeUnfold n (jdx n) c * kroneckerExcept U n j c`
  -- with `c = remainingModesEquiv n J (fun ⟨k,_⟩ => jdx k)`.
  set c : Fin (Tensor.prodExcept n J) :=
    remainingModesEquiv n J (fun (k : {k : Fin D // k ≠ n}) => jdx k.val) with hc
  -- Expand `S.modeUnfold n (jdx n) c`: it is `S` at the multi-index
  -- recovered from `(jdx n, (remainingModesEquiv n J).symm c)`.
  rw [modeUnfold_apply]
  -- `(remainingModesEquiv n J).symm c = fun ⟨k,_⟩ => jdx k` by inversion.
  have hcsymm : (remainingModesEquiv n J).symm c =
      (fun (k : {k : Fin D // k ≠ n}) => jdx k.val) := by
    rw [hc, Equiv.symm_apply_apply]
  rw [hcsymm]
  -- The recovered multi-index is `(separateModeEquiv n J).symm (jdx n, rest) = jdx`.
  have hrecover : (separateModeEquiv n J).symm
      (jdx n, (fun (k : {k : Fin D // k ≠ n}) => jdx k.val)) = jdx := by
    exact (separateModeEquiv n J).symm_apply_apply jdx
  rw [hrecover]
  -- Expand `kroneckerExcept U n j c` via its pointwise formula.
  rw [kroneckerExcept_apply]
  -- LHS: `S jdx * ∏ d, U d (idxRow d) (jdx d)`.
  -- Split off the mode-`n` factor from the LHS product.
  rw [← Finset.mul_prod_erase (Finset.univ : Finset (Fin D))
        (fun d => U d (idxRow d) (jdx d)) (Finset.mem_univ n)]
  rw [hidxRow_n]
  -- Goal:
  --   S jdx * (U n i (jdx n) * ∏_{d≠n} U d (idxRow d) (jdx d))
  --   = U n i (jdx n) * S jdx * ∏_{d≠n}
  --       (if h : d=n then 1 else U d ((remEquiv I).symm j ⟨d,h⟩) ((remEquiv J).symm c ⟨d,h⟩)).
  -- The remaining-modes products match: for `d ≠ n`,
  --   idxRow d = (remEquiv I).symm j ⟨d,_⟩  and  jdx d = (remEquiv J).symm c ⟨d,_⟩.
  rw [show ∀ (a b cc : 𝕜), a * (b * cc) = b * a * cc from fun a b cc => by ring]
  congr 1
  refine Finset.prod_congr rfl (fun d hd => ?_)
  have hd_ne : d ≠ n := Finset.ne_of_mem_erase hd
  rw [dif_neg hd_ne, hidxRow_off d hd_ne]
  -- The column argument: `jdx d = (remEquiv J).symm c ⟨d, hd_ne⟩`.
  congr 1
  rw [hcsymm]

/-! ### The iff-form of the matricized identity -/

/-- **The iff-form of the Tucker matricized identity.** A tensor `T` equals
    the Tucker reconstruction `tuckerTensor S U` if and only if its mode-`n`
    matricization equals `U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ`.

    The forward direction is `tuckerTensor_modeUnfold`; the reverse uses
    `modeUnfold_injective`. -/
theorem modeUnfold_eq_iff_tuckerTensor (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) (n : Fin D)
    (T : Tensor 𝕜 D I) :
    T.modeUnfold n = U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ
      ↔ T = tuckerTensor S U := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · apply modeUnfold_injective n
    rw [h, tuckerTensor_modeUnfold]
  · subst h
    exact tuckerTensor_modeUnfold S U n

/-! ### Order-3 sanity check at all three modes

The plan's stepping stone: the order-3 case of the matricized identity at
each of the three modes, validating that the Kolda-Bader row/column reindex
inside `kroneckerExcept` produces the same ordering as
`Tensor.remainingModesEquiv` at order 3. Obtained as direct specializations
of the general-`D` identity (the general proof was tractable). -/

/-- Order-3 specialization of `tuckerTensor_modeUnfold` at mode 0. -/
theorem tuckerTensor_modeUnfold_three_zero {I J : Fin 3 → ℕ} (S : Tensor 𝕜 3 J)
    (U : (d : Fin 3) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    (tuckerTensor S U).modeUnfold 0 =
      U 0 * S.modeUnfold 0 * (kroneckerExcept U 0)ᵀ :=
  tuckerTensor_modeUnfold S U 0

/-- Order-3 specialization of `tuckerTensor_modeUnfold` at mode 1. -/
theorem tuckerTensor_modeUnfold_three_one {I J : Fin 3 → ℕ} (S : Tensor 𝕜 3 J)
    (U : (d : Fin 3) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    (tuckerTensor S U).modeUnfold 1 =
      U 1 * S.modeUnfold 1 * (kroneckerExcept U 1)ᵀ :=
  tuckerTensor_modeUnfold S U 1

/-- Order-3 specialization of `tuckerTensor_modeUnfold` at mode 2. -/
theorem tuckerTensor_modeUnfold_three_two {I J : Fin 3 → ℕ} (S : Tensor 𝕜 3 J)
    (U : (d : Fin 3) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    (tuckerTensor S U).modeUnfold 2 =
      U 2 * S.modeUnfold 2 * (kroneckerExcept U 2)ᵀ :=
  tuckerTensor_modeUnfold S U 2

/-! ### Definition of `kroneckerAll` -/

/-- The full Kronecker product over *all* modes of the factor matrices
    `U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜`, in the Kolda-Bader
    reverse-mode order `U⁽ᴰ⁻¹⁾ ⊗ ⋯ ⊗ U⁽⁰⁾`. Defined directly via the
    closed-form pointwise formula using `multiIndexEquiv` on **both** the row
    (`I`) and column (`J`) axes, so it lands natively in
    `Matrix (Fin (∏ d, I d)) (Fin (∏ d, J d)) 𝕜` with no `submatrix`.

    The row index `ri : Fin (∏ d, I d)` and column index `cj : Fin (∏ d, J d)`
    are read off as multi-indices via the inverse of `multiIndexEquiv` on the
    respective shapes, then the entry is `∏ d, U d (rowIdx d) (colIdx d)`. -/
def kroneckerAll
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    Matrix (Fin (∏ d : Fin D, I d)) (Fin (∏ d : Fin D, J d)) 𝕜 :=
  Matrix.of fun (ri : Fin (∏ d : Fin D, I d)) (cj : Fin (∏ d : Fin D, J d)) =>
    ∏ d : Fin D, U d ((multiIndexEquiv I).symm ri d) ((multiIndexEquiv J).symm cj d)

/-- Pointwise formula for `kroneckerAll`: the entry at row `ri` and column
    `cj` is the product over all modes `d` of `U d (rowIdx d) (colIdx d)`,
    where `rowIdx = (multiIndexEquiv I).symm ri` and
    `colIdx = (multiIndexEquiv J).symm cj`. -/
theorem kroneckerAll_apply
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜)
    (ri : Fin (∏ d : Fin D, I d)) (cj : Fin (∏ d : Fin D, J d)) :
    (kroneckerAll U) ri cj =
      ∏ d : Fin D, U d ((multiIndexEquiv I).symm ri d) ((multiIndexEquiv J).symm cj d) :=
  rfl

/-! ### The Tucker vectorized form -/

/-- **Tucker vectorized form (eq:Tucker-Norder-vectorized).** The
    vectorization of a Tucker reconstruction is the matrix-vector product of
    the full Kronecker product `kroneckerAll U` with the vectorized core:

    `vectorize (tuckerTensor S U) = (kroneckerAll U) *ᵥ vectorize S`.

    Proved directly pointwise: `vectorize (tuckerTensor S U) l` expands (via
    `multiIndexEquiv`) to `∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)` with
    `idx = (multiIndexEquiv I).symm l`, while
    `(kroneckerAll U *ᵥ vectorize S) l` expands (via `Matrix.mulVec` =
    `dotProduct` and `kroneckerAll_apply`) to a sum over the core-column flat
    index, which reindexes to a sum over `jdx` via `multiIndexEquiv J`. -/
theorem vectorize_tuckerTensor (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    Tensor.vectorize (tuckerTensor S U) =
      (kroneckerAll U) *ᵥ (Tensor.vectorize S) := by
  funext l
  -- LHS: expand vectorize and the Tucker closed form.
  rw [vectorize_apply, tuckerTensor_apply]
  -- LHS: `∑ jdx, S jdx * ∏ d, U d (idx d) (jdx d)`, `idx = (multiIndexEquiv I).symm l`.
  set idx : (d : Fin D) → Fin (I d) := (multiIndexEquiv I).symm l with hidx
  -- RHS: `(kroneckerAll U *ᵥ vectorize S) l = ∑ jcol, kroneckerAll U l jcol * vectorize S jcol`.
  change _ = (fun jcol => (kroneckerAll U) l jcol) ⬝ᵥ (Tensor.vectorize S)
  rw [dotProduct]
  -- RHS: `∑ jcol, kroneckerAll U l jcol * vectorize S jcol`.
  -- Reindex the RHS `jcol`-sum into a sum over `jdx` via `multiIndexEquiv J`.
  rw [← Equiv.sum_comp (multiIndexEquiv J)
    (fun jcol => (kroneckerAll U) l jcol * Tensor.vectorize S jcol)]
  -- Match termwise.
  refine Finset.sum_congr rfl (fun jdx _ => ?_)
  rw [kroneckerAll_apply, vectorize_apply, Equiv.symm_apply_apply]
  -- RHS summand: `(∏ d, U d (idx d) (jdx d)) * S jdx`; LHS: `S jdx * ∏ d, U d (idx d) (jdx d)`.
  rw [hidx, mul_comm]

/-! ### Kronecker orthonormality (adjoint-flavoured) -/

/-- `kroneckerAll (fun d => (U d)ᴴ) = (kroneckerAll U)ᴴ`. Companion to the
    orthonormality corollary below: the conjugate transpose of the full
    Kronecker product is the full Kronecker product of the conjugate
    transposes. Direct pointwise via `Matrix.conjTranspose_apply`, the
    multiplicativity of `star`, and the row/column swap of each `(U d)ᴴ`. -/
theorem kroneckerAll_conjTranspose
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜) :
    kroneckerAll (fun d => (U d)ᴴ) = (kroneckerAll U)ᴴ := by
  funext cj ri
  rw [kroneckerAll_apply, Matrix.conjTranspose_apply, kroneckerAll_apply]
  -- LHS: `∏ d, (U d)ᴴ ((multiIndexEquiv J).symm cj d) ((multiIndexEquiv I).symm ri d)`.
  -- RHS: `star (∏ d, U d ((multiIndexEquiv I).symm ri d) ((multiIndexEquiv J).symm cj d))`.
  rw [star_prod]
  refine Finset.prod_congr rfl (fun d _ => ?_)
  rw [Matrix.conjTranspose_apply]

/-- **Kronecker orthonormality corollary.** If each factor matrix `U d` has
    orthonormal columns (`(U d)ᴴ * (U d) = 1`), then the full Kronecker
    product `kroneckerAll U` has orthonormal columns:
    `(kroneckerAll U)ᴴ * (kroneckerAll U) = 1`.

    Consumed by Phase 5's optimal-core proof. Proved directly: the
    `(jcol, jcol')` entry is `∑ ri, star(K ri jcol) * K ri jcol'`, which
    reindexes (via `multiIndexEquiv I`) to a sum over row multi-indices,
    factors as `∏ d, ((U d)ᴴ * U d) (col_d) (col'_d)` (the prod-of-sums ↔
    sum-over-multi-index identity `Fintype.prod_sum`), and collapses to
    `∏ d, (1) (col_d) (col'_d) = (1) jcol jcol'` since each factor Gram is the
    identity. -/
theorem kroneckerAll_orthonormalCols
    (U : (d : Fin D) → Matrix (Fin (I d)) (Fin (J d)) 𝕜)
    (hU : ∀ d, (U d)ᴴ * (U d) = 1) :
    (kroneckerAll U)ᴴ * (kroneckerAll U) = 1 := by
  funext jcol jcol'
  rw [Matrix.mul_apply]
  -- Entry: `∑ ri, (kroneckerAll U)ᴴ jcol ri * kroneckerAll U ri jcol'`.
  simp only [Matrix.conjTranspose_apply, kroneckerAll_apply]
  -- `∑ ri, star(∏ d, U d (rowIdx d) (col_d)) * (∏ d, U d (rowIdx d) (col'_d))`.
  -- Reindex the `ri`-sum over row multi-indices via `multiIndexEquiv I`.
  rw [← Equiv.sum_comp (multiIndexEquiv I)
    (fun ri => star (∏ d : Fin D, U d ((multiIndexEquiv I).symm ri d)
        ((multiIndexEquiv J).symm jcol d)) *
      ∏ d : Fin D, U d ((multiIndexEquiv I).symm ri d) ((multiIndexEquiv J).symm jcol' d))]
  simp only [Equiv.symm_apply_apply]
  -- `∑ idx, star(∏ d, U d (idx d) (col_d)) * (∏ d, U d (idx d) (col'_d))`.
  -- Push `star` into the product, then merge the two products into one.
  simp only [star_prod]
  rw [show (∑ idx : (d : Fin D) → Fin (I d),
        (∏ d : Fin D, star (U d (idx d) ((multiIndexEquiv J).symm jcol d))) *
          ∏ d : Fin D, U d (idx d) ((multiIndexEquiv J).symm jcol' d))
      = ∑ idx : (d : Fin D) → Fin (I d),
          ∏ d : Fin D, (star (U d (idx d) ((multiIndexEquiv J).symm jcol d)) *
            U d (idx d) ((multiIndexEquiv J).symm jcol' d)) from
    Finset.sum_congr rfl (fun idx _ => (Finset.prod_mul_distrib).symm)]
  -- Now `∑ idx, ∏ d, g d (idx d)`; turn it into `∏ d, ∑ i, g d i` via `Fintype.prod_sum`.
  rw [← Fintype.prod_sum (fun (d : Fin D) (i : Fin (I d)) =>
    star (U d i ((multiIndexEquiv J).symm jcol d)) *
      U d i ((multiIndexEquiv J).symm jcol' d))]
  -- Each factor `∑ i, star(U d i col_d) * U d i col'_d = ((U d)ᴴ * U d) col_d col'_d`.
  have hfac : ∀ d : Fin D,
      (∑ i : Fin (I d), star (U d i ((multiIndexEquiv J).symm jcol d)) *
        U d i ((multiIndexEquiv J).symm jcol' d))
      = (1 : Matrix (Fin (J d)) (Fin (J d)) 𝕜)
          ((multiIndexEquiv J).symm jcol d) ((multiIndexEquiv J).symm jcol' d) := by
    intro d
    rw [← hU d, Matrix.mul_apply]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Matrix.conjTranspose_apply]
  rw [Finset.prod_congr rfl (fun d _ => hfac d)]
  -- `∏ d, (1) (col_d) (col'_d) = (1) jcol jcol'`.
  rw [Matrix.one_apply]
  by_cases hjj : jcol = jcol'
  · -- Diagonal: both sides are `1`. Each `col_d = col'_d`, so every factor is `1`.
    subst hjj
    simp
  · -- Off-diagonal: the multi-indices differ at some mode, killing the product.
    rw [if_neg hjj]
    -- The injective `(multiIndexEquiv J).symm` sends `jcol ≠ jcol'` to differing
    -- multi-indices, so some mode `d₀` has `col_{d₀} ≠ col'_{d₀}`.
    obtain ⟨d₀, hd₀⟩ : ∃ d, (multiIndexEquiv J).symm jcol d ≠ (multiIndexEquiv J).symm jcol' d := by
      by_contra hc
      push Not at hc
      exact hjj ((multiIndexEquiv J).symm.injective (funext hc))
    refine Finset.prod_eq_zero (Finset.mem_univ d₀) ?_
    rw [Matrix.one_apply, if_neg hd₀]

end Tensor
