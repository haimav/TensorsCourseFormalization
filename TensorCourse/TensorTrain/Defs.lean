/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.ContractiveProd
import TensorCourse.Tensor.ModeSplit

/-!
# Tensor Train Decomposition

The **tensor train (TT) decomposition** of an order-`D` tensor
`X ∈ 𝕜^{I₁×⋯×I_D}` writes it via `D` core tensors `T⁽ᵏ⁾`, with the index
formula (`TT-element-index`):

  `X(i₁,…,i_D) = ∑_{r₁,…,r_{D-1}} T⁽¹⁾(i₁,r₁) T⁽²⁾(r₁,i₂,r₂) ⋯ T⁽ᴰ⁾(r_{D-1},i_D)`.

We represent all cores **uniformly** as order-3 arrays with **bond
dimensions** `R : Fin (D+1) → ℕ`: core `k : Fin D` is
`T k : Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜`, connecting the
left bond `R k.castSucc` to the right bond `R k.succ`. A *genuine* TT
decomposition has unit boundary bonds `R 0 = R D = 1` (so `T⁽¹⁾`, `T⁽ᴰ⁾` are
effectively matrices, as in the lecture). The lecture endorses this uniform
"order-3 cores" view (also the physics MPS convention).

The **TT-ranks** are the internal bond dimensions `R 1, …, R (D-1)`.

## Main definitions

* `Tensor.ttTensor R T` — the TT reconstruction (index-sum form).
* `Tensor.HasTTDecomp X R` — `X` admits a TT decomposition with bond
  dimensions `R` and unit boundary bonds.

## Main results

* `Tensor.ttTensor_apply` — the defining index-sum.

## Decoupling from the contractive product

`ttTensor` (index-sum) is the workhorse carrying the structural theorems
(existence, error). The contractive product `⋈ₜ` (`Tensor.ctrProd`) is the
lecture's *operation* `X = T⁽¹⁾ ⋈ₜ ⋯ ⋈ₜ T⁽ᴰ⁾`. They meet at the
contractive-product matrix form (`modeSplit_ctrProd`), the bridge underlying
the lecture's "`TT-element-index` ⟺ `TT-decomposition`" exercise.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The **TT reconstruction** of a tensor from its cores, in index-sum form.
    Bond dimensions `R : Fin (D+1) → ℕ`; core `k` is indexed by its left bond
    `R k.castSucc`, output mode `I k`, and right bond `R k.succ`. The entry is
    the sum over all bond multi-indices of the product of the cores. -/
def ttTensor {D : ℕ} {I : Fin D → ℕ} (R : Fin (D + 1) → ℕ)
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜) :
    Tensor 𝕜 D I :=
  fun idx =>
    ∑ r : ((k : Fin (D + 1)) → Fin (R k)),
      ∏ k : Fin D, T k (r k.castSucc) (idx k) (r k.succ)

/-- The defining index-sum of the TT reconstruction. -/
theorem ttTensor_apply {D : ℕ} {I : Fin D → ℕ} (R : Fin (D + 1) → ℕ)
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin D) → Fin (I d)) :
    ttTensor R T idx =
      ∑ r : ((k : Fin (D + 1)) → Fin (R k)),
        ∏ k : Fin D, T k (r k.castSucc) (idx k) (r k.succ) :=
  rfl

/-- **`X` admits a TT decomposition with bond dimensions `R`.** The boundary
    bonds are unit (`R 0 = R D = 1`), so the first and last cores are
    effectively matrices, and the internal bonds `R 1, …, R (D-1)` are the
    TT-ranks. -/
def HasTTDecomp {D : ℕ} {I : Fin D → ℕ} (X : Tensor 𝕜 D I) (R : Fin (D + 1) → ℕ) :
    Prop :=
  R 0 = 1 ∧ R (Fin.last D) = 1 ∧
    ∃ T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜,
      X = ttTensor R T

/-! ### Core-tensor ↔ curried-core conversions

A TT core is naturally an order-3 tensor, but `ttTensor` consumes the cores in
curried form. `chainCore`/`uncurryCore` convert between the two: the order-3
tensor uses the `Fin.snoc ![R₀, out₀] R₁` shape (modes: left bond, output, right
bond — the `⋈ₜ`-left-operand shape), and `uncurryCore` reads it back as a curried
`Fin R₀ → Fin out₀ → Fin R₁ → 𝕜`. These mediate between the tensor-valued cores
of the packaged `TTDecomp` and the curried-core `ttTensor` development. -/

/-- The first TT core `T⁰ : R₀ × out₀ × R₁`, as an order-3 tensor with the
    right bond `R₁` as its **last** mode (the `⋈ₜ`-left-operand shape). The two
    free modes are the left bond `R₀` and the output `out₀`. -/
def chainCore {R0 out0 R1 : ℕ} (T0 : Fin R0 → Fin out0 → Fin R1 → 𝕜) :
    Tensor 𝕜 (2 + 1) (Fin.snoc ![R0, out0] R1) :=
  fun idx =>
    T0 (Fin.cast (by simp) (idx (Fin.castSucc 0)))
       (Fin.cast (by simp) (idx (Fin.castSucc 1)))
       (Fin.cast (by simp) (idx (Fin.last 2)))

theorem chainCore_apply {R0 out0 R1 : ℕ} (T0 : Fin R0 → Fin out0 → Fin R1 → 𝕜)
    (idx : (d : Fin (2 + 1)) → Fin ((Fin.snoc ![R0, out0] R1 : Fin (2 + 1) → ℕ) d)) :
    chainCore T0 idx =
      T0 (Fin.cast (by simp) (idx (Fin.castSucc 0)))
         (Fin.cast (by simp) (idx (Fin.castSucc 1)))
         (Fin.cast (by simp) (idx (Fin.last 2))) := rfl

/-- Read the `(R₀, out₀, R₁)`-core back out of an order-3 tensor of
    `chainCore` shape (the inverse of `chainCore`). -/
def uncurryCore {R0 out0 R1 : ℕ} (A : Tensor 𝕜 (2 + 1) (Fin.snoc ![R0, out0] R1)) :
    Fin R0 → Fin out0 → Fin R1 → 𝕜 :=
  fun a b c => A (Fin.lastCases (Fin.cast (by simp) c)
    (Fin.cons (Fin.cast (by simp) a) (Fin.cons (Fin.cast (by simp) b) (fun i => i.elim0))))

/-- `chainCore` recovers any order-3 tensor of `chainCore` shape from its
    uncurried core. -/
theorem chainCore_uncurryCore {R0 out0 R1 : ℕ}
    (A : Tensor 𝕜 (2 + 1) (Fin.snoc ![R0, out0] R1)) :
    chainCore (uncurryCore A) = A := by
  funext idx
  rw [chainCore_apply, uncurryCore]
  congr 1
  funext d
  refine Fin.lastCases ?_ (fun i => ?_) d
  · simp only [Fin.lastCases_last]; apply Fin.ext; simp
  · simp only [Fin.lastCases_castSucc]
    refine Fin.cases ?_ (fun j => ?_) i
    · simp only [Fin.cons_zero]; apply Fin.ext; simp
    · simp only [Fin.cons_succ]
      refine Fin.cases ?_ (fun k => k.elim0) j
      simp only [Fin.cons_zero]; apply Fin.ext; simp

/-- `uncurryCore` recovers any curried core from its `chainCore` tensor (the
    reverse round-trip of `chainCore_uncurryCore`). -/
theorem uncurryCore_chainCore {R0 out0 R1 : ℕ} (T0 : Fin R0 → Fin out0 → Fin R1 → 𝕜) :
    uncurryCore (chainCore T0) = T0 := by
  funext a b c
  rw [uncurryCore, chainCore_apply]
  refine congr (congr (congrArg T0 ?_) ?_) ?_
  · rw [Fin.lastCases_castSucc, Fin.cons_zero]; apply Fin.ext; simp
  · rw [Fin.lastCases_castSucc, Fin.cons_one, Fin.cons_zero]; apply Fin.ext; simp
  · rw [Fin.lastCases_last]; apply Fin.ext; simp

/-! ### The packaged decomposition `TTDecomp`

The index-sum reconstruction `ttTensor` is the primary, general-`D` API; the
structural theory (existence, the TT-SVD error bound) is phrased on it.
`TTDecomp` is a thin packaging on top — a named home for the lecture's noun
*"a TT decomposition"* (a list of order-3 core tensors) and the object the
existence theorem produces — mirroring how `CPDecomp` packages a CP
decomposition and `TuckerDecomp` a Tucker decomposition. Its reconstruction
`TTDecomp.tensor` is *definitionally* `ttTensor` of the curried cores (via the
accessor `coreFn`), so every `ttTensor` lemma applies to it after `tensor_eq`.

The cores are stored as genuine order-3 **tensors** (parallel to Tucker's tensor
core), while the curried accessor `coreFn` (`= uncurryCore`) plugs them into the
`ttTensor` development. The core shape `Fin.snoc ![R k.castSucc, I k] (R k.succ)`
reads (left bond, output, right bond) and is the `⋈ₜ`-left-operand shape, so a
stored core is also directly usable in the contractive-product chain. -/

/-- A **TT decomposition** packaged as a single object: one order-3 core tensor
    per mode, with bond dimensions `R : Fin (D+1) → ℕ`. Core `k` has shape
    `Fin.snoc ![R k.castSucc, I k] (R k.succ)` — modes (left bond `R k.castSucc`,
    output `I k`, right bond `R k.succ`). Mirrors `CPDecomp`/`TuckerDecomp`. -/
structure TTDecomp (𝕜 : Type*) [RCLike 𝕜] (D : ℕ) (I : Fin D → ℕ)
    (R : Fin (D + 1) → ℕ) where
  /-- `cores k` is the `k`-th order-3 core tensor, modes
      (left bond `R k.castSucc`, output `I k`, right bond `R k.succ`). -/
  cores : (k : Fin D) → Tensor 𝕜 (2 + 1) (Fin.snoc ![R k.castSucc, I k] (R k.succ))

/-- The **curried accessor**: core `k` read as
    `Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜` via `uncurryCore`.
    This is the form `ttTensor` consumes, so the packaged reconstruction reuses
    the entire `ttTensor` development unchanged. -/
def TTDecomp.coreFn {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (tt : TTDecomp 𝕜 D I R) :
    (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜 :=
  fun k => uncurryCore (tt.cores k)

/-- Reconstruct the tensor from a packaged TT decomposition. By definition this
    is the curried-core index-sum `ttTensor R tt.coreFn`. -/
def TTDecomp.tensor {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (tt : TTDecomp 𝕜 D I R) : Tensor 𝕜 D I :=
  ttTensor R tt.coreFn

/-- `TTDecomp.tensor` is definitionally `ttTensor` of the curried cores,
    exposing the whole `ttTensor` API to packaged decompositions (mirrors
    `TuckerDecomp.tensor_eq`). -/
@[simp] theorem TTDecomp.tensor_eq {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (tt : TTDecomp 𝕜 D I R) :
    tt.tensor = ttTensor R tt.coreFn := rfl

/-- A packaged TT decomposition is **proper** when its boundary bonds are unit
    (`R 0 = R D = 1`) — the genuine-TT condition: the first and last cores are
    effectively matrices and `R 1, …, R (D-1)` are the TT-ranks (as opposed to an
    open boundary / MPS). Mirrors `TuckerDecomp.IsOrthonormal` as the named
    "good" condition on the packaged object. -/
def TTDecomp.IsProper {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (_tt : TTDecomp 𝕜 D I R) : Prop :=
  R 0 = 1 ∧ R (Fin.last D) = 1

/-- Build a packaged `TTDecomp` from curried cores (inverse of `coreFn`), via
    `chainCore`. Lets the existence/error theory — phrased with curried cores —
    produce packaged `TTDecomp` objects. -/
def TTDecomp.ofCurried {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜) :
    TTDecomp 𝕜 D I R where
  cores := fun k => chainCore (T k)

@[simp] theorem TTDecomp.coreFn_ofCurried {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜) :
    (TTDecomp.ofCurried T).coreFn = T := by
  funext k
  exact uncurryCore_chainCore (T k)

@[simp] theorem TTDecomp.tensor_ofCurried {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜) :
    (TTDecomp.ofCurried T).tensor = ttTensor R T := by
  rw [TTDecomp.tensor_eq, TTDecomp.coreFn_ofCurried]

/-- **Bridge to `HasTTDecomp`.** `X` admits a TT decomposition with bond
    dimensions `R` (unit boundary) iff there is a *proper* packaged `TTDecomp`
    reconstructing `X`. Connects the `Prop`-level predicate to the packaged
    noun form. -/
theorem hasTTDecomp_iff_ttDecomp {D : ℕ} {I : Fin D → ℕ} {R : Fin (D + 1) → ℕ}
    {X : Tensor 𝕜 D I} :
    HasTTDecomp X R ↔
      (R 0 = 1 ∧ R (Fin.last D) = 1) ∧ ∃ tt : TTDecomp 𝕜 D I R, tt.tensor = X := by
  rw [HasTTDecomp]
  constructor
  · rintro ⟨h0, hlast, T, hX⟩
    refine ⟨⟨h0, hlast⟩, TTDecomp.ofCurried T, ?_⟩
    rw [TTDecomp.tensor_ofCurried]; exact hX.symm
  · rintro ⟨⟨h0, hlast⟩, tt, hX⟩
    exact ⟨h0, hlast, tt.coreFn, (hX.symm).trans tt.tensor_eq⟩

/-! ### The contractive-product matrix form (`eq:contractive-product-matrix-form`)

The contractive product `A ⋈ₜ B` has a matrix form: its mode split (after
`A`'s free modes) is the matrix product of the "last-mode matricization" of
`A` with the "first-mode matricization" of `B`, with inner dimension the
shared mode `S`. This is the lecture's Eq (eq:contractive-product-matrix-form)
and immediately gives the split-rank bound `rank (A ⋈ₜ B)_{[a]} ≤ S`. -/

variable {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ} {S : ℕ}

/-- Matricization of `A`'s **last** mode: rows index the free modes `da`,
    columns the contracted mode `S`. -/
def ctrLeftMat (A : Tensor 𝕜 (a + 1) (Fin.snoc da S)) :
    Matrix (Fin (∏ d, da d)) (Fin S) 𝕜 :=
  Matrix.of fun ri s =>
    A (Fin.lastCases (motive := fun d => Fin ((Fin.snoc da S : Fin (a + 1) → ℕ) d))
        (Fin.cast (@Fin.snoc_last a (fun _ => ℕ) S da).symm s)
        (fun i => Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i).symm
          ((multiIndexEquiv da).symm ri i)))

/-- Matricization of `B`'s **first** mode: rows index the contracted mode
    `S`, columns the free modes `db`. -/
def ctrRightMat (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    Matrix (Fin S) (Fin (∏ d, db d)) 𝕜 :=
  Matrix.of fun s cj =>
    B (Fin.cons (α := fun d => Fin ((Fin.cons S db : Fin (b + 1) → ℕ) d))
        (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) S db).symm s)
        (fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i).symm
          ((multiIndexEquiv db).symm cj i)))

/-- Inverse of `ctrLeftMat`: fold a `(∏ I₁) × S` matrix back into an
    order-`(p+1)` tensor whose last mode is the `S`-mode. -/
def ctrLeftFold (M : Matrix (Fin (∏ d, da d)) (Fin S) 𝕜) :
    Tensor 𝕜 (a + 1) (Fin.snoc da S) :=
  fun idx =>
    M (multiIndexEquiv da (fun i =>
        Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i) (idx (Fin.castSucc i))))
      (Fin.cast (@Fin.snoc_last a (fun _ => ℕ) S da) (idx (Fin.last a)))

/-- Inverse of `ctrRightMat`: fold an `S × (∏ I₂)` matrix back into an
    order-`(q+1)` tensor whose first mode is the `S`-mode. -/
def ctrRightFold (M : Matrix (Fin S) (Fin (∏ d, db d)) 𝕜) :
    Tensor 𝕜 (b + 1) (Fin.cons S db) :=
  fun idx =>
    M (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) S db) (idx 0))
      (multiIndexEquiv db (fun i =>
        Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i) (idx (Fin.succ i))))

/-- `ctrLeftMat` inverts `ctrLeftFold`. -/
@[simp]
theorem ctrLeftMat_ctrLeftFold (M : Matrix (Fin (∏ d, da d)) (Fin S) 𝕜) :
    ctrLeftMat (ctrLeftFold M) = M := by
  funext ri s
  rw [ctrLeftMat, Matrix.of_apply, ctrLeftFold]
  simp only [Fin.lastCases_castSucc, Fin.lastCases_last]
  congr 1
  · have h : (fun i => Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i)
        (Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i).symm
          ((multiIndexEquiv da).symm ri i)))
        = (multiIndexEquiv da).symm ri := by
      funext i; apply Fin.ext; simp
    rw [h, Equiv.apply_symm_apply]

/-- `ctrRightMat` inverts `ctrRightFold`. -/
@[simp]
theorem ctrRightMat_ctrRightFold (M : Matrix (Fin S) (Fin (∏ d, db d)) 𝕜) :
    ctrRightMat (ctrRightFold M) = M := by
  funext s cj
  rw [ctrRightMat, Matrix.of_apply, ctrRightFold]
  simp only [Fin.cons_zero, Fin.cons_succ]
  congr 1
  · have h : (fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i)
        (Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i).symm
          ((multiIndexEquiv db).symm cj i)))
        = (multiIndexEquiv db).symm cj := by
      funext i; apply Fin.ext; simp
    rw [h, Equiv.apply_symm_apply]

/-- **Contractive-product matrix form (`eq:contractive-product-matrix-form`).**
    The mode split of `A ⋈ₜ B` (grouping `A`'s free modes in the rows) is the
    matrix product of the two boundary matricizations:

      `modeSplit (A ⋈ₜ B) = ctrLeftMat A * ctrRightMat B`. -/
theorem modeSplit_ctrProd (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    modeSplit (A ⋈ₜ B) = ctrLeftMat A * ctrRightMat B := by
  funext ri cj
  rw [modeSplit_apply, ctrProd_apply, Matrix.mul_apply]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [ctrLeftMat, ctrRightMat, Matrix.of_apply, Matrix.of_apply,
    splitLeft_combineIndex, splitRight_combineIndex]

/-- **Split-rank bound for the contractive product.** The split rank of
    `A ⋈ₜ B` (cutting between `A`'s and `B`'s free modes) is at most the
    shared bond dimension `S`. (A multiplication can only decrease rank, and
    `ctrLeftMat A` has `S` columns.) -/
theorem splitRank_ctrProd_le (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    splitRank (A ⋈ₜ B) ≤ S := by
  rw [splitRank_def, modeSplit_ctrProd]
  calc (ctrLeftMat A * ctrRightMat B).rank
      ≤ (ctrLeftMat A).rank := Matrix.rank_mul_le_left _ _
    _ ≤ S := Matrix.rank_le_width _

end Tensor
