/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.MatMulTensor.Defs
import TensorCourse.Tensor.OuterProd
import TensorCourse.Tensor.Reshape
import TensorCourse.CPDecomposition.ReshapeRank
import Mathlib.Logic.Equiv.Prod
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Outer Product of Matrix Multiplication Tensors and Mode Fusion

This file proves that the matrix multiplication
tensor `matMulTensor 𝕜 (M * N)` is obtained from the outer product
`matMulTensor 𝕜 M ◯ₜ matMulTensor 𝕜 N` by a mode-fusing reindex. Concretely, if we
label the 6 modes of the outer product as `0, 1, 2, 3, 4, 5` (with modes `0, 1, 2`
coming from the first tensor and modes `3, 4, 5` from the second), we fuse modes
`{0, 3}`, `{1, 4}`, `{2, 5}` using the canonical bijection
`Fin (M*M) × Fin (N*N) ≃ Fin ((M*N) * (M*N))`.

The fusion is stated in the form `fuseEquiv σ fe`, where:

* `σ := Fin.append id id : Fin 6 → Fin 3` assigns each of the 6 modes to one of
  3 target modes (pairing mode `d` from the first copy with mode `d` from the
  second copy).
* `fe d₂` is an explicit per-fiber equivalence built from a "block-transpose" shuffle
  `(Fin M × Fin M) × (Fin N × Fin N) ≃ (Fin M × Fin N) × (Fin M × Fin N)`, i.e.
  `Equiv.prodProdProdComm`.

## Main definitions

* `Tensor.fuseSigma₃` — the mode-assignment map `Fin 6 → Fin 3`
* `Tensor.outerDimsEquiv` — the full per-fiber equivalence
  `((d₁ : fiber d₂) → Fin (dims d₁.val)) ≃ Fin ((M*N) * (M*N))`

## Main results

* `Tensor.matMulTensor_eq_fuse_outerProd` — `matMulTensor 𝕜 (M*N)` equals a mode-fusing
  reindex of `matMulTensor 𝕜 M ◯ₜ matMulTensor 𝕜 N`.

This is the structural step linking outer-product CP decompositions of
`matMulTensor` at two different sizes to a single CP decomposition at the
product size; combined with `cpRank_reindex_fuse_le` it yields the CP rank
bound used by `Lecture1.Theorem4`.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### The mode-assignment map σ -/

/-- The mode-assignment map for the outer-product-to-MMT fusion: modes
    `0, 1, 2, 3, 4, 5 : Fin 6` are sent to `0, 1, 2, 0, 1, 2 : Fin 3` respectively.
    Defined via `Fin.append id id` so that
    `σ (castAdd 3 d) = d` and `σ (natAdd 3 d) = d`. -/
def fuseSigma₃ : Fin 6 → Fin 3 :=
  Fin.append (fun d : Fin 3 => d) (fun d : Fin 3 => d)

@[simp]
theorem fuseSigma₃_castAdd (d : Fin 3) :
    fuseSigma₃ (Fin.castAdd 3 d) = d := by
  unfold fuseSigma₃
  rw [Fin.append_left]

@[simp]
theorem fuseSigma₃_natAdd (d : Fin 3) :
    fuseSigma₃ (Fin.natAdd 3 d) = d := by
  unfold fuseSigma₃
  rw [Fin.append_right]

/-! ### Helper lemmas for `combinedDims` applied to concrete indices -/

/-- Unfolds `combinedDims (matMulDims M) (matMulDims N)` on the left half. -/
theorem combinedDims_castAdd_matMul (M N : ℕ) (d : Fin 3) :
    combinedDims (matMulDims M) (matMulDims N) (Fin.castAdd 3 d) = M * M := by
  rw [combinedDims_castAdd]
  rfl

/-- Unfolds `combinedDims (matMulDims M) (matMulDims N)` on the right half. -/
theorem combinedDims_natAdd_matMul (M N : ℕ) (d : Fin 3) :
    combinedDims (matMulDims M) (matMulDims N) (Fin.natAdd 3 d) = N * N := by
  rw [combinedDims_natAdd]
  rfl

/-! ### The per-fiber equivalence

    For each target mode `d₂ : Fin 3`, the fiber `{d₁ : Fin 6 // fuseSigma₃ d₁ = d₂}`
    has exactly two elements (`castAdd 3 d₂` and `natAdd 3 d₂`). A function over
    this fiber with values in `Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val)`
    is therefore equivalent to a pair in `Fin (M*M) × Fin (N*N)`.
-/

/-- Extract the value of a fiber-function at the "left" element `castAdd 3 d₂`,
    as an element of `Fin (M*M)`. -/
def extractLeft (M N : ℕ) (d₂ : Fin 3)
    (f : (d₁ : { d₁ : Fin 6 // fuseSigma₃ d₁ = d₂ }) →
      Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val)) :
    Fin (M * M) :=
  (f ⟨Fin.castAdd 3 d₂, fuseSigma₃_castAdd d₂⟩).cast (combinedDims_castAdd_matMul M N d₂)

/-- Extract the value of a fiber-function at the "right" element `natAdd 3 d₂`,
    as an element of `Fin (N*N)`. -/
def extractRight (M N : ℕ) (d₂ : Fin 3)
    (f : (d₁ : { d₁ : Fin 6 // fuseSigma₃ d₁ = d₂ }) →
      Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val)) :
    Fin (N * N) :=
  (f ⟨Fin.natAdd 3 d₂, fuseSigma₃_natAdd d₂⟩).cast (combinedDims_natAdd_matMul M N d₂)

/-- Build a fiber-function from a pair `(a, b) ∈ Fin (M*M) × Fin (N*N)`.
    Uses `Fin.addCases` on the underlying `Fin 6` index. -/
def buildFiber (M N : ℕ) (d₂ : Fin 3) (p : Fin (M * M) × Fin (N * N)) :
    (d₁ : { d₁ : Fin 6 // fuseSigma₃ d₁ = d₂ }) →
      Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val) :=
  fun ⟨d₁, _⟩ =>
    Fin.addCases
      (motive := fun d₁ => Fin (combinedDims (matMulDims M) (matMulDims N) d₁))
      (fun d => (p.1).cast (combinedDims_castAdd_matMul M N d).symm)
      (fun d => (p.2).cast (combinedDims_natAdd_matMul M N d).symm)
      d₁

/-- For each target mode `d₂ : Fin 3`, a function over the 2-element fiber
    `σ⁻¹(d₂)` is equivalent to a pair in `Fin (M*M) × Fin (N*N)`. -/
def fiberPairEquiv (M N : ℕ) (d₂ : Fin 3) :
    ((d₁ : { d₁ : Fin 6 // fuseSigma₃ d₁ = d₂ }) →
      Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val)) ≃
    Fin (M * M) × Fin (N * N) where
  toFun f := (extractLeft M N d₂ f, extractRight M N d₂ f)
  invFun := buildFiber M N d₂
  left_inv f := by
    funext ⟨d₁, hd₁⟩
    -- Case-split on d₁ via Fin.addCases
    refine Fin.addCases (m := 3) (n := 3)
      (motive := fun d₁ => ∀ (hd₁ : fuseSigma₃ d₁ = d₂),
        buildFiber M N d₂ (extractLeft M N d₂ f, extractRight M N d₂ f)
          ⟨d₁, hd₁⟩ = f ⟨d₁, hd₁⟩) ?_ ?_ d₁ hd₁
    · intro d h
      -- h : fuseSigma₃ (castAdd 3 d) = d₂
      have hdeq : d = d₂ := by
        have := h
        rw [fuseSigma₃_castAdd] at this
        exact this
      subst hdeq
      simp only [buildFiber, Fin.addCases_left, extractLeft]
      rfl
    · intro d h
      have hdeq : d = d₂ := by
        have := h
        rw [fuseSigma₃_natAdd] at this
        exact this
      subst hdeq
      simp only [buildFiber, Fin.addCases_right, extractRight]
      rfl
  right_inv p := by
    rcases p with ⟨p1, p2⟩
    ext
    · simp only [extractLeft, buildFiber, Fin.addCases_left]
      rfl
    · simp only [extractRight, buildFiber, Fin.addCases_right]
      rfl

/-- The 4-way shuffle equivalence for block indices:
    `(Fin M × Fin M) × (Fin N × Fin N) ≃ (Fin M × Fin N) × (Fin M × Fin N)`.
    This is the block-transpose identifying `((row_A, col_A), (row_B, col_B))`
    with `((big_row, big_col_for_this_block), ...)` for the combined matrix. -/
def blockShuffle (M N : ℕ) :
    (Fin M × Fin M) × (Fin N × Fin N) ≃ (Fin M × Fin N) × (Fin M × Fin N) :=
  Equiv.prodProdProdComm (Fin M) (Fin M) (Fin N) (Fin N)

/-- The full per-fiber equivalence for the mode-fusion: for each target mode
    `d₂ : Fin 3`, a function on the 2-element fiber `σ⁻¹(d₂)` with values in the
    combined dimensions is equivalent to `Fin (matMulDims (M*N) d₂) = Fin ((M*N)*(M*N))`.

    The construction chains:
    1. `fiberPairEquiv` — enumerate the fiber as `Fin (M*M) × Fin (N*N)`
    2. Unpair each factor into `(Fin M × Fin M) × (Fin N × Fin N)`
    3. `blockShuffle` — shuffle to `(Fin M × Fin N) × (Fin M × Fin N)`
    4. Pair each factor into `Fin (M*N) × Fin (M*N)`
    5. Pair the two into `Fin ((M*N) * (M*N))`. -/
def outerDimsEquiv (M N : ℕ) (d₂ : Fin 3) :
    ((d₁ : { d₁ : Fin 6 // fuseSigma₃ d₁ = d₂ }) →
      Fin (combinedDims (matMulDims M) (matMulDims N) d₁.val)) ≃
    Fin (matMulDims (M * N) d₂) :=
  (fiberPairEquiv M N d₂).trans <|
    (Equiv.prodCongr finProdFinEquiv.symm finProdFinEquiv.symm).trans <|
      (blockShuffle M N).trans <|
        (Equiv.prodCongr finProdFinEquiv finProdFinEquiv).trans finProdFinEquiv

/-! ### The main theorem -/

/-- Key helper: after decoding via `fuseEquiv` and `outerDimsEquiv`, the
    `finProdFinEquiv.symm` of the left-half of the unfused index at mode `d`
    is the pair `(a_d, c_d)` where the doubly-unpaired `idx d` is
    `((a_d, b_d), (c_d, d_d))`. -/
theorem finProdFinEquiv_symm_splitLeft (M N : ℕ)
    (idx : (d : Fin 3) → Fin (matMulDims (M * N) d)) (d : Fin 3) :
    (finProdFinEquiv.symm
        (splitLeft (dims₁ := matMulDims M) (dims₂ := matMulDims N)
          ((fuseEquiv fuseSigma₃ (outerDimsEquiv M N)).symm idx) d) :
        Fin M × Fin M) =
      ((finProdFinEquiv.symm (finProdFinEquiv.symm (idx d)).1).1,
       (finProdFinEquiv.symm (finProdFinEquiv.symm (idx d)).2).1) := by
  unfold splitLeft
  simp only [fuseEquiv, Equiv.coe_fn_symm_mk]
  unfold outerDimsEquiv
  simp only [Equiv.symm_trans_apply, Equiv.prodCongr_symm, Equiv.symm_symm]
  unfold blockShuffle
  simp only [Equiv.prodProdProdComm_symm]
  simp only [fiberPairEquiv, Equiv.coe_fn_symm_mk]
  unfold buildFiber
  simp only [Fin.addCases_left]
  rw [fuseSigma₃_castAdd]
  simp only [Equiv.prodProdProdComm_apply, Equiv.prodCongr_apply, Prod.map_fst, Prod.map_snd]
  -- Now the goal is:
  --   finProdFinEquiv.symm (Fin.cast h1 (Fin.cast h2 (finProdFinEquiv (a, c)))) = (a, c)
  -- where h1, h2 are proof terms. The casts are trivial (identity) since both
  -- endpoints are M*M; use Fin.ext + val equality to bypass the cast abstraction.
  apply finProdFinEquiv.injective
  rw [Equiv.apply_symm_apply]
  -- Goal: Fin.cast h1 (Fin.cast h2 (finProdFinEquiv (a, c))) = finProdFinEquiv (a, c)
  -- Both sides are Fin (M*M); show they're equal by val
  ext
  -- The remaining goal is a Nat equation involving divNat/modNat.
  simp only [Fin.val_cast, finProdFinEquiv_apply_val]
  rfl

/-- Similar helper for the right half. -/
theorem finProdFinEquiv_symm_splitRight (M N : ℕ)
    (idx : (d : Fin 3) → Fin (matMulDims (M * N) d)) (d : Fin 3) :
    (finProdFinEquiv.symm
        (splitRight (dims₁ := matMulDims M) (dims₂ := matMulDims N)
          ((fuseEquiv fuseSigma₃ (outerDimsEquiv M N)).symm idx) d) :
        Fin N × Fin N) =
      ((finProdFinEquiv.symm (finProdFinEquiv.symm (idx d)).1).2,
       (finProdFinEquiv.symm (finProdFinEquiv.symm (idx d)).2).2) := by
  unfold splitRight
  simp only [fuseEquiv, Equiv.coe_fn_symm_mk]
  unfold outerDimsEquiv
  simp only [Equiv.symm_trans_apply, Equiv.prodCongr_symm, Equiv.symm_symm]
  unfold blockShuffle
  simp only [Equiv.prodProdProdComm_symm]
  simp only [fiberPairEquiv, Equiv.coe_fn_symm_mk]
  unfold buildFiber
  simp only [Fin.addCases_right]
  rw [fuseSigma₃_natAdd]
  simp only [Equiv.prodProdProdComm_apply, Equiv.prodCongr_apply, Prod.map_fst, Prod.map_snd]
  apply finProdFinEquiv.injective
  rw [Equiv.apply_symm_apply]
  ext
  simp only [Fin.val_cast, finProdFinEquiv_apply_val]
  rfl

/-- **Result 3.** The matrix multiplication tensor of size `M*N` equals a
    mode-fusing reindex of the outer product of the matrix multiplication
    tensors of sizes `M` and `N`:

    `matMulTensor 𝕜 (M*N) = reindex (fuseEquiv fuseSigma₃ (outerDimsEquiv M N))
                                    (matMulTensor 𝕜 M ◯ₜ matMulTensor 𝕜 N)`

    The proof is an index chase: both sides evaluate to a Kronecker-delta-like
    conditional on the decomposed indices, and the shuffle in `outerDimsEquiv`
    is exactly the identification of "independent M-block and N-block indices"
    with "a single (M*N)-block index". -/
theorem matMulTensor_eq_fuse_outerProd (M N : ℕ) :
    matMulTensor 𝕜 (M * N) =
      Tensor.reindex (fuseEquiv fuseSigma₃ (outerDimsEquiv M N))
        (matMulTensor 𝕜 M ◯ₜ matMulTensor 𝕜 N) := by
  ext idx
  simp only [reindex_apply, outerProd_apply, matMulTensor_apply,
    finProdFinEquiv_symm_splitLeft, finProdFinEquiv_symm_splitRight]
  -- The goal is now `(if P then 1 else 0) = (if Q then 1 else 0) * (if R then 1 else 0)`
  -- where P is the 3-mode MMT condition on idx (in Fin (M*N) × Fin (M*N) pairs),
  -- Q is the condition on (a_d, c_d) (first components of doubly-unpaired idx d),
  -- and R is the condition on (b_d, d_d) (second components).
  --
  -- Key fact: P ↔ (Q ∧ R). Given this, we show equality by splitting on whether P holds.
  have hPQR : ((finProdFinEquiv.symm (idx 0)).2 = (finProdFinEquiv.symm (idx 2)).2 ∧
               (finProdFinEquiv.symm (idx 1)).1 = (finProdFinEquiv.symm (idx 2)).1 ∧
               (finProdFinEquiv.symm (idx 0)).1 = (finProdFinEquiv.symm (idx 1)).2) ↔
              (((finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).2).1 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).2).1 ∧
                (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).1).1 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).1).1 ∧
                (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).1).1 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).2).1) ∧
               ((finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).2).2 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).2).2 ∧
                (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).1).2 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).1).2 ∧
                (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).1).2 =
                  (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).2).2)) := by
    constructor
    · rintro ⟨h1, h2, h3⟩
      have h1' := congrArg finProdFinEquiv.symm h1
      have h2' := congrArg finProdFinEquiv.symm h2
      have h3' := congrArg finProdFinEquiv.symm h3
      exact ⟨⟨congrArg Prod.fst h1', congrArg Prod.fst h2', congrArg Prod.fst h3'⟩,
             ⟨congrArg Prod.snd h1', congrArg Prod.snd h2', congrArg Prod.snd h3'⟩⟩
    · rintro ⟨⟨hM1, hM2, hM3⟩, ⟨hN1, hN2, hN3⟩⟩
      refine ⟨?_, ?_, ?_⟩
      · exact finProdFinEquiv.symm.injective (Prod.ext hM1 hN1)
      · exact finProdFinEquiv.symm.injective (Prod.ext hM2 hN2)
      · exact finProdFinEquiv.symm.injective (Prod.ext hM3 hN3)
  -- Now case-split on whether both conditions hold
  by_cases hP : (finProdFinEquiv.symm (idx 0)).2 = (finProdFinEquiv.symm (idx 2)).2 ∧
                (finProdFinEquiv.symm (idx 1)).1 = (finProdFinEquiv.symm (idx 2)).1 ∧
                (finProdFinEquiv.symm (idx 0)).1 = (finProdFinEquiv.symm (idx 1)).2
  · -- LHS is 1; need RHS to also be 1.
    rw [if_pos hP]
    obtain ⟨hQ, hR⟩ := hPQR.mp hP
    rw [if_pos hQ, if_pos hR]
    simp
  · -- LHS is 0; need RHS to also be 0.
    rw [if_neg hP]
    -- To show the product is 0, we show at least one factor is 0
    have : ¬ ((finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).2).1 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).2).1 ∧
            (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).1).1 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).1).1 ∧
            (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).1).1 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).2).1) ∨
          ¬ ((finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).2).2 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).2).2 ∧
            (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).1).2 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 2)).1).2 ∧
            (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 0)).1).2 =
              (finProdFinEquiv.symm (finProdFinEquiv.symm (idx 1)).2).2) := by
      by_contra hne
      rw [not_or] at hne
      obtain ⟨hneQ, hneR⟩ := hne
      rw [not_not] at hneQ hneR
      exact hP (hPQR.mpr ⟨hneQ, hneR⟩)
    rcases this with hQne | hRne
    · rw [if_neg hQne]; simp
    · rw [if_neg hRne]; simp

end Tensor
