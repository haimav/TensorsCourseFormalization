/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorTrain.Defs

/-!
# TT chain reconstruction: `ttTensor` as a chain of contractive products

This file establishes the lecture's "`TT-element-index` ⟺ `TT-decomposition`"
equivalence at the level needed for the existence induction: the index-sum
`ttTensor`, with its leading bond *exposed* (`ttOpen`), is the contractive
product of the first core with the open reconstruction of the remaining
cores. Iterating this is the tensor-train chain `T⁽¹⁾ ⋈ₜ ⋯ ⋈ₜ T⁽ᴰ⁾`.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- **Left-open TT reconstruction.** Like `ttTensor` but with the leading
    bond `R 0` *exposed* as the first output mode. The first coordinate of
    `idx` selects the value of bond `0`; the remaining bonds are summed. -/
def ttOpen {m : ℕ} {out : Fin m → ℕ} (R : Fin (m + 1) → ℕ)
    (T : (k : Fin m) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜) :
    Tensor 𝕜 (m + 1) (Fin.cons (R 0) out) :=
  fun idx =>
    ∑ r : (j : Fin (m + 1)) → Fin (R j),
      if (r 0).val = (idx 0).val then
        ∏ k : Fin m, T k (r k.castSucc)
          (Fin.cast (by rw [Fin.cons_succ]) (idx k.succ)) (r k.succ)
      else 0

theorem ttOpen_apply {m : ℕ} {out : Fin m → ℕ} (R : Fin (m + 1) → ℕ)
    (T : (k : Fin m) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin (m + 1)) → Fin ((Fin.cons (R 0) out : Fin (m + 1) → ℕ) d)) :
    ttOpen R T idx =
      ∑ r : (j : Fin (m + 1)) → Fin (R j),
        if (r 0).val = (idx 0).val then
          ∏ k : Fin m, T k (r k.castSucc)
            (Fin.cast (by rw [Fin.cons_succ]) (idx k.succ)) (r k.succ)
        else 0 := rfl

/-- Collapsing the leading-bond delta: `ttOpen` equals the sum over the
    *internal* bonds (positions `1 … m`), with the leading bond pinned to the
    exposed first coordinate `idx 0`. -/
theorem ttOpen_eq_sum_internal {m : ℕ} {out : Fin m → ℕ} (R : Fin (m + 1) → ℕ)
    (T : (k : Fin m) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin (m + 1)) → Fin ((Fin.cons (R 0) out : Fin (m + 1) → ℕ) d)) :
    ttOpen R T idx =
      ∑ r' : (j : Fin m) → Fin (R j.succ),
        ∏ k : Fin m, T k
          ((Fin.cons (idx 0) r' : (j : Fin (m + 1)) → Fin (R j)) k.castSucc)
          (Fin.cast (by rw [Fin.cons_succ]) (idx k.succ))
          ((Fin.cons (idx 0) r' : (j : Fin (m + 1)) → Fin (R j)) k.succ) := by
  rw [ttOpen_apply]
  -- reindex the bond sum r : ∀ j, Fin (R j)  ≃  Fin (R 0) × (∀ j : Fin m, Fin (R j.succ))
  rw [← Equiv.sum_comp (Fin.consEquiv (fun j => Fin (R j)))
        (fun r => if (r 0).val = (idx 0).val then
          ∏ k : Fin m, T k (r k.castSucc)
            (Fin.cast (by rw [Fin.cons_succ]) (idx k.succ)) (r k.succ) else 0)]
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl (fun r' _ => ?_)
  -- inner sum over r0 : Fin (R 0); the delta picks r0 with `r0.val = (idx 0).val`.
  set b0 : Fin (R 0) := Fin.cast (by rw [Fin.cons_zero]) (idx 0) with hb0
  rw [Finset.sum_eq_single_of_mem b0 (Finset.mem_univ _)]
  · -- chosen term: condition true and indices match `Fin.cons (idx 0) r'`
    simp only [Fin.consEquiv_apply, Fin.cons_zero]
    rw [if_pos (by rw [hb0]; simp [Fin.val_cast])]
    have hb : b0 = idx 0 := by rw [hb0]; apply Fin.ext; simp [Fin.val_cast]
    rw [hb]
  · -- other `r0 ≠ b0` give a false condition
    intro r0 _ hne
    simp only [Fin.consEquiv_apply, Fin.cons_zero]
    rw [if_neg]
    intro hc
    apply hne
    apply Fin.ext
    rw [hb0]
    simpa [Fin.val_cast] using hc

/-- **Peel the first core.** `ttOpen` of `m+1` cores is the sum over the first
    internal bond `R 1` of (first core) · (`ttOpen` of the remaining `m`
    cores), the latter evaluated with `R 1` exposed. This is the chain
    recursion `ttOpen R T = T⁽⁰⁾ ⋈ₜ ttOpen (tail)` at the index level. -/
theorem ttOpen_peel {m : ℕ} {out : Fin (m + 1) → ℕ} (R : Fin (m + 2) → ℕ)
    (T : (k : Fin (m + 1)) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin (m + 2)) → Fin ((Fin.cons (R 0) out : Fin (m + 2) → ℕ) d)) :
    ttOpen R T idx =
      ∑ s : Fin (R 1),
        T 0 (Fin.cast (by simp) (idx 0))
          (Fin.cast (by simp) (idx 1)) (Fin.cast (by simp) s) *
          ttOpen (fun j => R j.succ) (fun k => T k.succ)
            (Fin.cons (Fin.cast (by simp) s)
              (fun j => Fin.cast (by simp) (idx j.succ.succ))) := by
  rw [ttOpen_eq_sum_internal]
  simp only [Fin.prod_univ_succ]
  -- reindex the internal-bond sum, peeling the new bond `R 1` at position 0
  rw [← Equiv.sum_comp (Fin.consEquiv (fun j => Fin (R j.succ)))]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [ttOpen_eq_sum_internal, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun r'' _ => ?_)
  simp only [Fin.consEquiv_apply, Fin.cons_zero, Fin.cons_succ]
  rfl

/-! ### First-two-modes split and the `ttOpen` matrix form -/

/-- The **first-two-modes split**, stated natively in `m+2` form (no
    `Nat`-order friction). Groups modes `0,1` into the rows and the rest into
    the columns. -/
def modeSplit2 {m : ℕ} {dims : Fin (m + 2) → ℕ} (X : Tensor 𝕜 (m + 2) dims) :
    Matrix (Fin (dims 0 * dims 1)) (Fin (∏ j : Fin m, dims j.succ.succ)) 𝕜 :=
  Matrix.of fun ri cj =>
    X (Fin.cons (Fin.cast (by simp) (finProdFinEquiv.symm ri).1)
        (Fin.cons (Fin.cast (by simp) (finProdFinEquiv.symm ri).2)
          (fun j => Fin.cast (by simp) ((multiIndexEquiv (fun j => dims j.succ.succ)).symm cj j))))

theorem modeSplit2_apply {m : ℕ} {dims : Fin (m + 2) → ℕ} (X : Tensor 𝕜 (m + 2) dims)
    (ri : Fin (dims 0 * dims 1)) (cj : Fin (∏ j : Fin m, dims j.succ.succ)) :
    modeSplit2 X ri cj =
      X (Fin.cons (Fin.cast (by simp) (finProdFinEquiv.symm ri).1)
          (Fin.cons (Fin.cast (by simp) (finProdFinEquiv.symm ri).2)
            (fun j => Fin.cast (by simp)
              ((multiIndexEquiv (fun j => dims j.succ.succ)).symm cj j)))) := rfl

/-- `modeSplit2` is injective (a bijective reshape composed with `Matrix.of`). -/
theorem modeSplit2_injective {m : ℕ} {dims : Fin (m + 2) → ℕ} :
    Function.Injective (modeSplit2 (𝕜 := 𝕜) (dims := dims)) := by
  intro X Y h
  funext idx
  have h' := congr_fun (congr_fun h (finProdFinEquiv (idx 0, idx 1)))
    (multiIndexEquiv (fun j => dims j.succ.succ) (fun j => idx j.succ.succ))
  rw [modeSplit2_apply, modeSplit2_apply] at h'
  simp only [Equiv.symm_apply_apply] at h'
  -- the reconstructed index equals `idx`
  have hidx : (Fin.cons (Fin.cast (by simp) (idx 0))
      (Fin.cons (Fin.cast (by simp) (idx 1))
        (fun j => Fin.cast (by simp) (idx j.succ.succ)))
        : (d : Fin (m + 2)) → Fin (dims d)) = idx := by
    funext d
    refine Fin.cases ?_ (fun i => ?_) d
    · rw [Fin.cons_zero]; apply Fin.ext; simp
    · refine Fin.cases ?_ (fun i' => ?_) i
      · rw [Fin.cons_succ, Fin.cons_zero]; apply Fin.ext; simp
      · rw [Fin.cons_succ, Fin.cons_succ]; apply Fin.ext; simp
  rw [hidx] at h'
  exact h'

/-- Reshape a first core `T₀ : R₀ × out₀ × R₁` to the matrix with rows
    `R₀ · out₀` and columns `R₁`. -/
def coreToMat {R0 out0 R1 : ℕ} (T0 : Fin R0 → Fin out0 → Fin R1 → 𝕜) :
    Matrix (Fin (R0 * out0)) (Fin R1) 𝕜 :=
  Matrix.of fun ri s => T0 (finProdFinEquiv.symm ri).1 (finProdFinEquiv.symm ri).2 s

/-- First-mode matricization of an open tensor: mode `0` (the exposed bond)
    in the rows, the remaining modes in the columns. -/
def openToMat {m : ℕ} {out : Fin m → ℕ} {L : ℕ}
    (W : Tensor 𝕜 (m + 1) (Fin.cons L out)) :
    Matrix (Fin L) (Fin (∏ j : Fin m, out j)) 𝕜 :=
  Matrix.of fun s cj =>
    W (Fin.cons (Fin.cast (by simp) s)
        (fun j => Fin.cast (by simp) ((multiIndexEquiv out).symm cj j)))

end Tensor
