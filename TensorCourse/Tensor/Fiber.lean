/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import Mathlib.Data.Matrix.Basic

/-!
# Fibers and Slices

A *fiber* of a tensor is obtained by fixing all but one of the multi-index
coordinates — it is a vector along the free mode. A *slice* fixes all but
two coordinates and yields a matrix.

## Main definitions

* `Tensor.fiber T n ctx` — the mode-`n` fiber of `T` given a context that fixes
  every other mode
* `Tensor.slice T m n hmn ctx` — the `(m, n)`-slice of `T` given a context that
  fixes every other mode
* `Tensor.hslice` / `Tensor.lslice` / `Tensor.fslice` — order-3 sugars that fix
  modes 0 / 1 / 2 respectively

## Main results

* Linearity of `fiber` and `slice` (with respect to `+` and `•`)
* `fiber_ext` — a tensor is determined by its mode-`n` fibers
* `fslice_eq_slice`, `fiber_fslice_column` — compatibility between the order-3
  sugars and the general `slice`/`fiber`
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- The mode-`n` fiber of a tensor: fix every mode other than `n` via `ctx`
    and let mode `n` vary. The result is a vector of length `dims n`. -/
def fiber (n : Fin D)
    (ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) :
    Fin (dims n) → 𝕜 :=
  fun iₙ => T (fun d =>
    if h : d = n then h ▸ iₙ
    else ctx ⟨d, h⟩)

@[simp]
theorem fiber_apply (n : Fin D)
    (ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) (iₙ : Fin (dims n)) :
    fiber n ctx T iₙ =
      T (fun d => if h : d = n then h ▸ iₙ else ctx ⟨d, h⟩) := rfl

/-- Fibers are additive in the tensor argument. -/
theorem fiber_add (n : Fin D)
    (ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val))
    (T₁ T₂ : Tensor 𝕜 D dims) :
    fiber n ctx (T₁ + T₂) = fiber n ctx T₁ + fiber n ctx T₂ := by
  funext iₙ
  simp [fiber]

/-- Fibers commute with scalar multiplication. -/
theorem fiber_smul (n : Fin D) (c : 𝕜)
    (ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) :
    fiber n ctx (c • T) = c • fiber n ctx T := by
  funext iₙ
  simp [fiber]

/-- A tensor is determined by its mode-`n` fibers: if two tensors agree on
    all mode-`n` fibers (for every context that fixes the remaining modes),
    then they are equal. -/
theorem fiber_ext {T₁ T₂ : Tensor 𝕜 D dims} (n : Fin D)
    (h : ∀ (ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val)),
        fiber n ctx T₁ = fiber n ctx T₂) :
    T₁ = T₂ := by
  ext idx
  -- Build the context by restricting `idx` to modes other than `n`.
  let ctx : (d : {m : Fin D // m ≠ n}) → Fin (dims d.val) :=
    fun d => idx d.val
  have hfib := congrFun (h ctx) (idx n)
  simp only [fiber_apply] at hfib
  -- The re-assembled index equals `idx`.
  have hidx : (fun d => if h : d = n then h ▸ idx n else ctx ⟨d, h⟩) = idx := by
    funext d
    by_cases hd : d = n
    · subst hd
      simp
    · simp [hd, ctx]
  rw [hidx] at hfib
  exact hfib

/-- The `(m, n)`-slice of a tensor: fix every mode other than `m` and `n` via
    `ctx` and let modes `m` and `n` vary. The hypothesis `hmn : m ≠ n` is
    not used in the definition itself but is part of the intended API so that
    the context type is nondegenerate. The result is a matrix of size
    `dims m × dims n`. -/
def slice (m n : Fin D) (_hmn : m ≠ n)
    (ctx : (d : {k : Fin D // k ≠ m ∧ k ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) :
    Matrix (Fin (dims m)) (Fin (dims n)) 𝕜 :=
  fun iₘ iₙ => T (fun d =>
    if hm : d = m then hm ▸ iₘ
    else if hn : d = n then hn ▸ iₙ
    else ctx ⟨d, hm, hn⟩)

@[simp]
theorem slice_apply (m n : Fin D) (hmn : m ≠ n)
    (ctx : (d : {k : Fin D // k ≠ m ∧ k ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) (iₘ : Fin (dims m)) (iₙ : Fin (dims n)) :
    slice m n hmn ctx T iₘ iₙ =
      T (fun d =>
        if hm : d = m then hm ▸ iₘ
        else if hn : d = n then hn ▸ iₙ
        else ctx ⟨d, hm, hn⟩) := rfl

/-- Slices are additive in the tensor argument. -/
theorem slice_add (m n : Fin D) (hmn : m ≠ n)
    (ctx : (d : {k : Fin D // k ≠ m ∧ k ≠ n}) → Fin (dims d.val))
    (T₁ T₂ : Tensor 𝕜 D dims) :
    slice m n hmn ctx (T₁ + T₂) =
      slice m n hmn ctx T₁ + slice m n hmn ctx T₂ := by
  funext iₘ iₙ
  simp [slice]

/-- Slices commute with scalar multiplication. -/
theorem slice_smul (m n : Fin D) (hmn : m ≠ n) (c : 𝕜)
    (ctx : (d : {k : Fin D // k ≠ m ∧ k ≠ n}) → Fin (dims d.val))
    (T : Tensor 𝕜 D dims) :
    slice m n hmn ctx (c • T) = c • slice m n hmn ctx T := by
  funext iₘ iₙ
  simp [slice]

/-! ### Order-3 slice sugars

For order-3 tensors the three standard slice types fix a single mode.
Implementing them via a dependent `match` on `d : Fin 3` avoids the
subtype unpacking that would be forced by routing through `slice`. -/

section Order3

variable {dims₃ : Fin 3 → ℕ}

/-- Horizontal slice of an order-3 tensor: fix mode 0 (the horizontal index). -/
def hslice (i : Fin (dims₃ 0)) (T : Tensor 𝕜 3 dims₃) :
    Matrix (Fin (dims₃ 1)) (Fin (dims₃ 2)) 𝕜 :=
  fun j k => T (fun d => match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => k)

/-- Lateral slice of an order-3 tensor: fix mode 1 (the lateral index). -/
def lslice (j : Fin (dims₃ 1)) (T : Tensor 𝕜 3 dims₃) :
    Matrix (Fin (dims₃ 0)) (Fin (dims₃ 2)) 𝕜 :=
  fun i k => T (fun d => match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => k)

/-- Frontal slice of an order-3 tensor: fix mode 2 (the frontal index). -/
def fslice (k : Fin (dims₃ 2)) (T : Tensor 𝕜 3 dims₃) :
    Matrix (Fin (dims₃ 0)) (Fin (dims₃ 1)) 𝕜 :=
  fun i j => T (fun d => match d with
    | ⟨0, _⟩ => i
    | ⟨1, _⟩ => j
    | ⟨2, _⟩ => k)

@[simp]
theorem hslice_apply (i : Fin (dims₃ 0)) (T : Tensor 𝕜 3 dims₃)
    (j : Fin (dims₃ 1)) (k : Fin (dims₃ 2)) :
    hslice i T j k = T (fun d => match d with
      | ⟨0, _⟩ => i
      | ⟨1, _⟩ => j
      | ⟨2, _⟩ => k) := rfl

@[simp]
theorem lslice_apply (j : Fin (dims₃ 1)) (T : Tensor 𝕜 3 dims₃)
    (i : Fin (dims₃ 0)) (k : Fin (dims₃ 2)) :
    lslice j T i k = T (fun d => match d with
      | ⟨0, _⟩ => i
      | ⟨1, _⟩ => j
      | ⟨2, _⟩ => k) := rfl

@[simp]
theorem fslice_apply (k : Fin (dims₃ 2)) (T : Tensor 𝕜 3 dims₃)
    (i : Fin (dims₃ 0)) (j : Fin (dims₃ 1)) :
    fslice k T i j = T (fun d => match d with
      | ⟨0, _⟩ => i
      | ⟨1, _⟩ => j
      | ⟨2, _⟩ => k) := rfl

/-- Context builder for `fslice`: on input `d ≠ 0, 1` (i.e. mode 2), return `k`. -/
private def fsliceCtx (k : Fin (dims₃ 2))
    (d : {m : Fin 3 // m ≠ 0 ∧ m ≠ 1}) : Fin (dims₃ d.val) :=
  match d with
  | ⟨⟨0, _⟩, hne⟩ => absurd rfl hne.1
  | ⟨⟨1, _⟩, hne⟩ => absurd rfl hne.2
  | ⟨⟨2, _⟩, _⟩ => k

/-- Context builder for `hslice`: on input `d ≠ 1, 2` (i.e. mode 0), return `i`. -/
private def hsliceCtx (i : Fin (dims₃ 0))
    (d : {m : Fin 3 // m ≠ 1 ∧ m ≠ 2}) : Fin (dims₃ d.val) :=
  match d with
  | ⟨⟨0, _⟩, _⟩ => i
  | ⟨⟨1, _⟩, hne⟩ => absurd rfl hne.1
  | ⟨⟨2, _⟩, hne⟩ => absurd rfl hne.2

/-- Context builder for `lslice`: on input `d ≠ 0, 2` (i.e. mode 1), return `j`. -/
private def lsliceCtx (j : Fin (dims₃ 1))
    (d : {m : Fin 3 // m ≠ 0 ∧ m ≠ 2}) : Fin (dims₃ d.val) :=
  match d with
  | ⟨⟨0, _⟩, hne⟩ => absurd rfl hne.1
  | ⟨⟨1, _⟩, _⟩ => j
  | ⟨⟨2, _⟩, hne⟩ => absurd rfl hne.2

/-- Context builder for the mode-0 fiber of an order-3 tensor at `(j, k)`. -/
private def fiber0Ctx (j : Fin (dims₃ 1)) (k : Fin (dims₃ 2))
    (d : {m : Fin 3 // m ≠ 0}) : Fin (dims₃ d.val) :=
  match d with
  | ⟨⟨0, _⟩, hne⟩ => absurd rfl hne
  | ⟨⟨1, _⟩, _⟩ => j
  | ⟨⟨2, _⟩, _⟩ => k

/-- Compatibility: the frontal slice `T.fslice k` agrees with the general
    slice that fixes modes 0 and 1 (i.e. varies modes 0 and 1) once mode 2 is
    set to `k`. -/
theorem fslice_eq_slice (T : Tensor 𝕜 3 dims₃) (k : Fin (dims₃ 2)) :
    fslice k T = slice 0 1 (by decide) (fsliceCtx k) T := by
  funext i j
  simp only [fslice_apply, slice_apply]
  congr 1
  funext d
  match d with
  | ⟨0, _⟩ => simp
  | ⟨1, _⟩ => simp
  | ⟨2, _⟩ => simp [fsliceCtx]

/-- Compatibility: the horizontal slice `T.hslice i` agrees with the general
    slice that varies modes 1 and 2 with mode 0 fixed to `i`. -/
theorem hslice_eq_slice (T : Tensor 𝕜 3 dims₃) (i : Fin (dims₃ 0)) :
    hslice i T = slice 1 2 (by decide) (hsliceCtx i) T := by
  funext j k
  simp only [hslice_apply, slice_apply]
  congr 1
  funext d
  match d with
  | ⟨0, _⟩ => simp [hsliceCtx]
  | ⟨1, _⟩ => simp
  | ⟨2, _⟩ => simp

/-- Compatibility: the lateral slice `T.lslice j` agrees with the general
    slice that varies modes 0 and 2 with mode 1 fixed to `j`. -/
theorem lslice_eq_slice (T : Tensor 𝕜 3 dims₃) (j : Fin (dims₃ 1)) :
    lslice j T = slice 0 2 (by decide) (lsliceCtx j) T := by
  funext i k
  simp only [lslice_apply, slice_apply]
  congr 1
  funext d
  match d with
  | ⟨0, _⟩ => simp
  | ⟨1, _⟩ => simp [lsliceCtx]
  | ⟨2, _⟩ => simp

/-- For an order-3 tensor, the mode-0 fiber at a fixed `(j, k)` equals the
    column of the frontal slice at `k` indexed by `j`. -/
theorem fiber_zero_eq_fslice_column (T : Tensor 𝕜 3 dims₃)
    (j : Fin (dims₃ 1)) (k : Fin (dims₃ 2)) :
    fiber 0 (fiber0Ctx j k) T = fun i => fslice k T i j := by
  funext i
  simp only [fiber_apply, fslice_apply]
  congr 1
  funext d
  match d with
  | ⟨0, _⟩ => simp
  | ⟨1, _⟩ => simp [fiber0Ctx]
  | ⟨2, _⟩ => simp [fiber0Ctx]

end Order3

end Tensor
