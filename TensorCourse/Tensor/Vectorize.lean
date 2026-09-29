/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs
import TensorCourse.Tensor.InnerProd
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pi
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Tensor Vectorization

Vectorization flattens a tensor to a vector of length `∏ d, dims d`. The
flattening is mediated by an equivalence `multiIndexEquiv` between the
multi-index set `(d : Fin D) → Fin (dims d)` and `Fin (∏ d, dims d)`.

## Main definitions

* `multiIndexEquiv` — equivalence between multi-indices and `Fin (∏ dims)`
  built as an explicit column-major recursive construction (peel off mode 0).
* `Tensor.vectorize T` — the vector obtained by flattening `T`

## Main results

* `vectorize_injective` — vectorization is injective (in fact a bijection)
* `dotProd_eq_vectorize_dotProd` — the tensor inner product equals the
  Euclidean dot product of the vectorizations (with conjugation on the first)

## Ordering convention (Kolda-Bader, column-major, mode 0 fastest)

The equivalence `multiIndexEquiv` is built recursively by peeling off mode 0
(via `Fin.consEquiv`) and composing with `finProdFinEquiv`. Under this
convention, the multi-index `(i₀, i₁, …, i_{D-1})` lands at flat index

    i₀ + (dims 0) · i₁ + (dims 0)(dims 1) · i₂ + ⋯

i.e. *mode 0 varies fastest, mode D-1 slowest*. This matches the lecture's
recursive `vec` (Lecture 3, lines 100–111) after unrolling, and is the
standard Kolda-Bader / MATLAB-style column-major convention. Every downstream
lemma relating vectorization to mode-n unfolding, Kronecker, Khatri-Rao, etc.
depends on this choice.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- Equivalence between the multi-index set `(d : Fin D) → Fin (dims d)` and
    `Fin (∏ d, dims d)`, built recursively in **column-major order with
    mode 0 varying fastest** (Kolda-Bader convention).

    Construction: for `D = 0`, the multi-index set is a singleton and the
    target is `Fin 1`. For `D + 1`, peel off mode 0 via `Fin.consEquiv.symm`
    to obtain a pair `(i₀, rest)` with `i₀ : Fin (dims 0)` and
    `rest : (i : Fin D) → Fin (dims i.succ)`; recursively flatten `rest`,
    then combine via `finProdFinEquiv` so that `(rest, i₀) ↦ i₀ + (dims 0) · rest`.
    This places mode 0 at the fastest-varying position, as required. -/
def multiIndexEquiv : {D : ℕ} → (dims : Fin D → ℕ) →
    ((d : Fin D) → Fin (dims d)) ≃ Fin (∏ d : Fin D, dims d)
  | 0, dims =>
      have hprod : ∏ d : Fin 0, dims d = 1 := Finset.prod_of_isEmpty _
      { toFun := fun _ => ⟨0, by rw [hprod]; exact Nat.one_pos⟩
        invFun := fun _ => fun d : Fin 0 => Fin.elim0 d
        left_inv := fun f => funext (fun d : Fin 0 => Fin.elim0 d)
        right_inv := fun i => by
          apply Fin.ext
          have hlt : i.val < 1 := hprod ▸ i.isLt
          change (0 : ℕ) = i.val
          omega }
  | D + 1, dims =>
      -- Peel off mode 0: `(∀ d, Fin (dims d)) ≃ Fin (dims 0) × (∀ i, Fin (dims i.succ))`.
      have heq : (∏ i : Fin D, dims i.succ) * dims 0 = ∏ d : Fin (D + 1), dims d := by
        rw [Fin.prod_univ_succ]; ring
      let peelOff :
          ((d : Fin (D + 1)) → Fin (dims d)) ≃
            Fin (dims 0) × ((i : Fin D) → Fin (dims i.succ)) :=
        (Fin.consEquiv (fun d => Fin (dims d))).symm
      let flattenRest :
          ((i : Fin D) → Fin (dims i.succ)) ≃
            Fin (∏ i : Fin D, dims i.succ) :=
        multiIndexEquiv (fun i => dims i.succ)
      -- Combine: `Fin (dims 0) × Fin (prod_rest) ≃ Fin (prod_rest * dims 0)`
      -- via `swap` + `finProdFinEquiv`, where `(rest, i₀) ↦ i₀ + (dims 0) * rest`.
      let swap : Fin (dims 0) × Fin (∏ i : Fin D, dims i.succ) ≃
          Fin (∏ i : Fin D, dims i.succ) × Fin (dims 0) := Equiv.prodComm _ _
      let combine : Fin (∏ i : Fin D, dims i.succ) × Fin (dims 0) ≃
          Fin ((∏ i : Fin D, dims i.succ) * dims 0) := finProdFinEquiv
      let prodCast : Fin ((∏ i : Fin D, dims i.succ) * dims 0) ≃
          Fin (∏ d : Fin (D + 1), dims d) := (Fin.castOrderIso heq).toEquiv
      peelOff.trans ((Equiv.prodCongr (Equiv.refl _) flattenRest).trans
        (swap.trans (combine.trans prodCast)))

/-- Vectorization of a tensor: reshape into a vector of length `∏ d, dims d`.
    Composes `T` with the inverse of `multiIndexEquiv`. -/
def vectorize (T : Tensor 𝕜 D dims) :
    Fin (∏ d : Fin D, dims d) → 𝕜 :=
  fun i => T ((multiIndexEquiv dims).symm i)

@[simp]
theorem vectorize_apply (T : Tensor 𝕜 D dims)
    (i : Fin (∏ d : Fin D, dims d)) :
    vectorize T i = T ((multiIndexEquiv dims).symm i) := rfl

/-- Vectorization sends the multi-index `idx` to its image under
    `multiIndexEquiv`. -/
theorem vectorize_multiIndexEquiv (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims d)) :
    vectorize T (multiIndexEquiv dims idx) = T idx := by
  simp [vectorize]

/-- Vectorization is injective. -/
theorem vectorize_injective :
    Function.Injective (vectorize (𝕜 := 𝕜) (dims := dims)) := by
  intro T₁ T₂ h
  ext idx
  have := congrFun h (multiIndexEquiv dims idx)
  simpa [vectorize] using this

/-- Tensor inner product equals the Euclidean dot product of the
    vectorizations (with conjugation on the first argument).

    Note: the conjugation is on the **first** argument (the standard math
    inner-product convention, conjugate-linear in arg 1). Over `ℝ`
    conjugation is the identity, so this coincides with the lecture's
    unconjugated `vec(X)ᵀ vec(Y)`. -/
theorem dotProd_eq_vectorize_dotProd (T₁ T₂ : Tensor 𝕜 D dims) :
    ⟪T₁, T₂⟫ₜ =
      ∑ i : Fin (∏ d : Fin D, dims d),
        starRingEnd 𝕜 (vectorize T₁ i) * vectorize T₂ i := by
  rw [dotProd_def]
  -- Reindex the sum via the equivalence `multiIndexEquiv`.
  rw [← Equiv.sum_comp (multiIndexEquiv dims).symm
      (fun idx => starRingEnd 𝕜 (T₁ idx) * T₂ idx)]
  rfl

end Tensor
