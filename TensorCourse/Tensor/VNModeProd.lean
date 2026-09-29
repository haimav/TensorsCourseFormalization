/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.NModeProd
import TensorCourse.Tensor.Cast
import TensorCourse.Matrix.KroneckerLemmas

/-!
# Vector n-Mode Product of a Tensor

The vector n-mode product contracts an order-`(D+1)` tensor with a vector
along its `n`-th mode, producing an order-`D` tensor. Conceptually, it is
`T ×ₙ vᵀ` (where `vᵀ` is `v` viewed as a `1 × (dims n)` matrix) followed
by squeezing the resulting singleton mode. We define it directly as the
contracting sum to keep the output type clean:

  `(T.vNModeProd n v) idx = ∑ iₙ : Fin (dims n), v iₙ * T (insertNth n iₙ idx)`.

## Main definitions

* `Tensor.vNModeProd` — the vector n-mode product; per-mode infixes
  `×̄₁`, `×̄₂`, `×̄₃` (for modes 0, 1, 2 respectively).

## Main results

* `Tensor.vNModeProd_apply` — pointwise formula (the definition itself).
* `Tensor.vNModeProd_add_T`, `Tensor.vNModeProd_smul_T` — linearity in the
  tensor argument.
* `Tensor.vNModeProd_add_v`, `Tensor.vNModeProd_smul_v` — linearity in the
  vector argument.
* `Tensor.outerProd3_vNModeProd_two` — rank-1 reduction:
  `(u ◯ₜ v ◯ₜ w) ×̄₃ x = (w ⬝ᵥ x) • (u ◯ₜ v)`.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### Definition of `vNModeProd` -/

variable {D : ℕ} {dims : Fin (D + 1) → ℕ}

/-- Vector n-mode product: contract `T` along mode `n` with the vector `v`.

    Pointwise, `(T.vNModeProd n v) idx = ∑ iₙ, v iₙ * T (insertNth n iₙ idx)`,
    where `idx : (i : Fin D) → Fin (dims (n.succAbove i))` is a multi-index
    of the output shape and `Fin.insertNth n iₙ idx` reconstructs the
    `(D+1)`-mode multi-index by placing `iₙ` at position `n`.

    Conceptually equivalent to `T ×ₙ vᵀ` (viewing `v` as a `1 × (dims n)`
    row matrix) followed by squeezing the resulting singleton mode. -/
def vNModeProd (n : Fin (D + 1)) (v : Fin (dims n) → 𝕜)
    (T : Tensor 𝕜 (D + 1) dims) :
    Tensor 𝕜 D (fun i => dims (n.succAbove i)) :=
  fun idx => ∑ iₙ : Fin (dims n), v iₙ * T (Fin.insertNth n iₙ idx)

/-! ### Per-mode notation -/

scoped infixl:65 " ×̄₁ " => fun T v => Tensor.vNModeProd 0 v T
scoped infixl:65 " ×̄₂ " => fun T v => Tensor.vNModeProd 1 v T
scoped infixl:65 " ×̄₃ " => fun T v => Tensor.vNModeProd 2 v T

/-! ### Pointwise formula -/

/-- Element-wise formula for the vector n-mode product. This is the
    defining equation; rewrite with it explicitly when the contracting
    sum is wanted (it is deliberately not `@[simp]`, to avoid eagerly
    expanding `vNModeProd` into a large sum). -/
theorem vNModeProd_apply (n : Fin (D + 1)) (v : Fin (dims n) → 𝕜)
    (T : Tensor 𝕜 (D + 1) dims)
    (idx : (i : Fin D) → Fin (dims (n.succAbove i))) :
    (T.vNModeProd n v) idx =
      ∑ iₙ : Fin (dims n), v iₙ * T (Fin.insertNth n iₙ idx) := rfl

/-! ### Linearity in `T` -/

/-- Vector n-mode product is additive in the tensor argument. -/
theorem vNModeProd_add_T (n : Fin (D + 1)) (v : Fin (dims n) → 𝕜)
    (T₁ T₂ : Tensor 𝕜 (D + 1) dims) :
    (T₁ + T₂).vNModeProd n v = T₁.vNModeProd n v + T₂.vNModeProd n v := by
  funext idx
  change ∑ iₙ : Fin (dims n), v iₙ * (T₁ + T₂) (Fin.insertNth n iₙ idx) =
       (∑ iₙ : Fin (dims n), v iₙ * T₁ (Fin.insertNth n iₙ idx)) +
       (∑ iₙ : Fin (dims n), v iₙ * T₂ (Fin.insertNth n iₙ idx))
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  change v iₙ * (T₁ (Fin.insertNth n iₙ idx) + T₂ (Fin.insertNth n iₙ idx)) =
       v iₙ * T₁ (Fin.insertNth n iₙ idx) + v iₙ * T₂ (Fin.insertNth n iₙ idx)
  ring

/-- Scalar multiplication pulls out of the tensor argument. -/
theorem vNModeProd_smul_T (n : Fin (D + 1)) (v : Fin (dims n) → 𝕜)
    (c : 𝕜) (T : Tensor 𝕜 (D + 1) dims) :
    (c • T).vNModeProd n v = c • T.vNModeProd n v := by
  funext idx
  change ∑ iₙ : Fin (dims n), v iₙ * (c * T (Fin.insertNth n iₙ idx)) =
       c * (∑ iₙ : Fin (dims n), v iₙ * T (Fin.insertNth n iₙ idx))
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  ring

/-! ### Linearity in `v` -/

/-- Vector n-mode product is additive in the vector argument. -/
theorem vNModeProd_add_v (n : Fin (D + 1)) (v₁ v₂ : Fin (dims n) → 𝕜)
    (T : Tensor 𝕜 (D + 1) dims) :
    T.vNModeProd n (v₁ + v₂) = T.vNModeProd n v₁ + T.vNModeProd n v₂ := by
  funext idx
  change ∑ iₙ : Fin (dims n), (v₁ iₙ + v₂ iₙ) * T (Fin.insertNth n iₙ idx) =
       (∑ iₙ : Fin (dims n), v₁ iₙ * T (Fin.insertNth n iₙ idx)) +
       (∑ iₙ : Fin (dims n), v₂ iₙ * T (Fin.insertNth n iₙ idx))
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  ring

/-- Scalar multiplication pulls out of the vector argument. -/
theorem vNModeProd_smul_v (n : Fin (D + 1)) (c : 𝕜) (v : Fin (dims n) → 𝕜)
    (T : Tensor 𝕜 (D + 1) dims) :
    T.vNModeProd n (c • v) = c • T.vNModeProd n v := by
  funext idx
  change ∑ iₙ : Fin (dims n), (c * v iₙ) * T (Fin.insertNth n iₙ idx) =
       c * (∑ iₙ : Fin (dims n), v iₙ * T (Fin.insertNth n iₙ idx))
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  ring

/-! ### Rank-1 reduction lemma (Eq. eq:vmode-prod-rank-1 in Lecture 4) -/

/-- The output shape of `((u ◯ₜ v) ◯ₜ w) ×̄₃ x` agrees with the shape of
    `u ◯ₜ v`. Both are functions `Fin 2 → ℕ`. The propositional equality
    is needed to bridge the two output shapes; it does not hold by `rfl`
    because `Fin.succAbove (2 : Fin 3) i` does not reduce definitionally
    to `Fin.castAdd 1 i`. -/
theorem outer3_vmode_dim_eq {I J K : ℕ} :
    (fun i : Fin 2 =>
      combinedDims (combinedDims (fun _ : Fin 1 => I) (fun _ : Fin 1 => J))
        (fun _ : Fin 1 => K) ((2 : Fin 3).succAbove i)) =
    combinedDims (fun _ : Fin 1 => I) (fun _ : Fin 1 => J) := by
  funext i
  have h2 : ((2 : Fin 3).succAbove i : Fin 3) = i.castSucc := by
    change (Fin.last 2).succAbove i = i.castSucc
    rw [Fin.succAbove_last]
  rw [show ((2 : Fin 3).succAbove i : Fin 3) = Fin.castAdd 1 i from by
        apply Fin.ext; rw [h2]; rfl]
  exact combinedDims_castAdd i

/-- **Vector n-mode product on a rank-1 order-3 outer product (mode 2).**
    For vectors `u : Fin I`, `v : Fin J`, `w, x : Fin K`, contracting the
    rank-1 outer product `u ◯ₜ v ◯ₜ w` along mode 2 with `x` produces
    `(wᵀ x) · (u ◯ₜ v)`. This is the lecture's rank-1 vector-n-mode lemma.

    The output type of the LHS is `Tensor 𝕜 2` indexed by
    `fun i => combinedDims (combinedDims (·=I) (·=J)) (·=K) ((2:Fin 3).succAbove i)`,
    while the RHS-pre-cast is indexed by `combinedDims (·=I) (·=J)`.
    Bridge with `outer3_vmode_dim_eq.symm ▸`. -/
theorem outerProd3_vNModeProd_two
    {I J K : ℕ}
    (u : Fin I → 𝕜) (v : Fin J → 𝕜) (w : Fin K → 𝕜) (x : Fin K → 𝕜) :
    (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v ◯ₜ Tensor.toTensor1 w) ×̄₃ x
      = outer3_vmode_dim_eq.symm ▸
          ((w ⬝ᵥ x) • (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)) := by
  -- Unfold `w ⬝ᵥ x` to the explicit sum so the rest of the proof — which
  -- was originally written against `(∑ k, w k * x k)` — goes through unchanged.
  simp only [dotProduct]
  -- Define a local type-family abbreviation (as `Type`-valued `let` so it
  -- unfolds reliably), and a corresponding `Fin.insertNth` wrapper. Using
  -- the abbreviation lets us pass the family explicitly to `Fin.insertNth`,
  -- sidestepping Lean's higher-order unification failure when the family
  -- is left implicit. Both abbreviations unfold definitionally to their
  -- underlying expressions, so `splitLeft`/`splitRight` continue to apply.
  set α : Fin 3 → Type :=
    fun k => Fin (combinedDims (combinedDims (fun _ : Fin 1 => I)
      (fun _ : Fin 1 => J)) (fun _ : Fin 1 => K) k) with hα
  funext (idx : (i : Fin 2) → α ((2 : Fin 3).succAbove i))
  -- Unfold the `×̄₃` lambda layer.
  change (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v ◯ₜ Tensor.toTensor1 w).vNModeProd 2 x idx
        = (outer3_vmode_dim_eq.symm ▸
            ((∑ k, w k * x k) • (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v))) idx
  -- Resolve the `▸` cast on the RHS.
  rw [cast_apply_dims outer3_vmode_dim_eq.symm
        ((∑ k, w k * x k) • (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)) idx]
  -- Apply the simp formula on LHS. After this, LHS is the sum
  -- `∑ iₙ : Fin K, x iₙ * ((u◯v) ◯ w) (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)`.
  rw [vNModeProd_apply]
  -- Massage RHS.
  simp only [Tensor.smul_apply]
  rw [show (∑ k, w k * x k) = ∑ k, x k * w k from by
        refine Finset.sum_congr rfl (fun k _ => ?_); ring]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl (fun iₙ _ => ?_)
  -- Compute the LHS summand at `iₙ`. Use the local `ins` to keep the
  -- multi-index well-formed under Lean's unification.
  -- Show the summand of LHS is (u ◯ v) (...) * w iₙ.
  have hRight : (Tensor.toTensor1 w)
      (splitRight (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)) = w iₙ := by
    rw [Tensor.toTensor1_apply]
    -- Show value-level equality.
    change w (((Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)
                (Fin.natAdd 2 0)).cast _) = w iₙ
    have h_natAdd : (Fin.natAdd 2 (0 : Fin 1) : Fin 3) = 2 := rfl
    have h_apply : (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) 2 = iₙ :=
      Fin.insertNth_apply_same _ _ _
    -- Reduce `(... .cast ...)` via `Fin.ext`: the cast preserves `.val`.
    have : ((Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) (Fin.natAdd 2 0)).cast
        (combinedDims_natAdd 0) = iₙ := by
      apply Fin.ext
      change ((Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) (Fin.natAdd 2 0)).val = iₙ.val
      rw [show ((Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) (Fin.natAdd 2 0)).val =
          ((Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) 2).val from by rw [h_natAdd]]
      rw [h_apply]
    rw [this]
  have hLeft : (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
      (splitLeft (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)) =
      (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
        (fun d : Fin 2 => (idx d).cast (congr_fun outer3_vmode_dim_eq d)) := by
    change (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
        (splitLeft (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)) = _
    congr 1
    funext d
    apply Fin.ext
    -- `splitLeft (Fin.insertNth (2:Fin 3) iₙ idx) d` evaluates to
    -- `(Fin.insertNth (2:Fin 3) iₙ idx (Fin.castAdd 1 d)).cast _`.
    -- The RHS is `(idx d).cast _`. After `Fin.ext`, both sides are `.val`s.
    -- We use the `Fin.insertNth` simp formula which says, for `j ≠ i`,
    -- the application returns the corresponding component of the tuple.
    -- Specifically, `insertNth_apply_succAbove` says
    -- `i.insertNth x p (i.succAbove k) = p k`.
    -- We need to view `Fin.castAdd 1 d` as `(2 : Fin 3).succAbove d`.
    -- These are equal at `.val` level but possibly not as `Fin 3` terms.
    -- Approach: pose `hcs : Fin.castAdd 1 d = (Fin.last 2).succAbove d`,
    -- pose `h2 : (2 : Fin 3) = Fin.last 2`, then transport.
    have hcs : (Fin.castAdd 1 d : Fin 3) = (Fin.last 2).succAbove d := by
      rw [Fin.succAbove_last]
      rfl
    have h2 : (2 : Fin 3) = Fin.last 2 := rfl
    -- Step 1: Replace `(2 : Fin 3)` with `Fin.last 2` everywhere via `h2`.
    -- This is just unfolding the numeric literal — purely definitional.
    -- Step 2: Replace `Fin.castAdd 1 d` with `(Fin.last 2).succAbove d` at
    -- the value level.
    -- Step 3: Apply `insertNth_apply_succAbove` and conclude.
    -- We do this by explicitly computing both sides at the value level.
    have step :
        (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx (Fin.castAdd 1 d)).val =
        (idx d).val := by
      -- Replace the literal `2` by `Fin.last 2`; this is `rfl` so no-op.
      change (Fin.insertNth (α := α) (Fin.last 2) iₙ idx (Fin.castAdd 1 d)).val =
          (idx d).val
      -- Replace `Fin.castAdd 1 d` by `(Fin.last 2).succAbove d` at the value
      -- level. We can't `rw [hcs]` directly due to dependent typing, so we
      -- introduce a term for the post-rewrite value and prove `.val`-equality.
      have heq_val : ∀ (j j' : Fin 3) (h : j = j'),
          (Fin.insertNth (α := α) (Fin.last 2) iₙ idx j).val =
          (Fin.insertNth (α := α) (Fin.last 2) iₙ idx j').val := by
        intros j j' h
        cases h
        rfl
      rw [heq_val _ _ hcs]
      rw [Fin.insertNth_apply_succAbove]
    -- Unfold `splitLeft` and `Fin.cast` `.val`s on both sides.
    change (splitLeft (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) d).val =
        ((idx d).cast _).val
    rw [show (splitLeft (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) d).val =
        (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx (Fin.castAdd 1 d)).val from rfl]
    rw [show ((idx d).cast (congr_fun outer3_vmode_dim_eq d)).val = (idx d).val from rfl]
    exact step
  -- Now the LHS summand is x iₙ * ((u◯ᵥ) (split-left) * w iₙ).
  change x iₙ * ((Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v ◯ₜ Tensor.toTensor1 w)
        (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)) = _
  rw [show (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v ◯ₜ Tensor.toTensor1 w)
        (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx) =
        (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
            (splitLeft (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx))
          * (Tensor.toTensor1 w)
            (splitRight (Fin.insertNth (α := α) (2 : Fin 3) iₙ idx)) from rfl]
  rw [hLeft, hRight]
  -- Match the cast-index expressions: both casts have the same value via
  -- `Fin.cast` value-preservation. The expressions are definitionally equal.
  have huv : (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
      (fun d : Fin 2 => (idx d).cast (congr_fun outer3_vmode_dim_eq d)) =
      (Tensor.toTensor1 u ◯ₜ Tensor.toTensor1 v)
        (fun d : Fin 2 => (idx d).cast (congr_fun outer3_vmode_dim_eq.symm.symm d)) := by
    rfl
  rw [huv]
  ring

end Tensor
