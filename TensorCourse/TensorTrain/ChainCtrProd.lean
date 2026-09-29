/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorTrain.Chain
import TensorCourse.Tensor.ContractiveProdAssoc

/-!
# The TT chain as contractive products: `ttOpen` ⋈-peel (Eq2 ⟺ Eq6, general `D`)

`ttOpen_peel` (in `TensorTrain/Chain.lean`) peels the first core of `ttOpen`
at the *index* level. Here we upgrade it to the **contractive-product** level:

  `ttOpen R T = (first core as a tensor) ⋈ₜ ttOpen (tail)`

(up to the canonical reindex, since the left-hand side has order `m+2` while
the right-hand side, an `⋈ₜ` of an order-3 tensor with an order-`(m+1)` tensor,
has order `2+m`). Iterating this is the lecture's tensor-train chain
`X = T⁽¹⁾ ⋈ₜ ⋯ ⋈ₜ T⁽ᴰ⁾` (Eq6), establishing the `TT-element-index ⟺
TT-decomposition` equivalence for general order.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The dimension equality powering the chain-peel reindex: the `cons`-form
    output dims `cons R₀ out` (order `m+2`) match the `⋈ₜ`-result dims
    `combinedDims ![R₀, out₀] (out∘succ)` (order `2+m`), under the
    `add_comm 2 m` reindex of the position. -/
theorem chainPeel_hd {m : ℕ} (R0 : ℕ) (out : Fin (m + 1) → ℕ)
    (i : Fin (m + 2)) :
    (Fin.cons R0 out : Fin (m + 2) → ℕ) i
      = combinedDims ![R0, out 0] (fun k : Fin m => out k.succ)
          (Fin.cast (Nat.add_comm 2 m).symm i) := by
  refine Fin.cases ?_ (fun i' => ?_) i
  · rw [show Fin.cast (Nat.add_comm 2 m).symm (0 : Fin (m + 2))
          = Fin.castAdd m (0 : Fin 2) from by apply Fin.ext; simp]
    rw [combinedDims_castAdd]; simp
  · refine Fin.cases ?_ (fun i'' => ?_) i'
    · rw [show Fin.cast (Nat.add_comm 2 m).symm (Fin.succ (0 : Fin (m + 1)))
            = Fin.castAdd m (1 : Fin 2) from by apply Fin.ext; simp]
      rw [combinedDims_castAdd]; simp
    · rw [show Fin.cast (Nat.add_comm 2 m).symm (Fin.succ (Fin.succ i''))
            = Fin.natAdd 2 i'' from by apply Fin.ext; simp; omega]
      rw [combinedDims_natAdd]; simp

/-- **Chain peel as a contractive product (Eq2 ⟺ Eq6, general `D`).**
    `ttOpen R T` equals the contractive product of its first core (as an
    order-3 tensor `chainCore (T 0)`) with the `ttOpen` of the remaining
    cores, modulo the canonical `add_comm 2 m` reindex of the free modes.
    Iterating this gives the lecture's chain `X = T⁽¹⁾ ⋈ₜ ⋯ ⋈ₜ T⁽ᴰ⁾`. -/
theorem ttOpen_eq_ctrProd_peel {m : ℕ} {out : Fin (m + 1) → ℕ} (R : Fin (m + 2) → ℕ)
    (T : (k : Fin (m + 1)) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜) :
    ttOpen R T
      = reindex (castEquiv (Nat.add_comm 2 m) (fun i => chainPeel_hd (R 0) out i))
          (chainCore (T 0) ⋈ₜ ttOpen (fun j => R j.succ) (fun k => T k.succ)) := by
  funext idx
  rw [ttOpen_peel, reindex_apply, ctrProd_apply]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [chainCore_apply]
  refine congr_arg₂ (· * ·) ?_ ?_
  -- the first-core factor `T 0 (idx 0) (idx 1) s = T 0 …`, via an explicit
  -- congruence on its three arguments
  · refine congr (congr (congrArg (T 0) ?_) ?_) ?_
    · -- left bond argument: position 0
      apply Fin.ext
      simp only [Fin.lastCases_castSucc, castEquiv_symm_apply, splitLeft, Fin.val_cast]
      rw [show Fin.cast (Nat.add_comm 2 m) (Fin.castAdd m (0 : Fin 2)) = (0 : Fin (m + 2))
            from by apply Fin.ext; simp]
    · -- output argument: position 1
      apply Fin.ext
      simp only [Fin.lastCases_castSucc, castEquiv_symm_apply, splitLeft, Fin.val_cast]
      rw [show Fin.cast (Nat.add_comm 2 m) (Fin.castAdd m (1 : Fin 2))
            = (1 : Fin (m + 2)) from by apply Fin.ext; simp]
    · -- right bond argument: the summation variable `s`
      apply Fin.ext; simp only [Fin.lastCases_last, Fin.val_cast]
  · -- the tail factor `ttOpen (tail)`: reduce to the argument multi-index equality
    refine congrArg _ ?_
    funext d
    refine Fin.cases ?_ (fun j => ?_) d
    · -- leading bond of the tail: the summation variable `s`
      apply Fin.ext; simp [Fin.cons_zero, Fin.val_cast]
    · -- a tail output mode
      apply Fin.ext
      simp only [Fin.cons_succ, castEquiv_symm_apply, splitRight, Fin.val_cast]
      rw [show Fin.cast (Nat.add_comm 2 m) (Fin.natAdd 2 j) = Fin.succ (Fin.succ j)
            from by apply Fin.ext; simp]

end Tensor
