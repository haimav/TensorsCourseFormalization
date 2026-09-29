/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.ContractiveProd
import TensorCourse.Tensor.ModeSplit

/-!
# Tensor Contraction (Tensor Trace) and `⋈ₜ = Tr(◯ₜ)`

Lecture 6 defines the **tensor contraction** (a.k.a. tensor trace) `Tr_{k,l}`,
which contracts two equal-sized modes `k`, `l` of a tensor by summing the
shared index, and then *defines* the contractive product as

  `A ⋈ₜ B = Tr_{D_A, D_A+1}(A ◯ₜ B)`

— contract the **adjacent** boundary modes of the outer product (the last
mode of `A` and the first mode of `B`, which are adjacent after `◯ₜ`).

We formalize the contraction of those two adjacent boundary modes,
`Tensor.traceAdj`, directly on a tensor of `combinedDims (Fin.snoc da S)
(Fin.cons S db)` shape (order `(a+1)+(b+1)`, the shape of an outer product
of a `snoc`-tensor and a `cons`-tensor): it sums the shared `S`-mode that
sits at the boundary, returning an order-`(a+b)` tensor of shape
`combinedDims da db`. This is exactly the lecture's `Tr` at the adjacent
shared modes — the contraction used to define `⋈ₜ`.

## Main results

* `Tensor.traceAdj` — contract the adjacent boundary modes.
* `Tensor.ctrProd_eq_traceAdj_outerProd` — `A ⋈ₜ B = Tr(A ◯ₜ B)`, the
  lecture's definition of the contractive product as a trace of an outer
  product.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ} {S : ℕ}

/-- **Adjacent boundary contraction (the lecture's `Tr` at the shared modes).**
    Contract the two adjacent boundary modes — the last mode of the `snoc`
    block and the first mode of the `cons` block, both of size `S` — of a
    tensor of shape `combinedDims (Fin.snoc da S) (Fin.cons S db)`, by summing
    the shared `S`-index. The result has shape `combinedDims da db`.

    The reconstructed full index places the left free modes (`splitLeft idx`)
    in the `snoc` block, the right free modes (`splitRight idx`) in the `cons`
    block, and the summation variable `s` at *both* boundary positions. -/
def traceAdj
    (C : Tensor 𝕜 ((a + 1) + (b + 1)) (combinedDims (Fin.snoc da S) (Fin.cons S db))) :
    Tensor 𝕜 (a + b) (combinedDims da db) :=
  fun idx =>
    ∑ s : Fin S,
      C (combineIndex
          (Fin.lastCases (motive := fun d => Fin ((Fin.snoc da S : Fin (a + 1) → ℕ) d))
            (Fin.cast (@Fin.snoc_last a (fun _ => ℕ) S da).symm s)
            (fun i => Fin.cast (@Fin.snoc_castSucc a (fun _ => ℕ) S da i).symm
              (splitLeft idx i)))
          (Fin.cons (α := fun d => Fin ((Fin.cons S db : Fin (b + 1) → ℕ) d))
            (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) S db).symm s)
            (fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) S db i).symm
              (splitRight idx i))))

/-- **`⋈ₜ = Tr(◯ₜ)` (Lecture 6 definition of the contractive product).** The
    contractive product is the adjacent boundary contraction of the outer
    product: `A ⋈ₜ B = Tr(A ◯ₜ B)`. Immediate from the round-trips
    `splitLeft (combineIndex f g) = f` and `splitRight (combineIndex f g) = g`,
    which turn the outer-product entry at the reconstructed index back into
    the `ctrProd` summand. -/
theorem ctrProd_eq_traceAdj_outerProd
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S))
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    A ⋈ₜ B = traceAdj (A ◯ₜ B) := by
  funext idx
  rw [ctrProd_apply, traceAdj]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [outerProd_apply, splitLeft_combineIndex, splitRight_combineIndex]

end Tensor
