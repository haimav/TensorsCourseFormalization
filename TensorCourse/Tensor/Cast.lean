/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Defs

/-!
# Cast-through Lemmas for Tensors

Evaluating a type-cast tensor at a multi-index reduces to evaluating the
underlying tensor at a reindexed multi-index. These small `subst`-based
lemmas are shared by every file that juggles `Function.update`-induced
dimension-function equalities (`Tensor.NModeProd`, `Tensor.NModeInner`,
`TuckerDecomposition.Defs`), so they live here in one place rather than
being re-declared privately in each.

## Main results

* `Tensor.cast_apply` — evaluation of `h ▸ T`.
* `Tensor.cast_idx_val` — `.val` is preserved when transporting an index
  along `h : dims = dims'`.
* `Tensor.cast_tensor_apply` — evaluation of `cast (congrArg (Tensor 𝕜 D) h) T`.
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ}

/-- Cast-through for `▸` on a tensor: evaluating a type-cast tensor at
    an index equals evaluating the underlying tensor on the cast index. -/
theorem cast_apply {dims dims' : Fin D → ℕ}
    (h : dims = dims') (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims' d)) :
    (h ▸ T) idx = T (fun d => (h ▸ idx : (d : Fin D) → Fin (dims d)) d) := by
  subst h; rfl

/-- Variant of `cast_apply` whose RHS reindexes each component with `Fin.cast`
    (rather than `▸` on the whole index tuple). The two normal forms are used
    by different downstream proofs. -/
theorem cast_apply_dims {dims dims' : Fin D → ℕ}
    (h : dims = dims') (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims' d)) :
    (h ▸ T) idx = T (fun d => (idx d).cast (congr_fun h.symm d)) := by
  subst h; rfl

/-- Cast-through for `▸` on a dependent index: the underlying natural-number
    value is preserved under transport along `h : dims = dims'`. -/
theorem cast_idx_val {dims dims' : Fin D → ℕ} (h : dims = dims')
    (idx : (d : Fin D) → Fin (dims d)) (d : Fin D) :
    ((h ▸ idx : (d : Fin D) → Fin (dims' d)) d).val = (idx d).val := by
  subst h; rfl

/-- Cast-through for `cast (congrArg (Tensor 𝕜 D) h)` on a tensor: evaluating
    the casted tensor at `idx` equals evaluating the underlying tensor at a
    `.val`-preserving reindex of `idx`. -/
theorem cast_tensor_apply {dims dims' : Fin D → ℕ}
    (h : dims = dims') (T : Tensor 𝕜 D dims)
    (idx : (d : Fin D) → Fin (dims' d)) :
    cast (congrArg (Tensor 𝕜 D) h) T idx =
      T (fun d => (idx d).cast (congrFun h d).symm) := by
  subst h
  simp [Fin.cast_eq_self]

end Tensor

end
