/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.ConjTranspose

/-!
# Eckart–Young–Mirsky for the Frobenius norm (axiom B2)

This file isolates the **one genuine axiom** of the Lecture 5
formalization: Eckart–Young–Mirsky best-rank-`k` optimality for the
Frobenius norm. Mathlib has a `singularValues` development
(`Analysis/InnerProductSpace/SingularValues.lean`) but does **not** expose
the Frobenius best-rank-`k` optimality theorem in a usable matrix form;
deriving it from the singular-value machinery is substantial work. We
axiomatize the textbook fact behind a stable signature, promotable to a
theorem once Mathlib's SVD API matures.

Reference: Eckart & Young (1936), *The approximation of one matrix by
another of lower rank*; Bernstein, *Matrix Mathematics* (Fact 9.14.x).

## What is axiomatized

* `Matrix.frobNormSq M = ∑ i, ∑ j, ‖M i j‖²` — the squared matrix
  Frobenius norm (a plain definition, *not* an axiom). It matches the
  tensor-level `Tensor.frobNormSq_modeUnfold` bridge, so the matrix axiom
  hooks directly into the tensor error analysis.
* `Matrix.exists_best_rank_approx` (**axiom B2**) — for any matrix `M` and
  any `k ≤ I`, there is a matrix `U` with orthonormal columns and at most
  `k` columns (`U : Matrix (Fin I) (Fin k) 𝕜`, `Uᴴ U = 1`) such that the
  rank-`≤ k` projection `U Uᴴ M` is a **best** rank-`≤ k` approximation of
  `M` in Frobenius norm: every `B` with `rank B ≤ k` satisfies
  `‖M − U Uᴴ M‖_F² ≤ ‖M − B‖_F²`. The hypothesis `k ≤ I` is required: it is
  exactly the regime in which `k` orthonormal columns exist in `𝕜^I`, so
  the `Uᴴ U = 1` clause is satisfiable.

## Why this statement is faithful, consistent, and minimal

* **Faithful.** This *is* Eckart–Young–Mirsky for the Frobenius norm: the
  best rank-`k` approximation of `M` is the orthogonal projection of `M`'s
  columns onto the span of its top-`k` left singular vectors, which has the
  form `U Uᴴ M` with `U` (the truncated left-singular matrix) carrying
  orthonormal columns. We package exactly that projector. Phrasing the
  bound on the *squared* Frobenius norm (rather than the norm) is an
  equivalent restatement (both sides nonneg) that is more convenient
  downstream.
* **Non-degenerate, satisfiable hypotheses.** The hypotheses are "`M` is a
  matrix and `k ≤ I`". The guard `k ≤ I` is not cosmetic: the `Uᴴ U = 1`
  clause requires `k` orthonormal columns in `𝕜^I`, which exist **iff**
  `k ≤ I`. For `k > I` the existential `∃ U, Uᴴ U = 1 ∧ …` is *unsatisfiable*
  (rank `(Uᴴ U) ≤ min I k = I < k = rank 1`), so an unguarded B2 would prove
  `False`; the `k ≤ I` hypothesis removes exactly that regime. Within
  `k ≤ I` there is no regime in which the hypotheses hold while the
  conclusion fails. When `rank M ≤ k (≤ I)` the truncated SVD reconstructs
  `M` exactly (`U Uᴴ M = M`), and the conclusion reads `0 ≤ ‖M − B‖_F²`,
  which is *true*, not vacuous. When `k = 0`, `U` has no columns,
  `U Uᴴ M = 0`, and the conclusion `‖M‖_F² ≤ ‖M − B‖_F²` for every rank-`0`
  (i.e. zero) `B` is again *true*. In every regime within the guard the
  conclusion is a genuine, satisfiable inequality.
* **Consistent.** Under `k ≤ I` the true singular value decomposition is a
  model: take `U` to be the first `k` left singular vectors of `M` (padded
  with orthonormal columns from the orthogonal complement if `k > rank M`,
  which is possible precisely because `k ≤ I = dim 𝕜^I`). Then `Uᴴ U = 1`,
  `U Uᴴ M` is the truncated SVD, and the Frobenius optimality is the
  classical Eckart–Young theorem. So no contradiction can be derived from
  B2.
* **Minimal.** The conclusion is precisely the lower bound consumed by
  `hosvd_quasi_optimal`: the truncated-projection residual `‖M − U Uᴴ M‖_F`
  lower-bounds every rank-`≤ k` approximation error `‖M − B‖_F`. No extra
  structure (singular values, ordering, uniqueness) is asserted.
-/

noncomputable section

open scoped Matrix
open Finset BigOperators

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The **squared Frobenius norm** of a matrix: `∑ i, ∑ j, ‖M i j‖²`. A
    plain definition (not an axiom). Matches the entrywise sum produced by
    the tensor-level bridge `Tensor.frobNormSq_modeUnfold`. -/
def frobNormSq {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) : ℝ :=
  ∑ i, ∑ j, ‖M i j‖ ^ 2

/-- `frobNormSq` unfolds to the double entrywise sum (definitional). -/
theorem frobNormSq_def {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) :
    M.frobNormSq = ∑ i, ∑ j, ‖M i j‖ ^ 2 := rfl

/-- The squared Frobenius norm is nonnegative. -/
theorem frobNormSq_nonneg {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) :
    0 ≤ M.frobNormSq :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- **Axiom B2 — Eckart–Young–Mirsky (Frobenius norm).** For every matrix
    `M : Matrix (Fin I) (Fin C) 𝕜` and every `k ≤ I`, there is a matrix `U`
    with orthonormal columns (`Uᴴ U = 1`) and `k` columns such that the
    rank-`≤ k` projection `U Uᴴ M` is a best rank-`≤ k` Frobenius
    approximation of `M`: every `B` with `rank B ≤ k` has

      ‖M − U Uᴴ M‖_F²  ≤  ‖M − B‖_F².

    This is the truncated singular value decomposition: `U` collects the
    top-`k` left singular vectors of `M` (so `Uᴴ U = 1` and `U Uᴴ M` is the
    truncated SVD), and the inequality is the classical Eckart–Young
    optimality. The true SVD is a model, so B2 is consistent.

    **The `k ≤ I` guard is essential for soundness.** The clause `Uᴴ U = 1`
    asserts that `U`'s `k` columns are orthonormal vectors in `𝕜^I`, so
    `rank (Uᴴ U) ≤ rank U ≤ min I k`. The identity `1 : Matrix (Fin k) …`
    has rank `k`, so `Uᴴ U = 1` forces `k ≤ I`. Without the guard, taking
    `k = I + 1` makes the existential `∃ U, Uᴴ U = 1 ∧ …` unsatisfiable, and
    the axiom would prove `False`. Under `k ≤ I`, `k` orthonormal columns do
    exist in `𝕜^I` and the truncated-SVD projector `U Uᴴ` is a genuine best
    rank-`≤ k` Frobenius approximation, so B2 is true and consistent (the
    true SVD is a model). This is the same class of bug as the prior `A8`
    incident: a statement true on its intended domain but asserted on a
    larger domain where it fails.

    Reference: Eckart & Young (1936); Bernstein, *Matrix Mathematics*,
    Fact 9.14.x. -/
axiom exists_best_rank_approx {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜)
    (k : ℕ) (hk : k ≤ I) :
    ∃ U : Matrix (Fin I) (Fin k) 𝕜,
      Uᴴ * U = 1 ∧
      ∀ B : Matrix (Fin I) (Fin C) 𝕜, B.rank ≤ k →
        (M - U * Uᴴ * M).frobNormSq ≤ (M - B).frobNormSq

end Matrix

end
