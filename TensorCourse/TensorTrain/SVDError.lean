/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorTrain.Existence
import TensorCourse.TensorTrain.ExistenceFull
import TensorCourse.Tensor.ContractiveProdAssoc
import TensorCourse.TuckerDecomposition.ErrorBounds

/-!
# Truncated TT-SVD: the per-step optimal-truncation error (`thm:tt-svd-error`)

The truncated TT-SVD peels one core at a time: at each step it keeps the best
rank-`R` approximation of the current mode split. This file formalizes the
**per-step** content of the error theorem `thm:tt-svd-error`:

* `Tensor.frobNormSq_modeSplit` — the mode split (a bijective reshape) preserves
  the squared Frobenius norm.
* `Tensor.exists_truncated_ctrProd` — every `X` factors as `X = A ⋈ₜ V + E`
  with `ctrLeftMat A` orthonormal of `R` columns and the residual
  `‖E‖² = bestRankApproxErrorSq X_{[a]} R` exactly the **Eckart–Young best
  rank-`R` residual** of the mode split. This is the optimal truncation at one
  TT-SVD step; the full error bound assembles these via a Pythagorean induction.
-/

noncomputable section

open Finset BigOperators
open scoped Matrix

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The squared Frobenius norm as the real part of `tr(Mᴴ M)`. -/
theorem frobNormSq_eq_re_trace {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) :
    M.frobNormSq = RCLike.re (Matrix.trace (Mᴴ * M)) := by
  rw [Matrix.frobNormSq_def, Matrix.trace, map_sum, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Matrix.diag_apply, Matrix.mul_apply, map_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Matrix.conjTranspose_apply, ← starRingEnd_apply, RCLike.conj_mul, ← RCLike.ofReal_pow,
    RCLike.ofReal_re]

/-- Multiplying by an orthonormal-column matrix (`Uᴴ U = 1`) preserves the
    squared Frobenius norm. -/
theorem frobNormSq_orthonormal_mul {I R C : ℕ} {U : Matrix (Fin I) (Fin R) 𝕜}
    (hU : Uᴴ * U = 1) (N : Matrix (Fin R) (Fin C) 𝕜) :
    (U * N).frobNormSq = N.frobNormSq := by
  rw [frobNormSq_eq_re_trace, frobNormSq_eq_re_trace]
  congr 2
  rw [Matrix.conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ U N, hU,
    Matrix.one_mul]

/-- **Matrix Pythagoras (Frobenius).** If `Pᴴ Q = 0` then
    `‖P + Q‖² = ‖P‖² + ‖Q‖²`. -/
theorem frobNormSq_add_of_orthogonal {I C : ℕ} {P Q : Matrix (Fin I) (Fin C) 𝕜}
    (h : Pᴴ * Q = 0) :
    (P + Q).frobNormSq = P.frobNormSq + Q.frobNormSq := by
  have h2 : Qᴴ * P = 0 := by
    have := congrArg Matrix.conjTranspose h
    rwa [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose,
      Matrix.conjTranspose_zero] at this
  rw [frobNormSq_eq_re_trace, frobNormSq_eq_re_trace, frobNormSq_eq_re_trace,
    Matrix.conjTranspose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, h, h2]
  simp [Matrix.trace_add, map_add]

/-- **An orthogonal projection does not increase the Frobenius norm.** For
    `W` with orthonormal columns (`Wᴴ W = 1`), the projector `W Wᴴ` satisfies
    `‖W Wᴴ N‖ ≤ ‖N‖`. (Pythagoras: `N = W Wᴴ N + (N − W Wᴴ N)` with the two
    summands orthogonal.) -/
theorem frobNormSq_orthonormal_proj_le {I R C : ℕ} {W : Matrix (Fin I) (Fin R) 𝕜}
    (hW : Wᴴ * W = 1) (N : Matrix (Fin I) (Fin C) 𝕜) :
    (W * Wᴴ * N).frobNormSq ≤ N.frobNormSq := by
  have hPidem : W * Wᴴ * (W * Wᴴ) = W * Wᴴ := by
    rw [Matrix.mul_assoc W Wᴴ (W * Wᴴ), ← Matrix.mul_assoc Wᴴ W Wᴴ, hW, Matrix.one_mul]
  have hPadj : (W * Wᴴ)ᴴ = W * Wᴴ := by
    rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_conjTranspose]
  have horth : (W * Wᴴ * N)ᴴ * (N - W * Wᴴ * N) = 0 := by
    rw [Matrix.conjTranspose_mul, hPadj, Matrix.mul_sub,
      ← Matrix.mul_assoc (Nᴴ * (W * Wᴴ)) (W * Wᴴ) N,
      Matrix.mul_assoc Nᴴ (W * Wᴴ) (W * Wᴴ), hPidem, sub_self]
  have hsum : (W * Wᴴ * N).frobNormSq + (N - W * Wᴴ * N).frobNormSq = N.frobNormSq := by
    rw [← Matrix.frobNormSq_add_of_orthogonal horth]; congr 1; abel
  rw [← hsum]
  exact le_add_of_nonneg_right (Matrix.frobNormSq_nonneg _)

/-- The best rank-`k` residual, unpacked: there is an orthonormal-column factor
    `U` (the Eckart–Young projector of axiom B2) with
    `bestRankApproxErrorSq M k = ‖M − U Uᴴ M‖²`. -/
theorem bestRankApproxErrorSq_eq_of {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜)
    (k : ℕ) (hk : k ≤ I) :
    ∃ U : Matrix (Fin I) (Fin k) 𝕜, Uᴴ * U = 1 ∧
      Matrix.bestRankApproxErrorSq M k = (M - U * Uᴴ * M).frobNormSq := by
  refine ⟨Classical.choose (Matrix.exists_best_rank_approx M k hk),
    (Classical.choose_spec (Matrix.exists_best_rank_approx M k hk)).1, ?_⟩
  rw [Matrix.bestRankApproxErrorSq, dif_pos hk]

/-- **Orthonormal compression cannot worsen the best rank-`R` residual** (the
    analytic content of the carried-residual step in `thm:tt-svd-error`). For
    `W` with orthonormal columns (`Wᴴ W = 1`),

      `bestRankApproxErrorSq (Wᴴ M) R ≤ bestRankApproxErrorSq M R`.

    Compress the best rank-`R` approximation of `M` by `Wᴴ` to get a rank-`≤ R`
    competitor for `Wᴴ M`, then use that `Wᴴ` (equivalently the projector
    `W Wᴴ`) does not increase the Frobenius norm. -/
theorem bestRankApproxErrorSq_orthonormal_compress {I I' C : ℕ}
    {W : Matrix (Fin I) (Fin I') 𝕜} (hW : Wᴴ * W = 1)
    (M : Matrix (Fin I) (Fin C) 𝕜) (R : ℕ) (hR : R ≤ I') :
    Matrix.bestRankApproxErrorSq (Wᴴ * M) R ≤ Matrix.bestRankApproxErrorSq M R := by
  have hI' : I' ≤ I :=
    calc I' = (Wᴴ * W).rank := by rw [hW, Matrix.rank_one, Fintype.card_fin]
    _ ≤ Wᴴ.rank := Matrix.rank_mul_le_left _ _
    _ ≤ I := Matrix.rank_le_width Wᴴ
  obtain ⟨U, hUorth, hUeq⟩ := bestRankApproxErrorSq_eq_of M R (le_trans hR hI')
  rw [hUeq]
  have hrank : (Wᴴ * (U * Uᴴ * M)).rank ≤ R :=
    le_trans (Matrix.rank_mul_le_right _ _)
      (le_trans (Matrix.rank_mul_le_left _ _)
        (le_trans (Matrix.rank_mul_le_left _ _) (Matrix.rank_le_width U)))
  calc Matrix.bestRankApproxErrorSq (Wᴴ * M) R
      ≤ ((Wᴴ * M) - Wᴴ * (U * Uᴴ * M)).frobNormSq :=
        Matrix.bestRankApproxErrorSq_le (Wᴴ * M) R hR _ hrank
    _ = (Wᴴ * (M - U * Uᴴ * M)).frobNormSq := by rw [Matrix.mul_sub]
    _ = (W * Wᴴ * (M - U * Uᴴ * M)).frobNormSq := by
        rw [← Matrix.frobNormSq_orthonormal_mul hW (Wᴴ * (M - U * Uᴴ * M)),
          ← Matrix.mul_assoc W Wᴴ (M - U * Uᴴ * M)]
    _ ≤ (M - U * Uᴴ * M).frobNormSq := Matrix.frobNormSq_orthonormal_proj_le hW _

/-- **Reindexing preserves orthonormal columns.** If `Mᴴ M = 1`, then for any
    row/column re-indexings `e₁, e₂`, the reindexed matrix again has orthonormal
    columns. (Used to flatten a Kronecker product `1 ⊗ U` into a single-`Fin`
    compression matrix while keeping `Wᴴ W = 1`.) -/
theorem reindex_conjTranspose_mul_self {m n m' n' : Type*} [Fintype m]
    [Fintype m'] [DecidableEq n] [DecidableEq n'] (M : Matrix m n 𝕜)
    (hM : Mᴴ * M = 1) (e₁ : m ≃ m') (e₂ : n ≃ n') :
    (Matrix.reindex e₁ e₂ M)ᴴ * (Matrix.reindex e₁ e₂ M) = 1 := by
  simp only [Matrix.reindex_apply, Matrix.conjTranspose_submatrix]
  rw [Matrix.submatrix_mul_equiv, hM, Matrix.submatrix_one_equiv]

/-- `1 ⊗ₜ U` has orthonormal columns when `U` does. -/
theorem one_kronecker_orthonormal {K I R : ℕ} (U : Matrix (Fin I) (Fin R) 𝕜)
    (hU : Uᴴ * U = 1) :
    ((1 : Matrix (Fin K) (Fin K) 𝕜) ⊗ₜ U)ᴴ * ((1 : Matrix (Fin K) (Fin K) 𝕜) ⊗ₜ U) = 1 := by
  refine Matrix.kronecker_orthonormalCols _ _ ?_ hU
  change (1 : Matrix (Fin K) (Fin K) 𝕜)ᴴ * 1 = 1
  rw [Matrix.conjTranspose_one, Matrix.one_mul]

/-- Reindexing the rows/columns of a matrix by equivalences preserves the
    squared Frobenius norm. -/
theorem frobNormSq_submatrix_equiv {I C I' C' : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜)
    (e₁ : Fin I' ≃ Fin I) (e₂ : Fin C' ≃ Fin C) :
    (M.submatrix e₁ e₂).frobNormSq = M.frobNormSq := by
  simp only [Matrix.frobNormSq_def, Matrix.submatrix_apply]
  rw [← Equiv.sum_comp e₁ (fun i => ∑ j, ‖M i j‖ ^ 2)]
  exact Finset.sum_congr rfl fun i' _ => Equiv.sum_comp e₂ (fun j => ‖M (e₁ i') j‖ ^ 2)

/-- The best rank-`R` residual is invariant under row/column reindexing. -/
theorem bestRankApproxErrorSq_reindex {I C I' C' : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜)
    (e₁ : Fin I' ≃ Fin I) (e₂ : Fin C' ≃ Fin C) (R : ℕ) :
    bestRankApproxErrorSq (M.submatrix e₁ e₂) R = bestRankApproxErrorSq M R := by
  have hII : I' = I := by simpa using Fintype.card_eq.mpr ⟨e₁⟩
  subst hII
  have hsub : ∀ (A B : Matrix (Fin I') (Fin C) 𝕜),
      A.submatrix e₁ e₂ - B.submatrix e₁ e₂ = (A - B).submatrix e₁ e₂ :=
    fun A B => by funext i j; simp [Matrix.submatrix_apply, Matrix.sub_apply]
  by_cases hR : R ≤ I'
  · refine le_antisymm ?_ ?_
    · obtain ⟨U, _, hUeq⟩ := bestRankApproxErrorSq_eq_of M R hR
      refine (bestRankApproxErrorSq_le (M.submatrix e₁ e₂) R hR
        ((U * Uᴴ * M).submatrix e₁ e₂) ?_).trans_eq ?_
      · exact (Matrix.rank_submatrix_le _ _ _).trans
          (le_trans (Matrix.rank_mul_le_left _ _)
            (le_trans (Matrix.rank_mul_le_left _ _) (Matrix.rank_le_width _)))
      · rw [hsub, frobNormSq_submatrix_equiv, hUeq]
    · obtain ⟨U, _, hUeq⟩ := bestRankApproxErrorSq_eq_of (M.submatrix e₁ e₂) R hR
      refine (bestRankApproxErrorSq_le M R hR
        ((U * Uᴴ * M.submatrix e₁ e₂).submatrix e₁.symm e₂.symm) ?_).trans_eq ?_
      · exact (Matrix.rank_submatrix_le _ _ _).trans
          (le_trans (Matrix.rank_mul_le_left _ _)
            (le_trans (Matrix.rank_mul_le_left _ _) (Matrix.rank_le_width _)))
      · rw [show M - (U * Uᴴ * M.submatrix e₁ e₂).submatrix e₁.symm e₂.symm
              = (M.submatrix e₁ e₂ - U * Uᴴ * M.submatrix e₁ e₂).submatrix e₁.symm e₂.symm from by
            funext i j
            simp [Matrix.submatrix_apply, Matrix.sub_apply, Equiv.apply_symm_apply],
          frobNormSq_submatrix_equiv, hUeq]
  · rw [bestRankApproxErrorSq, bestRankApproxErrorSq, dif_neg hR, dif_neg hR]

/-- For `k = I` (full row rank target) the best rank-`k` residual is `0`: a
    square orthonormal-column `U` is unitary, so `U Uᴴ = 1` and `M = U Uᴴ M`. -/
theorem bestRankApproxErrorSq_self {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) :
    bestRankApproxErrorSq M I = 0 := by
  obtain ⟨U, hUorth, hUeq⟩ := bestRankApproxErrorSq_eq_of M I (le_refl I)
  rw [hUeq]
  have hUU : U * Uᴴ = 1 := mul_eq_one_comm.mp hUorth
  rw [hUU, Matrix.one_mul, sub_self, Matrix.frobNormSq_def]
  simp

/-- **Capping the truncation rank at the row count is harmless.** Since for
    `R ≥ I` the best rank-`R` residual is `0`, replacing `R` by `min R I`
    leaves `bestRankApproxErrorSq` unchanged. Lets the recursion peel with a
    capped rank `min (ranks k) (∏ da)` (required by `exists_truncated_ctrProd`)
    while still matching the uncapped `kSplitErr`. -/
theorem bestRankApproxErrorSq_min {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) (R : ℕ) :
    bestRankApproxErrorSq M (min R I) = bestRankApproxErrorSq M R := by
  rcases le_or_gt R I with hR | hR
  · rw [min_eq_left hR]
  · rw [min_eq_right hR.le, bestRankApproxErrorSq_self, bestRankApproxErrorSq, dif_neg (by omega)]

/-- Multiplying by `Uᴴ` (with `Uᴴ U = 1`) does not increase the Frobenius norm. -/
theorem frobNormSq_conjTranspose_mul_le {I R C : ℕ} {U : Matrix (Fin I) (Fin R) 𝕜}
    (hU : Uᴴ * U = 1) (M : Matrix (Fin I) (Fin C) 𝕜) :
    (Uᴴ * M).frobNormSq ≤ M.frobNormSq := by
  rw [← Matrix.frobNormSq_orthonormal_mul hU (Uᴴ * M), ← Matrix.mul_assoc]
  exact Matrix.frobNormSq_orthonormal_proj_le hU M

/-- The best rank-`k` residual is nonnegative (it is either a squared Frobenius
    norm or the placeholder `0`). -/
theorem bestRankApproxErrorSq_nonneg {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜) (k : ℕ) :
    0 ≤ bestRankApproxErrorSq M k := by
  rw [bestRankApproxErrorSq]
  split_ifs
  · exact frobNormSq_nonneg _
  · exact le_refl 0

/-- If `M` already has rank `≤ k`, its best rank-`k` residual is `0` (`M` is its
    own best rank-`k` approximation). Used to discard the trailing degenerate
    split (a single-column matricization) of the order-`D` tensor error bound. -/
theorem bestRankApproxErrorSq_eq_zero_of_rank_le {I C : ℕ} (M : Matrix (Fin I) (Fin C) 𝕜)
    (k : ℕ) (hk : M.rank ≤ k) :
    bestRankApproxErrorSq M k = 0 := by
  by_cases h : k ≤ I
  · refine le_antisymm ((bestRankApproxErrorSq_le M k h M hk).trans_eq ?_)
      (bestRankApproxErrorSq_nonneg _ _)
    rw [sub_self, frobNormSq_def]; simp
  · rw [bestRankApproxErrorSq, dif_neg h]

end Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {p q : ℕ} {I₁ : Fin p → ℕ} {I₂ : Fin q → ℕ}

/-- Reindexing respects subtraction. -/
theorem reindex_sub {D₁ D₂ : ℕ} {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (T₁ T₂ : Tensor 𝕜 D₁ dims₁) :
    reindex e (T₁ - T₂) = reindex e T₁ - reindex e T₂ := by
  ext idx; simp [reindex]

/-- Reindexing a tensor (a bijective relabeling of multi-indices) preserves the
    squared Frobenius norm. -/
theorem frobNormSq_reindex {D₁ D₂ : ℕ} {dims₁ : Fin D₁ → ℕ} {dims₂ : Fin D₂ → ℕ}
    (e : ((d : Fin D₁) → Fin (dims₁ d)) ≃ ((d : Fin D₂) → Fin (dims₂ d)))
    (T : Tensor 𝕜 D₁ dims₁) :
    ‖reindex e T‖_F² = ‖T‖_F² := by
  simp only [frobNormSq_def, reindex_apply]
  exact Equiv.sum_comp e.symm (fun idx' => ‖T idx'‖ ^ 2)

/-- **Cons-peeling formula for `multiIndexEquiv` (value form).** Flattening a
    multi-index over `dims : Fin (D+1) → ℕ` puts mode `0` at the fastest-varying
    position: the flat value is `idx₀ + dims₀ · (flatten of the tail)`. This is
    the structural recursion of `multiIndexEquiv` read off at the value level. -/
theorem multiIndexEquiv_val_cons {D : ℕ} (dims : Fin (D + 1) → ℕ)
    (idx : (d : Fin (D + 1)) → Fin (dims d)) :
    ((multiIndexEquiv dims) idx).val
      = (idx 0).val
        + dims 0 * ((multiIndexEquiv (fun i => dims i.succ)) (fun i => idx i.succ)).val := by
  conv_lhs => rw [multiIndexEquiv]
  rfl

/-- **`multiIndexEquiv` value is invariant under an order-preserving
    re-indexing of the modes** (`Fin.cast` of a dimension-count equality). The
    flattened value only depends on the dims/index data, not on how the mode
    count is presented. -/
theorem multiIndexEquiv_val_cast {n n' : ℕ} (h : n = n')
    (dims : Fin n' → ℕ) (idx : (d : Fin n') → Fin (dims d)) :
    ((multiIndexEquiv (dims ∘ Fin.cast h)) (fun i => idx (Fin.cast h i))).val
      = ((multiIndexEquiv dims) idx).val := by
  subst h
  rfl

/-- **Closed form (mixed-radix) for the `multiIndexEquiv` value.** With mode `0`
    fastest-varying, the flat value is the Horner/mixed-radix sum
    `∑ₖ idxₖ · (∏_{j<k} dimsⱼ)`, the inner product being `Fin.partialProd`. This
    reduces flattening identities to pure `Finset` algebra (no dependent casts). -/
theorem multiIndexEquiv_val_eq_sum :
    ∀ {D : ℕ} (dims : Fin D → ℕ) (idx : (d : Fin D) → Fin (dims d)),
      ((multiIndexEquiv dims) idx).val
        = ∑ k : Fin D, (idx k).val * Fin.partialProd dims k.castSucc
  | 0, dims, idx => by
      rw [Fin.sum_univ_zero]; rfl
  | D' + 1, dims, idx => by
      rw [multiIndexEquiv_val_cons, Fin.sum_univ_succ,
        multiIndexEquiv_val_eq_sum (fun i => dims i.succ) (fun i => idx i.succ),
        Fin.castSucc_zero, Fin.partialProd_zero, mul_one, Finset.mul_sum]
      congr 1
      apply Finset.sum_congr rfl
      intro k _
      rw [← Fin.succ_castSucc, Fin.partialProd_succ',
        show Fin.tail dims = (fun i => dims i.succ) from rfl]
      ring

/-- Partial product of `combinedDims` up to a point inside the first block:
    it only sees the first factor `I₁`. -/
theorem partialProd_combinedDims_castAdd {p' q' : ℕ} (I₁ : Fin p' → ℕ) (I₂ : Fin q' → ℕ)
    (i : Fin p') :
    Fin.partialProd (combinedDims I₁ I₂) (Fin.castAdd q' i).castSucc
      = Fin.partialProd I₁ i.castSucc := by
  have hlen : (i : ℕ) ≤ (List.ofFn I₁).length := by rw [List.length_ofFn]; omega
  simp only [Fin.partialProd, combinedDims_eq_append, List.ofFn_fin_append,
    Fin.val_castSucc, Fin.val_castAdd]
  rw [List.take_append_of_le_length hlen]

/-- Partial product of `combinedDims` up to a point inside the second block:
    it sees all of `I₁` and the first part of `I₂`. -/
theorem partialProd_combinedDims_natAdd {p' q' : ℕ} (I₁ : Fin p' → ℕ) (I₂ : Fin q' → ℕ)
    (j : Fin q') :
    Fin.partialProd (combinedDims I₁ I₂) (Fin.natAdd p' j).castSucc
      = (∏ d, I₁ d) * Fin.partialProd I₂ j.castSucc := by
  simp only [Fin.partialProd, combinedDims_eq_append, List.ofFn_fin_append,
    Fin.val_castSucc, Fin.val_natAdd]
  rw [show p' + (↑j : ℕ) = (List.ofFn I₁).length + ↑j from by rw [List.length_ofFn],
    List.take_append, List.prod_append,
    List.take_of_length_le (Nat.le_add_right (List.ofFn I₁).length ↑j),
    Nat.add_sub_cancel_left, List.prod_ofFn]

/-- **Helper 1: `multiIndexEquiv` factors over `combinedDims`.** Flattening a
    combined multi-index is the mixed-radix combination of the two parts, with
    the first block `I₁` fastest-varying: `flat = flat₁ + (∏ I₁) · flat₂`. -/
theorem multiIndexEquiv_val_combinedDims {p' q' : ℕ} (I₁ : Fin p' → ℕ) (I₂ : Fin q' → ℕ)
    (f₁ : (d : Fin p') → Fin (I₁ d)) (f₂ : (d : Fin q') → Fin (I₂ d)) :
    ((multiIndexEquiv (combinedDims I₁ I₂)) (combineIndex f₁ f₂)).val
      = (multiIndexEquiv I₁ f₁).val + (∏ d, I₁ d) * (multiIndexEquiv I₂ f₂).val := by
  rw [multiIndexEquiv_val_eq_sum (combinedDims I₁ I₂) (combineIndex f₁ f₂),
    Fin.sum_univ_add, multiIndexEquiv_val_eq_sum I₁ f₁, multiIndexEquiv_val_eq_sum I₂ f₂,
    Finset.mul_sum]
  congr 1
  · apply Finset.sum_congr rfl
    intro i _
    rw [partialProd_combinedDims_castAdd,
      show (combineIndex f₁ f₂ (Fin.castAdd q' i)).val = (f₁ i).val from by
        simp only [combineIndex, Fin.addCases_left, Fin.val_cast]]
  · apply Finset.sum_congr rfl
    intro j _
    rw [partialProd_combinedDims_natAdd,
      show (combineIndex f₁ f₂ (Fin.natAdd p' j)).val = (f₂ j).val from by
        simp only [combineIndex, Fin.addCases_right, Fin.val_cast]]
    ring

/-- The product of a `combinedDims` shape factors as the product of the parts. -/
theorem prod_combinedDims {p' q' : ℕ} (I₁ : Fin p' → ℕ) (I₂ : Fin q' → ℕ) :
    ∏ d, combinedDims I₁ I₂ d = (∏ d, I₁ d) * (∏ d, I₂ d) := by
  rw [Fin.prod_univ_add]
  congr 1
  · exact Finset.prod_congr rfl fun i _ => combinedDims_castAdd i
  · exact Finset.prod_congr rfl fun j _ => combinedDims_natAdd j

/-- The product of a `Fin.cons` shape factors off the head. -/
theorem prod_cons {m : ℕ} (R : ℕ) (dm : Fin m → ℕ) :
    ∏ d, (Fin.cons R dm : Fin (m + 1) → ℕ) d = R * ∏ d, dm d := by
  simp only [Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]

/-- The mode split is a bijective reshape, so it preserves the squared
    Frobenius norm. -/
theorem frobNormSq_modeSplit (X : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) :
    (modeSplit X).frobNormSq = ‖X‖_F² := by
  rw [Matrix.frobNormSq_def, frobNormSq_def]
  rw [← Equiv.sum_comp
        ((splitIndexEquiv (I₁ := I₁) (I₂ := I₂)).trans
          (Equiv.prodCongr (multiIndexEquiv I₁) (multiIndexEquiv I₂))).symm
        (fun idx => ‖X idx‖ ^ 2)]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun ri _ => Finset.sum_congr rfl fun cj _ => ?_
  rw [modeSplit_apply]
  rfl

/-- Index equivalence behind `ctrRightMat`: a `cons`-shaped multi-index splits
    into the contracted mode `Fin S` and the flattened free modes
    `Fin (∏ db)`. -/
def ctrRightIndexEquiv {b : ℕ} {db : Fin b → ℕ} {S : ℕ} :
    (Fin S × Fin (∏ d, db d)) ≃
      ((d : Fin (b + 1)) → Fin ((Fin.cons S db : Fin (b + 1) → ℕ) d)) where
  toFun p := Fin.cons (Fin.cast (by simp) p.1)
    (fun i => Fin.cast (by simp) ((multiIndexEquiv db).symm p.2 i))
  invFun idx := (Fin.cast (by simp) (idx 0),
    multiIndexEquiv db (fun i => Fin.cast (by simp) (idx i.succ)))
  left_inv p := by
    refine Prod.ext ?_ ?_
    · apply Fin.ext; simp
    · simp only [Fin.cons_succ]
      rw [show (fun i => Fin.cast (by simp) (Fin.cast (by simp)
            ((multiIndexEquiv db).symm p.2 i))) = (multiIndexEquiv db).symm p.2 from by
        funext i; apply Fin.ext; simp]
      simp
  right_inv idx := by
    funext d
    refine Fin.cases ?_ (fun i => ?_) d
    · simp only [Fin.cons_zero]; apply Fin.ext; simp
    · simp only [Fin.cons_succ, Equiv.symm_apply_apply]; apply Fin.ext; simp

/-- The first-mode matricization `ctrRightMat` preserves the squared Frobenius
    norm (it is a bijective reshape). -/
theorem frobNormSq_ctrRightMat {b : ℕ} {db : Fin b → ℕ} {S : ℕ}
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    (ctrRightMat B).frobNormSq = ‖B‖_F² := by
  rw [Matrix.frobNormSq_def, frobNormSq_def]
  rw [← Equiv.sum_comp ctrRightIndexEquiv (fun idx => ‖B idx‖ ^ 2)]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun cj _ => ?_
  rw [ctrRightMat]
  rfl

/-- `ctrRightMat` is additive under subtraction. -/
theorem ctrRightMat_sub {b : ℕ} {db : Fin b → ℕ} {S : ℕ}
    (B C : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    ctrRightMat (B - C) = ctrRightMat B - ctrRightMat C := by
  funext s cj
  simp only [ctrRightMat, Matrix.of_apply, Matrix.sub_apply, Pi.sub_apply]

/-- **Orthonormal contractive product preserves the Frobenius norm.** If the
    left factor's boundary matricization `ctrLeftMat A` has orthonormal
    columns, then `‖A ⋈ₜ B‖ = ‖B‖` (the Pythagorean-step ingredient). -/
theorem frobNormSq_ctrProd_orthonormal {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ} {S : ℕ}
    (A : Tensor 𝕜 (a + 1) (Fin.snoc da S)) (B : Tensor 𝕜 (b + 1) (Fin.cons S db))
    (hA : (ctrLeftMat A)ᴴ * ctrLeftMat A = 1) :
    ‖A ⋈ₜ B‖_F² = ‖B‖_F² := by
  rw [← frobNormSq_modeSplit, modeSplit_ctrProd, Matrix.frobNormSq_orthonormal_mul hA,
    frobNormSq_ctrRightMat]

variable {a b : ℕ} {da : Fin a → ℕ} {db : Fin b → ℕ}

/-- **Entrywise formula for the carried residual.** The carried residual
    `V = ctrRightFold(Uᴴ · modeSplit X)` of one TT-SVD step is, pointwise,
    `V(s, js) = ∑ₜ Uᴴ(s,t) · X(combineIndex (mIE⁻¹ t) js)`: it compresses the
    first-block (`da`) index of `X` by `Uᴴ`, leaving the trailing modes `db`
    untouched. This exposes `V` directly in terms of `X` and `U`, the starting
    point for the carried-residual reshape identity. -/
theorem ctrRightFold_modeSplit_apply {R : ℕ} (X : Tensor 𝕜 (a + b) (combinedDims da db))
    (U : Matrix (Fin (∏ d, da d)) (Fin R) 𝕜) (s : Fin R)
    (js : (d : Fin b) → Fin (db d)) :
    (ctrRightFold (Uᴴ * modeSplit X) : Tensor 𝕜 (b + 1) (Fin.cons R db)) (Fin.cons s js)
      = ∑ t, (Uᴴ) s t * X (combineIndex ((multiIndexEquiv da).symm t) js) := by
  have hrow : (Fin.cast (@Fin.cons_zero b (fun _ => ℕ) R db)
      ((Fin.cons s js : (d : Fin (b + 1)) → Fin ((Fin.cons R db : Fin (b + 1) → ℕ) d)) 0)) = s := by
    apply Fin.ext; simp
  have hcol : (multiIndexEquiv db fun i => Fin.cast (@Fin.cons_succ b (fun _ => ℕ) R db i)
      ((Fin.cons s js : (d : Fin (b + 1)) → Fin ((Fin.cons R db : Fin (b + 1) → ℕ) d))
        (Fin.succ i)))
      = multiIndexEquiv db js := by
    congr 1
  simp only [ctrRightFold, hrow, hcol, Matrix.mul_apply, modeSplit_apply,
    Equiv.symm_apply_apply]

/-- **The carried residual is non-expansive in the difference.** Compressing two
    tensors by the same `Uᴴ` (orthonormal columns) into carried residuals does
    not increase the Frobenius distance: `‖V_X − V_Y‖ ≤ ‖X − Y‖`. This is the
    ingredient that lets the quasi-optimality recursion carry a competitor. -/
theorem frobNormSq_carried_residual_sub_le {R : ℕ}
    (X Y : Tensor 𝕜 (a + b) (combinedDims da db))
    (U : Matrix (Fin (∏ d, da d)) (Fin R) 𝕜) (hU : Uᴴ * U = 1) :
    ‖(ctrRightFold (Uᴴ * modeSplit X) : Tensor 𝕜 (b + 1) (Fin.cons R db))
        - ctrRightFold (Uᴴ * modeSplit Y)‖_F²
      ≤ ‖X - Y‖_F² := by
  rw [← frobNormSq_ctrRightMat (ctrRightFold (Uᴴ * modeSplit X)
        - ctrRightFold (Uᴴ * modeSplit Y)), ctrRightMat_sub, ctrRightMat_ctrRightFold,
    ctrRightMat_ctrRightFold, ← Matrix.mul_sub, ← modeSplit_sub]
  exact (Matrix.frobNormSq_conjTranspose_mul_le hU _).trans_eq (frobNormSq_modeSplit _)

/-! ### The explicit per-step truncation witness

The peel-and-truncate step is a genuine *function* of `X` and the target rank,
not merely an existence claim: at each step we take the Eckart–Young best
rank-`R` projector `U` (axiom B2's `Classical.choose`) and read off the left
factor `A = ctrLeftFold U` and the carried residual `V = ctrRightFold(Uᴴ X_{[a]})`.
Naming these (`truncProj`, `truncLeft`, `truncRight`) lets the truncated TT-SVD
output `ttApproxOpen` below be an explicit witness, so the error theorems are stated
about a *named* approximation rather than a bare existential. -/

/-- The Eckart–Young best rank-`R` projector of the mode split `X_{[a]}` (axiom
    B2's chosen orthonormal-column matrix `U`); the witness behind one TT-SVD
    truncation step. -/
noncomputable def truncProj (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) : Matrix (Fin (∏ d, da d)) (Fin R) 𝕜 :=
  Classical.choose (Matrix.exists_best_rank_approx (modeSplit X) R hR)

theorem truncProj_orthonormal (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) : (truncProj X R hR)ᴴ * truncProj X R hR = 1 :=
  (Classical.choose_spec (Matrix.exists_best_rank_approx (modeSplit X) R hR)).1

/-- The **left truncation factor** `A` of one TT-SVD step (an explicit witness):
    the fold of the best-rank projector `truncProj`. -/
noncomputable def truncLeft (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) : Tensor 𝕜 (a + 1) (Fin.snoc da R) :=
  ctrLeftFold (truncProj X R hR)

/-- The **carried residual** `V` of one TT-SVD step (an explicit witness):
    `ctrRightFold(Uᴴ X_{[a]})` for the best-rank projector `U = truncProj`. -/
noncomputable def truncRight (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) : Tensor 𝕜 (b + 1) (Fin.cons R db) :=
  ctrRightFold ((truncProj X R hR)ᴴ * modeSplit X)

@[simp] theorem ctrLeftMat_truncLeft (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) : ctrLeftMat (truncLeft X R hR) = truncProj X R hR :=
  ctrLeftMat_ctrLeftFold _

theorem truncLeft_orthonormal (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) :
    (ctrLeftMat (truncLeft X R hR))ᴴ * ctrLeftMat (truncLeft X R hR) = 1 := by
  rw [ctrLeftMat_truncLeft]; exact truncProj_orthonormal X R hR

theorem ctrRightMat_truncRight (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) :
    ctrRightMat (truncRight X R hR) = (ctrLeftMat (truncLeft X R hR))ᴴ * modeSplit X := by
  rw [truncRight, ctrRightMat_ctrRightFold, ctrLeftMat_truncLeft]

/-- **Per-step truncation error.** The residual of one explicit TT-SVD step is
    exactly the Eckart–Young best rank-`R` residual of the mode split. -/
theorem truncStep_err (X : Tensor 𝕜 (a + b) (combinedDims da db)) (R : ℕ)
    (hR : R ≤ ∏ d, da d) :
    ‖X - truncLeft X R hR ⋈ₜ truncRight X R hR‖_F²
      = Matrix.bestRankApproxErrorSq (modeSplit X) R := by
  have hUbest := (Classical.choose_spec (Matrix.exists_best_rank_approx (modeSplit X) R hR)).2
  rw [truncLeft, truncRight, ← frobNormSq_modeSplit, modeSplit_sub, modeSplit_ctrProd,
    ctrLeftMat_ctrLeftFold, ctrRightMat_ctrRightFold, ← Matrix.mul_assoc]
  have hr : ∀ C : Matrix (Fin (∏ d, da d)) (Fin R) 𝕜, (C * Cᴴ * modeSplit X).rank ≤ R :=
    fun C => le_trans (Matrix.rank_mul_le_left _ _)
      (le_trans (Matrix.rank_mul_le_left _ _) (Matrix.rank_le_width _))
  refine le_antisymm ?_ ?_
  · rw [Matrix.bestRankApproxErrorSq, dif_pos hR]
    exact hUbest _ (hr _)
  · exact Matrix.bestRankApproxErrorSq_le (modeSplit X) R hR _ (hr _)

/-- **Per-step optimal truncation (one TT-SVD step).** Every `X` factors as
    `X = A ⋈ₜ V` plus an error whose squared Frobenius norm is exactly the
    Eckart–Young best rank-`R` residual of the mode split `X_{[a]}`, with
    `ctrLeftMat A` orthonormal (`R` columns). The truncated analogue of
    `exists_ctrProd_of_split` (which uses the exact split rank). The explicit
    witnesses are `truncLeft`/`truncRight`. -/
theorem exists_truncated_ctrProd (X : Tensor 𝕜 (a + b) (combinedDims da db))
    (R : ℕ) (hR : R ≤ ∏ d, da d) :
    ∃ (A : Tensor 𝕜 (a + 1) (Fin.snoc da R))
      (V : Tensor 𝕜 (b + 1) (Fin.cons R db)),
      (ctrLeftMat A)ᴴ * ctrLeftMat A = 1 ∧
      ctrRightMat V = (ctrLeftMat A)ᴴ * modeSplit X ∧
      ‖X - A ⋈ₜ V‖_F² = Matrix.bestRankApproxErrorSq (modeSplit X) R :=
  ⟨truncLeft X R hR, truncRight X R hR, truncLeft_orthonormal X R hR,
    ctrRightMat_truncRight X R hR, truncStep_err X R hR⟩

/-- **Pythagorean step error (the inductive heart of `thm:tt-svd-error`).**
    If `A` is the orthonormal truncation factor and `V` the exact remainder
    (`ctrRightMat V = (ctrLeftMat A)ᴴ X_{[a]}`), then for *any* remainder
    approximation `V'` the total error splits Pythagoreanly:

      `‖X − A ⋈ₜ V'‖² = ‖X − A ⋈ₜ V‖² + ‖V − V'‖²`.

    The first term is the (fixed) truncation error of this step; the second is
    the remainder error, which the induction bounds recursively. -/
theorem tt_truncation_step_error (X : Tensor 𝕜 (a + b) (combinedDims da db))
    {S : ℕ} (A : Tensor 𝕜 (a + 1) (Fin.snoc da S)) (V : Tensor 𝕜 (b + 1) (Fin.cons S db))
    (hAorth : (ctrLeftMat A)ᴴ * ctrLeftMat A = 1)
    (hV : ctrRightMat V = (ctrLeftMat A)ᴴ * modeSplit X)
    (V' : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    ‖X - A ⋈ₜ V'‖_F² = ‖X - A ⋈ₜ V‖_F² + ‖V - V'‖_F² := by
  rw [← frobNormSq_modeSplit (X - A ⋈ₜ V'), modeSplit_sub, modeSplit_ctrProd,
    ← frobNormSq_modeSplit (X - A ⋈ₜ V), modeSplit_sub, modeSplit_ctrProd]
  rw [show modeSplit X - ctrLeftMat A * ctrRightMat V'
        = (modeSplit X - ctrLeftMat A * ctrRightMat V)
          + ctrLeftMat A * (ctrRightMat V - ctrRightMat V') from by rw [Matrix.mul_sub]; abel]
  rw [Matrix.frobNormSq_add_of_orthogonal ?_]
  · rw [Matrix.frobNormSq_orthonormal_mul hAorth, ← ctrRightMat_sub, frobNormSq_ctrRightMat]
  · -- orthogonality: `(M − U·ctrRightMat V)ᴴ · (U · (…)) = 0` via `(M − U Uᴴ M)ᴴ U = 0`
    rw [← Matrix.mul_assoc,
      show (modeSplit X - ctrLeftMat A * ctrRightMat V)ᴴ * ctrLeftMat A = 0 from by
        rw [hV, Matrix.conjTranspose_sub, Matrix.conjTranspose_mul, Matrix.conjTranspose_mul,
          Matrix.conjTranspose_conjTranspose, Matrix.sub_mul,
          Matrix.mul_assoc (modeSplit X)ᴴ (ctrLeftMat A) ((ctrLeftMat A)ᴴ),
          Matrix.mul_assoc (modeSplit X)ᴴ (ctrLeftMat A * (ctrLeftMat A)ᴴ) (ctrLeftMat A),
          Matrix.mul_assoc (ctrLeftMat A) ((ctrLeftMat A)ᴴ) (ctrLeftMat A), hAorth,
          Matrix.mul_one, sub_self],
      Matrix.zero_mul]

/-- **The best-rank residual is bounded by any rank-`R` competitor's error
    (no Weierstrass).** For any `Y` whose mode split has rank `≤ R`, the
    Eckart–Young best rank-`R` residual of `X_{[a]}` is at most `‖X − Y‖²`.
    This is the per-split step of the quasi-optimality corollary: it compares
    directly against the *given* competitor `Y`, so no infimum/compactness is
    needed (mirroring `hosvd_quasi_optimal`). -/
theorem bestRankApproxErrorSq_le_competitor (X Y : Tensor 𝕜 (a + b) (combinedDims da db))
    (R : ℕ) (hR : R ≤ ∏ d, da d) (hY : (modeSplit Y).rank ≤ R) :
    Matrix.bestRankApproxErrorSq (modeSplit X) R ≤ ‖X - Y‖_F² := by
  rw [← frobNormSq_modeSplit (X - Y), modeSplit_sub]
  exact Matrix.bestRankApproxErrorSq_le (modeSplit X) R hR (modeSplit Y) hY

/-! ### Carried-residual reshape identity `(★)` -/

section PerStepReshape

variable {A M N R : ℕ} {dA : Fin A → ℕ} {dM : Fin M → ℕ} {dN : Fin N → ℕ}

/-- `X_{[A+M]}`, the mode-`(A+M)` split, expressed in `X`'s native
    `dA | (dM, dN)` grouping (no tensor reindex). -/
def deepSplitL (X : Tensor 𝕜 (A + (M + N)) (combinedDims dA (combinedDims dM dN))) :
    Matrix (Fin (∏ d, combinedDims dA dM d)) (Fin (∏ d, dN d)) 𝕜 :=
  Matrix.of fun ρ jn =>
    X (combineIndex (splitLeft ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
        (combineIndex (splitRight ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
          ((multiIndexEquiv dN).symm jn)))

/-- The mode-`(1+M)` split of the carried residual
    `V : Tensor ((M+N)+1) (cons R (combinedDims dM dN))`. -/
def deepSplitR (V : Tensor 𝕜 (M + N + 1) (Fin.cons R (combinedDims dM dN))) :
    Matrix (Fin (∏ d, (Fin.cons R dM : Fin (M + 1) → ℕ) d)) (Fin (∏ d, dN d)) 𝕜 :=
  Matrix.of fun σ jn =>
    V (Fin.cons (α := fun d => Fin ((Fin.cons R (combinedDims dM dN) : Fin (M + N + 1) → ℕ) d))
        ((multiIndexEquiv (Fin.cons R dM)).symm σ 0)
        (combineIndex (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ))
          ((multiIndexEquiv dN).symm jn)))

/-- Row reindex for the compression matrix: pairs `(dM-flat, dA-flat)` to the
    flattened `combinedDims dA dM` row index. -/
def eRow : (Fin (∏ d, dM d) × Fin (∏ d, dA d)) ≃ Fin (∏ d, combinedDims dA dM d) :=
  (Equiv.prodComm _ _).trans
    (((Equiv.prodCongr (multiIndexEquiv dA).symm (multiIndexEquiv dM).symm).trans
        (splitIndexEquiv (I₁ := dA) (I₂ := dM)).symm).trans
      (multiIndexEquiv (combinedDims dA dM)))

/-- Column reindex: pairs `(dM-flat, bond)` to the flattened `cons R dM` index. -/
def eCol : (Fin (∏ d, dM d) × Fin R) ≃ Fin (∏ d, (Fin.cons R dM : Fin (M + 1) → ℕ) d) :=
  (Equiv.prodComm _ _).trans
    (((Equiv.prodCongr (Equiv.refl (Fin R)) (multiIndexEquiv dM).symm).trans
        (Fin.consEquiv (fun d => Fin ((Fin.cons R dM : Fin (M + 1) → ℕ) d)))).trans
      (multiIndexEquiv (Fin.cons R dM)))

/-- The flattened `1_{∏dM} ⊗ U` compression matrix, mapping the `(A+M)`-split
    row space onto the carried-residual `(1+M)`-split row space. -/
def reshapeW (U : Matrix (Fin (∏ d, dA d)) (Fin R) 𝕜) :
    Matrix (Fin (∏ d, combinedDims dA dM d))
           (Fin (∏ d, (Fin.cons R dM : Fin (M + 1) → ℕ) d)) 𝕜 :=
  Matrix.reindex eRow eCol ((1 : Matrix (Fin (∏ d, dM d)) (Fin (∏ d, dM d)) 𝕜) ⊗ₜ U)

/-- `reshapeW` has orthonormal columns when `U` does. -/
theorem reshapeW_conjTranspose_mul_self (U : Matrix (Fin (∏ d, dA d)) (Fin R) 𝕜)
    (hU : Uᴴ * U = 1) :
    (reshapeW (dM := dM) U)ᴴ * reshapeW (dM := dM) U = 1 :=
  Matrix.reindex_conjTranspose_mul_self _ (Matrix.one_kronecker_orthonormal U hU) eRow eCol

/-- `eRow` sends the pair `(im, ida)` to the flattened combined index. -/
theorem eRow_apply (im : Fin (∏ d, dM d)) (ida : Fin (∏ d, dA d)) :
    eRow (dA := dA) (dM := dM) (im, ida)
      = multiIndexEquiv (combinedDims dA dM)
          (combineIndex ((multiIndexEquiv dA).symm ida) ((multiIndexEquiv dM).symm im)) := by
  simp only [eRow, Equiv.trans_apply, Equiv.prodComm_apply, Prod.swap_prod_mk,
    Equiv.prodCongr_apply, Prod.map_apply, splitIndexEquiv, Equiv.coe_fn_symm_mk]

/-- `eCol.symm` decodes the flattened `cons R dM` index into `(dM-flat, bond)`. -/
theorem eCol_symm_apply (σ : Fin (∏ d, (Fin.cons R dM : Fin (M + 1) → ℕ) d)) :
    (eCol (R := R) (dM := dM)).symm σ
      = (multiIndexEquiv dM (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ)),
         ((multiIndexEquiv (Fin.cons R dM)).symm σ) 0) := by
  rw [Equiv.symm_apply_eq]
  change σ = (multiIndexEquiv (Fin.cons R dM))
            (Fin.cons (α := fun d => Fin ((Fin.cons R dM : Fin (M + 1) → ℕ) d))
              (((multiIndexEquiv (Fin.cons R dM)).symm σ) 0)
              ((multiIndexEquiv dM).symm (multiIndexEquiv dM
                (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ)))))
  rw [Equiv.symm_apply_apply, Fin.cons_self_tail, Equiv.apply_symm_apply]

/-- **The carried-residual reshape identity `(★)`.** The deeper split of the
    carried residual `ctrRightFold (Uᴴ · modeSplit X)` equals the orthonormal
    compression `(reshapeW U)ᴴ` of the deeper split of `X`. -/
theorem deepSplitR_eq
    (X : Tensor 𝕜 (A + (M + N)) (combinedDims dA (combinedDims dM dN)))
    (U : Matrix (Fin (∏ d, dA d)) (Fin R) 𝕜) :
    deepSplitR (ctrRightFold (Uᴴ * modeSplit X))
      = (reshapeW (dM := dM) U)ᴴ * deepSplitL X := by
  ext σ jn
  have key : ∀ (im : Fin (∏ d, dM d)) (ida : Fin (∏ d, dA d)),
      (reshapeW (dM := dM) U)ᴴ σ (eRow (im, ida)) * deepSplitL X (eRow (im, ida)) jn
        = (if im = multiIndexEquiv dM (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ))
            then (Uᴴ ((multiIndexEquiv (Fin.cons R dM)).symm σ 0) ida
              * X (combineIndex ((multiIndexEquiv dA).symm ida)
                  (combineIndex ((multiIndexEquiv dM).symm im) ((multiIndexEquiv dN).symm jn))))
            else 0) := by
    intro im ida
    rw [Matrix.conjTranspose_apply, reshapeW, Matrix.reindex_apply, Matrix.submatrix_apply,
      Equiv.symm_apply_apply, eCol_symm_apply, Matrix.kroneckerMap_apply, Matrix.one_apply,
      deepSplitL, Matrix.of_apply, eRow_apply, Equiv.symm_apply_apply, splitLeft_combineIndex,
      splitRight_combineIndex]
    split_ifs with h
    · simp [mul_comm, Matrix.conjTranspose_apply]
    · simp
  rw [deepSplitR, Matrix.of_apply, ctrRightFold_modeSplit_apply, Matrix.mul_apply,
    ← Equiv.sum_comp (eRow (dA := dA) (dM := dM))
      (fun ρ => (reshapeW (dM := dM) U)ᴴ σ ρ * deepSplitL X ρ jn),
    Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [show (∑ im, (reshapeW (dM := dM) U)ᴴ σ (eRow (im, t)) * deepSplitL X (eRow (im, t)) jn)
        = ∑ im, (if im = multiIndexEquiv dM (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ))
            then (Uᴴ ((multiIndexEquiv (Fin.cons R dM)).symm σ 0) t
              * X (combineIndex ((multiIndexEquiv dA).symm t)
                  (combineIndex ((multiIndexEquiv dM).symm im) ((multiIndexEquiv dN).symm jn))))
            else 0) from Finset.sum_congr rfl fun im _ => key im t,
    Finset.sum_ite_eq']
  simp only [Finset.mem_univ, if_true, Equiv.symm_apply_apply]

/-- **Per-step optimal-truncation comparison (`thm:tt-svd-error`, lines on the
    carried residual).** The best rank-`R'` approximation error of the carried
    residual's deeper split is bounded by that of the corresponding split of the
    original tensor `X`. This is the reshape step of the error bound: combining
    the identity `(★)` (`deepSplitR_eq`) with the orthonormal-compression
    inequality (`bestRankApproxErrorSq_orthonormal_compress`). -/
theorem deepSplit_bestRankApproxErrorSq_le
    (X : Tensor 𝕜 (A + (M + N)) (combinedDims dA (combinedDims dM dN)))
    (U : Matrix (Fin (∏ d, dA d)) (Fin R) 𝕜) (hU : Uᴴ * U = 1) (R' : ℕ)
    (hR' : R' ≤ ∏ d, (Fin.cons R dM : Fin (M + 1) → ℕ) d) :
    Matrix.bestRankApproxErrorSq (deepSplitR (ctrRightFold (Uᴴ * modeSplit X))) R'
      ≤ Matrix.bestRankApproxErrorSq (deepSplitL X) R' := by
  rw [deepSplitR_eq]
  exact Matrix.bestRankApproxErrorSq_orthonormal_compress
    (reshapeW_conjTranspose_mul_self U hU) (deepSplitL X) R' hR'

/-! ### Bridging `deepSplitL`/`deepSplitR` to `modeSplit` of a regrouped tensor

To thread the per-step inequality `deepSplit_bestRankApproxErrorSq_le` through
the recursion (whose hypotheses are stated with `modeSplitAt`), we express each
`deepSplit` matrix as `modeSplit` of an appropriately regrouped tensor. Then
`bestRankApproxErrorSq_reindex` makes everything reindex-invariant. -/

/-- The dim equality powering the associativity regroup
    `A+(M+N) → (A+M)+N`. -/
theorem hdA_assoc (i : Fin ((A + M) + N)) :
    combinedDims (combinedDims dA dM) dN i
      = combinedDims dA (combinedDims dM dN) (Fin.cast (Nat.add_assoc A M N) i) := by
  rw [combinedDims_assoc]; simp only [Function.comp_apply]

/-- The associativity regroup equiv `A+(M+N) → (A+M)+N` at the multi-index
    level. -/
def assocRegroup :
    ((d : Fin (A + (M + N))) → Fin (combinedDims dA (combinedDims dM dN) d)) ≃
      ((d : Fin ((A + M) + N)) → Fin (combinedDims (combinedDims dA dM) dN d)) :=
  castEquiv (Nat.add_assoc A M N).symm (fun i => hdA_assoc i)

/-- `deepSplitL` is `modeSplit` of the associativity-regrouped tensor. -/
theorem deepSplitL_eq_modeSplit
    (X : Tensor 𝕜 (A + (M + N)) (combinedDims dA (combinedDims dM dN))) :
    deepSplitL X = modeSplit (reindex assocRegroup X) := by
  funext ρ jn
  rw [deepSplitL, Matrix.of_apply, modeSplit_apply, reindex_apply]
  congr 1
  rw [assocRegroup]
  funext d
  refine Fin.addCases (motive := fun d =>
      combineIndex (splitLeft ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
          (combineIndex (splitRight ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
            ((multiIndexEquiv dN).symm jn)) d
        = (castEquiv (Nat.add_assoc A M N).symm (fun i => hdA_assoc i)).symm
            (combineIndex ((multiIndexEquiv (combinedDims dA dM)).symm ρ)
              ((multiIndexEquiv dN).symm jn)) d)
    (fun iA => ?_) (fun iMN => ?_) d
  · -- d = castAdd (M+N) iA, iA : Fin A
    dsimp only
    rw [castEquiv_symm_apply]
    apply Fin.ext
    simp only [combineIndex, splitLeft, Fin.val_cast]
    rw [show Fin.cast (Nat.add_assoc A M N).symm (Fin.castAdd (M + N) iA)
          = Fin.castAdd N (Fin.castAdd M iA) from by apply Fin.ext; simp]
    simp only [Fin.addCases_left, Fin.val_cast]
  · -- d = natAdd A iMN, iMN : Fin (M+N), further split into M / N
    dsimp only
    refine Fin.addCases (motive := fun iMN =>
        combineIndex (splitLeft ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
            (combineIndex (splitRight ((multiIndexEquiv (combinedDims dA dM)).symm ρ))
              ((multiIndexEquiv dN).symm jn)) (Fin.natAdd A iMN)
          = (castEquiv (Nat.add_assoc A M N).symm (fun i => hdA_assoc i)).symm
              (combineIndex ((multiIndexEquiv (combinedDims dA dM)).symm ρ)
                ((multiIndexEquiv dN).symm jn)) (Fin.natAdd A iMN))
      (fun iM => ?_) (fun iN => ?_) iMN
    · -- iM : Fin M
      dsimp only
      rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [combineIndex, splitLeft, splitRight, Fin.val_cast]
      rw [show Fin.cast (Nat.add_assoc A M N).symm (Fin.natAdd A (Fin.castAdd N iM))
            = Fin.castAdd N (Fin.natAdd A iM) from by apply Fin.ext; simp]
      simp only [Fin.addCases_left, Fin.addCases_right, Fin.val_cast]
    · -- iN : Fin N
      dsimp only
      rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [combineIndex, splitRight, Fin.val_cast]
      rw [show Fin.cast (Nat.add_assoc A M N).symm (Fin.natAdd A (Fin.natAdd M iN))
            = Fin.natAdd (A + M) iN from by apply Fin.ext; simp; omega]
      simp only [Fin.addCases_right, Fin.val_cast]

/-- The dim equality powering the cons-associativity regroup
    `(M+N)+1 → (M+1)+N`. -/
theorem hdR_consassoc (i : Fin ((M + 1) + N)) :
    combinedDims (Fin.cons R dM) dN i
      = (Fin.cons R (combinedDims dM dN) : Fin (M + N + 1) → ℕ)
          (Fin.cast (Nat.add_right_comm M 1 N) i) := by
  rw [combinedDims_cons]; simp only [Function.comp_apply]

/-- The cons-associativity regroup equiv `(M+N)+1 → (M+1)+N` at the
    multi-index level. -/
def consRegroup :
    ((d : Fin (M + N + 1)) → Fin ((Fin.cons R (combinedDims dM dN) : Fin (M + N + 1) → ℕ) d)) ≃
      ((d : Fin ((M + 1) + N)) → Fin (combinedDims (Fin.cons R dM) dN d)) :=
  castEquiv (Nat.add_right_comm M 1 N).symm (fun i => hdR_consassoc i)

/-- `deepSplitR` is `modeSplit` of the cons-associativity-regrouped tensor. -/
theorem deepSplitR_eq_modeSplit
    (V : Tensor 𝕜 (M + N + 1) (Fin.cons R (combinedDims dM dN))) :
    deepSplitR V = modeSplit (reindex (consRegroup (R := R) (dM := dM) (dN := dN)) V) := by
  funext σ jn
  rw [deepSplitR, Matrix.of_apply, modeSplit_apply, reindex_apply]
  congr 1
  rw [consRegroup]
  funext d
  refine Fin.cases ?_ (fun d' => ?_) d
  · -- head: d = 0
    rw [castEquiv_symm_apply]
    apply Fin.ext
    simp only [Fin.cons_zero, combineIndex, Fin.val_cast]
    rw [show Fin.cast (Nat.add_right_comm M 1 N).symm (0 : Fin (M + N + 1))
          = Fin.castAdd N (0 : Fin (M + 1)) from by apply Fin.ext; simp]
    simp only [Fin.addCases_left, Fin.val_cast]
  · -- tail: d = succ d', d' : Fin (M+N), further split M / N
    refine Fin.addCases (motive := fun d' =>
        Fin.cons (α := fun d => Fin ((Fin.cons R (combinedDims dM dN) :
              Fin (M + N + 1) → ℕ) d))
            ((multiIndexEquiv (Fin.cons R dM)).symm σ 0)
            (combineIndex (Fin.tail ((multiIndexEquiv (Fin.cons R dM)).symm σ))
              ((multiIndexEquiv dN).symm jn)) d'.succ
          = (castEquiv (Nat.add_right_comm M 1 N).symm (fun i => hdR_consassoc i)).symm
              (combineIndex ((multiIndexEquiv (Fin.cons R dM)).symm σ)
                ((multiIndexEquiv dN).symm jn)) d'.succ)
      (fun iM => ?_) (fun iN => ?_) d'
    · -- iM : Fin M
      dsimp only
      rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [Fin.cons_succ, Fin.tail, combineIndex, Fin.val_cast]
      rw [show Fin.cast (Nat.add_right_comm M 1 N).symm (Fin.succ (Fin.castAdd N iM))
            = Fin.castAdd N (Fin.succ iM) from by apply Fin.ext; simp]
      simp only [Fin.addCases_left, Fin.val_cast]
    · -- iN : Fin N
      dsimp only
      rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [Fin.cons_succ, combineIndex, Fin.val_cast]
      rw [show Fin.cast (Nat.add_right_comm M 1 N).symm (Fin.succ (Fin.natAdd M iN))
            = Fin.natAdd (M + 1) iN from by apply Fin.ext; simp; omega]
      simp only [Fin.addCases_right, Fin.val_cast]

end PerStepReshape

/-- `ctrRightFold` inverts `ctrRightMat`: every order-`(b+1)` `cons`-shaped
    tensor is the fold of its first-mode matricization. -/
theorem ctrRightFold_ctrRightMat {b : ℕ} {db : Fin b → ℕ} {S : ℕ}
    (B : Tensor 𝕜 (b + 1) (Fin.cons S db)) :
    ctrRightFold (ctrRightMat B) = B := by
  funext idx
  rw [ctrRightFold, ctrRightMat, Matrix.of_apply]
  congr 1
  funext d
  refine Fin.cases ?_ (fun i => ?_) d
  · simp only [Fin.cons_zero]; apply Fin.ext; simp
  · simp only [Fin.cons_succ, Equiv.symm_apply_apply]; apply Fin.ext; simp

/-! ### Bridging `modeSplitAt` to `modeSplit` of a regrouped tensor -/

/-- Regroup the modes of an arbitrary-shape tensor into the explicit
    `combinedDims` of its first-`p` and last-`q` restrictions (same order). -/
def splitRegroup {p q : ℕ} (dims : Fin (p + q) → ℕ) :
    ((d : Fin (p + q)) → Fin (dims d)) ≃
      ((d : Fin (p + q)) →
        Fin (combinedDims (fun i => dims (Fin.castAdd q i))
          (fun j => dims (Fin.natAdd p j)) d)) :=
  castEquiv rfl (fun i => by
    rw [combinedDims_split_eq]; simp only [Fin.cast_eq_self])

/-- `modeSplitAt X` equals `modeSplit` of the split-regrouped tensor. -/
theorem modeSplitAt_eq_modeSplit {p q : ℕ} {dims : Fin (p + q) → ℕ}
    (X : Tensor 𝕜 (p + q) dims) :
    modeSplitAt X = modeSplit (reindex (splitRegroup dims) X) := by
  funext ri cj
  rw [modeSplitAt_apply, modeSplit_apply, reindex_apply]
  congr 1
  rw [splitRegroup]
  funext d
  rw [castEquiv_symm_apply]
  apply Fin.ext
  refine Fin.addCases (motive := fun d =>
      (Fin.addCases (motive := fun d => Fin (dims d))
          ((multiIndexEquiv (fun i => dims (Fin.castAdd q i))).symm ri)
          ((multiIndexEquiv (fun j => dims (Fin.natAdd p j))).symm cj) d).val
        = (combineIndex
            ((multiIndexEquiv (fun i => dims (Fin.castAdd q i))).symm ri)
            ((multiIndexEquiv (fun j => dims (Fin.natAdd p j))).symm cj) (Fin.cast rfl d)).val)
    (fun i => ?_) (fun j => ?_) d
  · simp only [Fin.cast_eq_self, combineIndex, Fin.addCases_left, Fin.val_cast]
  · simp only [Fin.cast_eq_self, combineIndex, Fin.addCases_right, Fin.val_cast]

/-- `bestRankApproxErrorSq` of `modeSplitAt` equals that of `modeSplit` of the
    regrouped tensor (an immediate consequence; both are reshapes of the same
    data with the same row/column partition). -/
theorem bestRankApproxErrorSq_modeSplitAt {p q : ℕ} {dims : Fin (p + q) → ℕ}
    (X : Tensor 𝕜 (p + q) dims) (R : ℕ) :
    Matrix.bestRankApproxErrorSq (modeSplitAt X) R
      = Matrix.bestRankApproxErrorSq (modeSplit (reindex (splitRegroup dims) X)) R := by
  rw [modeSplitAt_eq_modeSplit]

/-- `modeSplitAt` respects subtraction. -/
theorem modeSplitAt_sub {p q : ℕ} {dims : Fin (p + q) → ℕ} (X Y : Tensor 𝕜 (p + q) dims) :
    modeSplitAt (X - Y) = modeSplitAt X - modeSplitAt Y := by
  funext ri cj; simp only [modeSplitAt_apply, Pi.sub_apply, Matrix.sub_apply]

/-- The general mode split preserves the squared Frobenius norm. -/
theorem frobNormSq_modeSplitAt {p q : ℕ} {dims : Fin (p + q) → ℕ}
    (X : Tensor 𝕜 (p + q) dims) :
    (modeSplitAt X).frobNormSq = ‖X‖_F² := by
  rw [modeSplitAt_eq_modeSplit, frobNormSq_modeSplit, frobNormSq_reindex]

/-- **Master alignment lemma.** Two `combinedDims`-shaped tensors whose entries
    agree under a *block-respecting* relabeling (a row-block equiv `er` and a
    column-block equiv `ec`) have mode splits that are submatrices of one
    another, hence equal `bestRankApproxErrorSq`. This funnels every flattening
    convention (`modeSplitAt`, `deepSplitL`, `deepSplitR`) into one comparison. -/
theorem bestRankApproxErrorSq_modeSplit_block
    {p q p' q' : ℕ} {I₁ : Fin p → ℕ} {I₂ : Fin q → ℕ} {J₁ : Fin p' → ℕ} {J₂ : Fin q' → ℕ}
    (Z : Tensor 𝕜 (p + q) (combinedDims I₁ I₂)) (Z' : Tensor 𝕜 (p' + q') (combinedDims J₁ J₂))
    (er : ((d : Fin p) → Fin (I₁ d)) ≃ ((d : Fin p') → Fin (J₁ d)))
    (ec : ((d : Fin q) → Fin (I₂ d)) ≃ ((d : Fin q') → Fin (J₂ d)))
    (h : ∀ f g, Z (combineIndex f g) = Z' (combineIndex (er f) (ec g)))
    (R : ℕ) :
    Matrix.bestRankApproxErrorSq (modeSplit Z) R
      = Matrix.bestRankApproxErrorSq (modeSplit Z') R := by
  have hsub : modeSplit Z
      = (modeSplit Z').submatrix
          ((multiIndexEquiv I₁).symm.trans (er.trans (multiIndexEquiv J₁)))
          ((multiIndexEquiv I₂).symm.trans (ec.trans (multiIndexEquiv J₂))) := by
    funext ri cj
    rw [Matrix.submatrix_apply, modeSplit_apply, modeSplit_apply, h]
    simp only [Equiv.trans_apply, Equiv.symm_apply_apply]
  rw [hsub, Matrix.bestRankApproxErrorSq_reindex]

/-- The canonical equivalence between multi-index sets over two pointwise-equal
    dimension functions (just `Fin.cast` componentwise; preserves every
    component's value). -/
def dimsCastEquiv {p : ℕ} {I J : Fin p → ℕ} (h : ∀ i, I i = J i) :
    ((d : Fin p) → Fin (I d)) ≃ ((d : Fin p) → Fin (J d)) where
  toFun f := fun d => Fin.cast (h d) (f d)
  invFun g := fun d => Fin.cast (h d).symm (g d)
  left_inv f := by funext d; apply Fin.ext; simp
  right_inv g := by funext d; apply Fin.ext; simp

@[simp]
theorem dimsCastEquiv_apply_val {p : ℕ} {I J : Fin p → ℕ} (h : ∀ i, I i = J i)
    (f : (d : Fin p) → Fin (I d)) (d : Fin p) :
    ((dimsCastEquiv h f) d).val = (f d).val := rfl

/-- The canonical multi-index equivalence between two index sets whose *arity*
    and *dimensions* agree up to `Fin.cast` (value-preserving componentwise). -/
def arityDimsCastEquiv {p p' : ℕ} {I : Fin p → ℕ} {J : Fin p' → ℕ} (hp : p = p')
    (h : ∀ i : Fin p', J i = I (Fin.cast hp.symm i)) :
    ((d : Fin p) → Fin (I d)) ≃ ((d : Fin p') → Fin (J d)) :=
  castEquiv hp h

theorem arityDimsCastEquiv_apply_val {p p' : ℕ} {I : Fin p → ℕ} {J : Fin p' → ℕ} (hp : p = p')
    (h : ∀ i : Fin p', J i = I (Fin.cast hp.symm i)) (f : (d : Fin p) → Fin (I d)) (i : Fin p') :
    ((arityDimsCastEquiv hp h f) i).val = (f (Fin.cast hp.symm i)).val := rfl

/-- **Block alignment for two reindexes of the same tensor.** If two
    `combinedDims`-grouped reindexes `reindex e₁ T`, `reindex e₂ T` place every
    component of every multi-index at the same value (a value-preserving
    relabeling), then their mode splits have equal `bestRankApproxErrorSq`. The
    workhorse for matching `kSplitErr`'s split convention to the per-step
    `deepSplit` convention: all relabelings here are componentwise `Fin.cast`s,
    so the hypothesis is discharged by `Fin.ext`. -/
theorem bestRankApproxErrorSq_reindex_block_aligned
    {D p q p' q' : ℕ} {dims : Fin D → ℕ}
    {I₁ : Fin p → ℕ} {I₂ : Fin q → ℕ} {J₁ : Fin p' → ℕ} {J₂ : Fin q' → ℕ}
    (T : Tensor 𝕜 D dims)
    (e₁ : ((d : Fin D) → Fin (dims d)) ≃ ((d : Fin (p + q)) → Fin (combinedDims I₁ I₂ d)))
    (e₂ : ((d : Fin D) → Fin (dims d)) ≃ ((d : Fin (p' + q')) → Fin (combinedDims J₁ J₂ d)))
    (er : ((d : Fin p) → Fin (I₁ d)) ≃ ((d : Fin p') → Fin (J₁ d)))
    (ec : ((d : Fin q) → Fin (I₂ d)) ≃ ((d : Fin q') → Fin (J₂ d)))
    (h : ∀ f g, e₁.symm (combineIndex f g) = e₂.symm (combineIndex (er f) (ec g)))
    (R : ℕ) :
    Matrix.bestRankApproxErrorSq (modeSplit (reindex e₁ T)) R
      = Matrix.bestRankApproxErrorSq (modeSplit (reindex e₂ T)) R := by
  refine bestRankApproxErrorSq_modeSplit_block (reindex e₁ T) (reindex e₂ T) er ec ?_ R
  intro f g
  rw [reindex_apply, reindex_apply, h]

/-- The mode-`(k+2)` split matrix of a leading-bond tensor
    `W : Tensor (n+2) (cons L out)` (rows: first `k+2` modes; columns: the
    rest), obtained via `modeSplitAt` after an order-cast. -/
noncomputable def kSplitMat {n : ℕ} {out : Fin (n + 1) → ℕ} {L : ℕ}
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (k : Fin (n + 1)) :=
  modeSplitAt (p := k.val + 2) (q := n - k.val)
    (reindex (castEquiv (show n + 2 = (k.val + 2) + (n - k.val) from by omega)
      (fun _ => rfl)) W)

/-- The mode-`(k+2)` split error of a leading-bond tensor `W : Tensor (n+2) (cons L out)`,
    at truncation rank `ranks k` — the `k`-th carried-residual split that the
    truncated TT-SVD truncates. -/
noncomputable def kSplitErr {n : ℕ} {out : Fin (n + 1) → ℕ} {L : ℕ}
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ) (k : Fin (n + 1)) : ℝ :=
  Matrix.bestRankApproxErrorSq (kSplitMat W k) (ranks k)

/-- `kSplitErr` written as `bestRankApproxErrorSq` of a `modeSplit` of `W`
    regrouped at the `(k+2)`-split: bridges the `modeSplitAt`-with-order-cast
    definition to the `modeSplit`-of-`combinedDims` form the per-step machinery
    consumes. -/
theorem kSplitErr_eq_modeSplit {n : ℕ} {out : Fin (n + 1) → ℕ} {L : ℕ}
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ) (k : Fin (n + 1)) :
    kSplitErr W ranks k
      = Matrix.bestRankApproxErrorSq
          (modeSplit (reindex
            ((castEquiv (show n + 2 = (k.val + 2) + (n - k.val) from by omega) (fun _ => rfl)).trans
              (splitRegroup (fun d => (Fin.cons L out : Fin (n + 2) → ℕ)
                (Fin.cast (show n + 2 = (k.val + 2) + (n - k.val) from by omega).symm d)))) W))
          (ranks k) := by
  rw [kSplitErr, kSplitMat, bestRankApproxErrorSq_modeSplitAt, reindex_comp]

/-- Scalar step for the `√(D-1)` quasi-optimality corollary: from `a² ≤ N·b²`
    (with `a, b ≥ 0`) conclude `a ≤ √N · b`. -/
theorem le_sqrt_natCast_mul_of_sq_le {a b : ℝ} {N : ℕ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (h : a ^ 2 ≤ (N : ℝ) * b ^ 2) : a ≤ Real.sqrt N * b :=
  calc a = Real.sqrt (a ^ 2) := (Real.sqrt_sq ha).symm
    _ ≤ Real.sqrt ((N : ℝ) * b ^ 2) := Real.sqrt_le_sqrt h
    _ = Real.sqrt N * b := by rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hb]

/-- **Peel alignment.** The mode-`2` split error `kSplitErr W ranks 0` equals
    the best rank-`(ranks 0)` error of the mode split of `W` reindexed into the
    peel grouping `combinedDims ![L, out 0] (out ∘ succ)` (the form consumed by
    the per-step truncation `exists_truncated_ctrProd`). -/
theorem kSplitErr_zero_eq_peel {n : ℕ} {out : Fin (n + 1) → ℕ} {L : ℕ}
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ) :
    kSplitErr W ranks 0
      = Matrix.bestRankApproxErrorSq
          (modeSplit (reindex
            (castEquiv (Nat.add_comm 2 n) (fun i => chainPeel_hd L out i)).symm W)) (ranks 0) := by
  rw [kSplitErr_eq_modeSplit]
  have hrow : ∀ i : Fin 2, (fun d => (Fin.cons L out : Fin (n + 2) → ℕ)
        (Fin.cast (show n + 2 = (0 + 2) + (n - 0) from by omega).symm (Fin.castAdd (n - 0) d))) i
      = (![L, out 0] : Fin 2 → ℕ) i := by
    intro i
    refine Fin.cases ?_ (fun i' => ?_) i
    · rfl
    · refine Fin.cases ?_ (fun j => j.elim0) i'
      rfl
  have hcol : ∀ j : Fin n, (fun d => (Fin.cons L out : Fin (n + 2) → ℕ)
        (Fin.cast (show n + 2 = (0 + 2) + (n - 0) from by omega).symm (Fin.natAdd (0 + 2) d))) j
      = (fun k : Fin n => out k.succ) j := by
    intro j
    change (Fin.cons L out : Fin (n + 2) → ℕ) _ = out j.succ
    rw [show Fin.cast (show n + 2 = (0 + 2) + (n - 0) from by omega).symm (Fin.natAdd (0 + 2) j)
          = Fin.succ j.succ from by apply Fin.ext; simp, Fin.cons_succ]
  refine bestRankApproxErrorSq_reindex_block_aligned W _ _
    (dimsCastEquiv hrow) (dimsCastEquiv hcol) (fun f g => ?_) (ranks 0)
  funext d
  change ((castEquiv (show n + 2 = (0 + 2) + (n - 0) from by omega) (fun _ => rfl)).trans
        (splitRegroup _)).symm (combineIndex f g) d
      = (castEquiv (Nat.add_comm 2 n) (fun i => chainPeel_hd L out i)).symm.symm
        (combineIndex (dimsCastEquiv hrow f) (dimsCastEquiv hcol g)) d
  rw [Equiv.symm_trans_apply, Equiv.symm_symm]
  apply Fin.ext
  simp only [castEquiv, splitRegroup, Equiv.coe_fn_mk, Equiv.coe_fn_symm_mk,
    combineIndex, dimsCastEquiv, Fin.addCases, Fin.val_cast]
  by_cases h : (d : ℕ) < 2
  · rw [dif_pos (by simpa using h), dif_pos (by simpa using h)]
    simp only [Fin.val_cast]
    exact congrArg (fun a => (f a).val) (Fin.ext (by simp))
  · rw [dif_neg (by simpa using h), dif_neg (by simpa using h)]
    rw [eqRec_eq_cast, eqRec_eq_cast]
    have hcastval : ∀ {x y : ℕ} (heq : Fin x = Fin y) (z : Fin x), (cast heq z).val = z.val := by
      intro x y heq z
      have : x = y := by have := Fintype.card_eq.mpr ⟨(Equiv.cast heq)⟩; simpa using this
      subst this; rfl
    rw [hcastval, hcastval]
    simp only [Fin.val_cast]
    exact congrArg (fun a => (g a).val) (Fin.ext (by simp))

/-! ### Carried-residual threading via the deep split -/

section Threading

variable {n : ℕ} {out : Fin (n + 2) → ℕ} {L : ℕ} {S : ℕ}

/-- The first `k+1` of the trailing output modes `out ∘ succ`, as `dM`. -/
def threadDM (out : Fin (n + 2) → ℕ) (k : Fin (n + 1)) : Fin (k.val + 1) → ℕ :=
  fun i => out (Fin.succ (Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega)
    (Fin.castAdd (n - k.val) i)))

/-- The last `n−k` of the trailing output modes `out ∘ succ`, as `dN`. -/
def threadDN (out : Fin (n + 2) → ℕ) (k : Fin (n + 1)) : Fin (n - k.val) → ℕ :=
  fun j => out (Fin.succ (Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega)
    (Fin.natAdd (k.val + 1) j)))

/-- The trailing output modes split as `combinedDims (threadDM) (threadDN)`
    (the deep grouping the per-step reshape consumes). -/
theorem threadDM_DN_eq (k : Fin (n + 1)) (i : Fin ((k.val + 1) + (n - k.val))) :
    combinedDims (threadDM out k) (threadDN out k) i
      = (fun j : Fin (n + 1) => out j.succ)
          (Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega) i) := by
  refine Fin.addCases (motive := fun i =>
    combinedDims (threadDM out k) (threadDN out k) i
      = (fun j : Fin (n + 1) => out j.succ)
          (Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega) i))
    (fun iM => ?_) (fun iN => ?_) i
  · dsimp only; rw [combinedDims_castAdd, threadDM]
  · dsimp only; rw [combinedDims_natAdd, threadDN]

/-- The deep regroup of `W` (order `n+3`) into
    `combinedDims ![L,out 0] (combinedDims (threadDM) (threadDN))`. -/
def threadEX (W : Tensor 𝕜 (n + 3) (Fin.cons L out)) (k : Fin (n + 1)) :
    Tensor 𝕜 (2 + ((k.val + 1) + (n - k.val)))
      (combinedDims (![L, out 0] : Fin 2 → ℕ) (combinedDims (threadDM out k) (threadDN out k))) :=
  reindex (castEquiv (show n + 3 = 2 + ((k.val + 1) + (n - k.val)) from by omega)
    (fun i => by
      refine Fin.addCases (motive := fun i =>
        combinedDims (![L, out 0] : Fin 2 → ℕ) (combinedDims (threadDM out k) (threadDN out k)) i
          = (Fin.cons L out : Fin (n + 3) → ℕ)
              (Fin.cast (show n + 3 = 2 + ((k.val + 1) + (n - k.val)) from by omega).symm i))
        (fun i2 => ?_) (fun iMN => ?_) i
      · dsimp only
        rw [combinedDims_castAdd]
        refine Fin.cases ?_ (fun i1 => ?_) i2
        · change L = _
          rw [show Fin.cast (show n + 3 = 2 + ((k.val + 1) + (n - k.val)) from by omega).symm
                (Fin.castAdd ((k.val + 1) + (n - k.val)) (0 : Fin 2)) = (0 : Fin (n + 3)) from by
              apply Fin.ext; simp, Fin.cons_zero]
        · refine Fin.cases ?_ (fun j => j.elim0) i1
          change out 0 = _
          rw [show Fin.cast (show n + 3 = 2 + ((k.val + 1) + (n - k.val)) from by omega).symm
                (Fin.castAdd ((k.val + 1) + (n - k.val)) (Fin.succ (0 : Fin 1)))
                = Fin.succ (0 : Fin (n + 2)) from by
              apply Fin.ext; simp, Fin.cons_succ]
      · dsimp only
        rw [combinedDims_natAdd, threadDM_DN_eq]
        show (fun j : Fin (n + 1) => out j.succ) _ = _
        rw [show Fin.cast (show n + 3 = 2 + ((k.val + 1) + (n - k.val)) from by omega).symm
                (Fin.natAdd 2 iMN)
              = Fin.succ ((Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega)
                  iMN).succ)
            from by apply Fin.ext; simp; omega, Fin.cons_succ])) W

/-- **Deep-split alignment for `W`.** The `(k+1).succ`-split error of `W`
    equals the best rank-`(ranks k.succ)` error of the deep split `deepSplitL`
    of the deep regroup `threadEX W k`. (Both split `W` at physical position
    `k+3`; they differ only by the flattening convention, handled by the
    block-aligned reindex invariance.) -/
theorem kSplitErr_succ_eq_deepSplitL (W : Tensor 𝕜 (n + 3) (Fin.cons L out))
    (ranks : Fin (n + 2) → ℕ) (k : Fin (n + 1)) :
    kSplitErr W ranks k.succ
      = Matrix.bestRankApproxErrorSq (deepSplitL (threadEX W k)) (ranks k.succ) := by
  rw [kSplitErr_eq_modeSplit, deepSplitL_eq_modeSplit, threadEX, reindex_comp]
  have consEval : ∀ (P : Fin (n + 3)) (Q : Fin (n + 2)), P.val = Q.val + 1 →
      (Fin.cons L out : Fin (n + 3) → ℕ) P = out Q := by
    intro P Q hPQ
    rw [show P = Q.succ from Fin.ext (by simp [hPQ]), Fin.cons_succ]
  have consEval0 : ∀ (P : Fin (n + 3)), P.val = 0 →
      (Fin.cons L out : Fin (n + 3) → ℕ) P = L := by
    intro P hP
    rw [show P = 0 from Fin.ext (by simp [hP]), Fin.cons_zero]
  have consEval1 : ∀ (P : Fin (n + 3)), P.val = 1 →
      (Fin.cons L out : Fin (n + 3) → ℕ) P = out 0 := by
    intro P hP
    rw [consEval P 0 (by simp [hP])]
  refine bestRankApproxErrorSq_reindex_block_aligned W _ _
    (arityDimsCastEquiv (J := combinedDims (![L, out 0] : Fin 2 → ℕ) (threadDM out k))
      (show k.val.succ + 2 = 2 + (k.val + 1) from by omega) (fun i => ?_))
    (arityDimsCastEquiv (J := threadDN out k)
      (show (n + 1) - k.val.succ = n - k.val from by omega) (fun j => ?_)) (fun f g => ?_) _
  · -- row dims agree
    have hk : k.val ≤ n := Nat.lt_succ_iff.mp k.isLt
    have hks : k.succ.val = k.val + 1 := Fin.val_succ k
    refine Fin.addCases (motive := fun i =>
        combinedDims (![L, out 0] : Fin 2 → ℕ) (threadDM out k) i
        = (Fin.cons L out : Fin (n + 3) → ℕ) (Fin.cast _ (Fin.castAdd (n + 1 - k.succ.val)
            (Fin.cast _ i)))) (fun i2 => ?_) (fun iM => ?_) i
    · dsimp only
      rw [combinedDims_castAdd]
      refine Fin.cases ?_ (fun i1 => ?_) i2
      · rw [show (Fin.cast (by omega) (Fin.castAdd (n + 1 - k.succ.val) (Fin.cast (by omega)
              (Fin.castAdd (k.val + 1) (0 : Fin 2))))) = (0 : Fin (n + 3)) from by
            apply Fin.ext; simp, Fin.cons_zero]
        simp
      · refine Fin.cases ?_ (fun j => j.elim0) i1
        rw [show (Fin.cast (by omega) (Fin.castAdd (n + 1 - k.succ.val) (Fin.cast (by omega)
              (Fin.castAdd (k.val + 1) (Fin.succ (0 : Fin 1)))))) = Fin.succ (0 : Fin (n + 2))
            from by apply Fin.ext; simp, Fin.cons_succ]
        simp
    · dsimp only
      rw [combinedDims_natAdd, threadDM]
      rw [show (Fin.cast (by omega) (Fin.castAdd (n + 1 - k.succ.val) (Fin.cast (by omega)
            (Fin.natAdd 2 iM))))
            = Fin.succ (Fin.cast (show (k.val + 1) + (n - k.val) = n + 1 from by omega)
                (Fin.castAdd (n - k.val) iM)).succ
          from by apply Fin.ext; simp only [Fin.val_succ, Fin.val_cast, Fin.val_natAdd,
            Fin.val_castAdd]; omega, Fin.cons_succ]
  · -- col dims agree
    have hk : k.val ≤ n := Nat.lt_succ_iff.mp k.isLt
    have hks : k.succ.val = k.val + 1 := Fin.val_succ k
    rw [threadDN]
    refine (consEval _ _ ?_).symm
    simp only [Fin.val_succ, Fin.val_cast, Fin.val_natAdd]; omega
  · -- entrywise W-index equality: both relabelings place the value of `d`'s
    -- region (rows 0..1 = ![L,out0], 2..k+2 = dM, k+3.. = dN) identically.
    have hk : k.val ≤ n := Nat.lt_succ_iff.mp k.isLt
    have hcv : ∀ {x y : ℕ} (heq : Fin x = Fin y) (z : Fin x), (cast heq z).val = z.val := by
      intro x y heq z
      have : x = y := by have := Fintype.card_eq.mpr ⟨(Equiv.cast heq)⟩; simpa using this
      subst this; rfl
    funext d
    apply Fin.ext
    simp only [splitRegroup, assocRegroup, arityDimsCastEquiv, castEquiv, Equiv.trans, Equiv.symm,
      Equiv.coe_fn_mk, combineIndex, Function.comp_apply,
      Fin.val_cast, Fin.addCases]
    by_cases h3 : (d : ℕ) < k.val + 3
    · rw [dif_pos (show (d : ℕ) < (k.val).succ + 2 from by omega),
        dif_pos (show (d : ℕ) < Nat.succ 1 + (k.val + 1) from by omega)]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (f a).val) (Fin.ext (by simp))
    · rw [dif_neg (show ¬ (d : ℕ) < (k.val).succ + 2 from by omega),
        dif_neg (show ¬ (d : ℕ) < Nat.succ 1 + (k.val + 1) from by omega)]
      rw [eqRec_eq_cast, eqRec_eq_cast, hcv, hcv]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (g a).val)
        (Fin.ext (by simp only [Fin.val_subNat, Fin.val_cast]; omega))

/-- The deep regroup of the carried residual `V` (order `n+2`,
    `cons S (out ∘ succ)`) into `cons S (combinedDims (threadDM) (threadDN))`. -/
def threadEV (V : Tensor 𝕜 (n + 1 + 1) (Fin.cons S (fun k => out k.succ))) (k : Fin (n + 1)) :
    Tensor 𝕜 (((k.val + 1) + (n - k.val)) + 1)
      (Fin.cons S (combinedDims (threadDM out k) (threadDN out k))) :=
  reindex (castEquiv (show n + 1 + 1 = ((k.val + 1) + (n - k.val)) + 1 from by omega)
    (fun i => by
      refine Fin.cases ?_ (fun i' => ?_) i
      · change S = _
        rw [show Fin.cast (show n + 1 + 1 = ((k.val + 1) + (n - k.val)) + 1 from by omega).symm
              (0 : Fin (((k.val + 1) + (n - k.val)) + 1)) = (0 : Fin (n + 1 + 1)) from by
            apply Fin.ext; simp, Fin.cons_zero]
      · rw [Fin.cons_succ, threadDM_DN_eq]
        show (fun j : Fin (n + 1) => out j.succ) _ = _
        rw [show Fin.cast (show n + 1 + 1 = ((k.val + 1) + (n - k.val)) + 1 from by omega).symm
              (Fin.succ i') = Fin.succ (Fin.cast
                (show (k.val + 1) + (n - k.val) = n + 1 from by omega) i') from by
            apply Fin.ext; simp, Fin.cons_succ])) V

set_option maxHeartbeats 2000000 in
-- The threaded deep regroup builds a large dependent `Fin`-cast index term, and
-- the block-aligned reindex comparison exceeds the default heartbeat budget.
/-- **Deep-split alignment for the carried residual `V`.** The mode-`(k+2)`
    split error of `V` equals the best rank error of `deepSplitR` of its deep
    regroup `threadEV V k`. (Both split `V` at physical position `k+2`.) -/
theorem kSplitErr_eq_deepSplitR (V : Tensor 𝕜 (n + 1 + 1) (Fin.cons S (fun k => out k.succ)))
    (ranks : Fin (n + 2) → ℕ) (k : Fin (n + 1)) :
    kSplitErr V (fun j => ranks j.succ) k
      = Matrix.bestRankApproxErrorSq (deepSplitR (threadEV V k)) (ranks k.succ) := by
  rw [kSplitErr_eq_modeSplit, deepSplitR_eq_modeSplit, threadEV, reindex_comp]
  have hk : k.val ≤ n := Nat.lt_succ_iff.mp k.isLt
  have hcv : ∀ {x y : ℕ} (heq : Fin x = Fin y) (z : Fin x), (cast heq z).val = z.val := by
    intro x y heq z
    have : x = y := by have := Fintype.card_eq.mpr ⟨(Equiv.cast heq)⟩; simpa using this
    subst this; rfl
  have consEval : ∀ (P : Fin (n + 2)) (Q : Fin (n + 1)), P.val = Q.val + 1 →
      (Fin.cons S (fun k => out k.succ) : Fin (n + 2) → ℕ) P = (fun k => out k.succ) Q := by
    intro P Q hPQ
    rw [show P = Q.succ from Fin.ext (by simp [hPQ]), Fin.cons_succ]
  refine bestRankApproxErrorSq_reindex_block_aligned V _ _
    (arityDimsCastEquiv (J := Fin.cons S (threadDM out k))
      (show k.val + 2 = (k.val + 1) + 1 from by omega) (fun i => ?_))
    (arityDimsCastEquiv (J := threadDN out k)
      (show n - k.val = n - k.val from rfl) (fun j => ?_)) (fun f g => ?_) _
  · -- row dims agree (cons-shaped block)
    refine Fin.cases ?_ (fun i' => ?_) i
    · rfl
    · rw [Fin.cons_succ, threadDM]
      refine (consEval _ _ ?_).symm
      simp only [Fin.val_succ, Fin.val_cast, Fin.val_castAdd]
  · -- col dims agree
    rw [threadDN]
    refine (consEval _ _ ?_).symm
    simp only [Fin.val_cast, Fin.val_natAdd]; omega
  · -- entrywise V-index equality
    funext d
    apply Fin.ext
    simp only [splitRegroup, consRegroup, arityDimsCastEquiv, castEquiv, Equiv.trans, Equiv.symm,
      Equiv.coe_fn_mk, combineIndex, Function.comp_apply,
      Fin.val_cast, Fin.addCases]
    by_cases h2 : (d : ℕ) < k.val + 2
    · rw [dif_pos (show (d : ℕ) < (k.val).succ + 1 from by omega),
        dif_pos (show (d : ℕ) < (k.val + 1) + 1 from by omega)]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (f a).val) (Fin.ext (by simp))
    · rw [dif_neg (show ¬ (d : ℕ) < (k.val).succ + 1 from by omega),
        dif_neg (show ¬ (d : ℕ) < (k.val + 1) + 1 from by omega)]
      rw [eqRec_eq_cast, eqRec_eq_cast, hcv, hcv]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (g a).val) (Fin.ext (by simp only [Fin.val_subNat, Fin.val_cast]))

/-- Forward evaluation of `castEquiv` (companion to `castEquiv_symm_apply`). -/
theorem castEquiv_apply {D D' : ℕ} {dims : Fin D → ℕ} {dims' : Fin D' → ℕ}
    (hD : D = D') (hd : ∀ i : Fin D', dims' i = dims (Fin.cast hD.symm i))
    (f : (d : Fin D) → Fin (dims d)) (i : Fin D') :
    castEquiv hD hd f i = Fin.cast (hd i).symm (f (Fin.cast hD.symm i)) := rfl

/-- **Carried-residual reshape, transported to the threaded grouping.** The
    deep regroup `threadEV V k` of the carried residual is the orthonormal
    compression `(ctrLeftMat A)ᴴ` of the deep regroup `threadEX W k` of `W`.
    This carries the per-step relation `hVrel` (stated in the peel grouping
    `reindex E.symm W`) over to the threaded grouping: the trailing output modes
    are relabeled by the value-preserving cast `g := arityDimsCastEquiv` — the
    same regroup (`threadDM_DN_eq`) that defines `threadEV`/`threadEX`. -/
theorem ctrRightMat_threadEV_eq
    (W : Tensor 𝕜 (n + 3) (Fin.cons L out))
    (A : Tensor 𝕜 (2 + 1) (Fin.snoc (![L, out 0] : Fin 2 → ℕ) S))
    (V : Tensor 𝕜 (n + 1 + 1) (Fin.cons S (fun k => out k.succ)))
    (hVrel : ctrRightMat V = (ctrLeftMat A)ᴴ *
      modeSplit (reindex (castEquiv (Nat.add_comm 2 (n + 1))
        (fun i => chainPeel_hd L out i)).symm W)) (k : Fin (n + 1)) :
    ctrRightMat (threadEV V k) = (ctrLeftMat A)ᴴ * modeSplit (threadEX W k) := by
  have hp : (k.val + 1) + (n - k.val) = n + 1 := by omega
  have hd_g : ∀ i : Fin (n + 1), (fun j => out j.succ) i
      = combinedDims (threadDM out k) (threadDN out k) (Fin.cast hp.symm i) :=
    fun i => (threadDM_DN_eq k (Fin.cast hp.symm i)).symm
  set g := arityDimsCastEquiv hp hd_g with hg
  have hg_val : ∀ (μ : (d : Fin ((k.val + 1) + (n - k.val))) →
        Fin (combinedDims (threadDM out k) (threadDN out k) d)) (i : Fin (n + 1)),
      ((g μ) i).val = (μ (Fin.cast hp.symm i)).val := by
    intro μ i; rw [hg]; exact arityDimsCastEquiv_apply_val hp hd_g μ i
  have hcv : ∀ {x y : ℕ} (heq : Fin x = Fin y) (z : Fin x), (cast heq z).val = z.val := by
    intro x y heq z
    have : x = y := by have := Fintype.card_eq.mpr ⟨(Equiv.cast heq)⟩; simpa using this
    subst this; rfl
  -- (i): `ctrRightMat (threadEV V k)` is `ctrRightMat V` with columns regrouped by `g`.
  have hi : ∀ (s : Fin S)
      (cj : Fin (∏ d, combinedDims (threadDM out k) (threadDN out k) d)),
      ctrRightMat (threadEV V k) s cj
        = ctrRightMat V s (multiIndexEquiv (fun j => out j.succ)
            (g ((multiIndexEquiv (combinedDims (threadDM out k)
              (threadDN out k))).symm cj))) := by
    intro s cj
    rw [ctrRightMat, ctrRightMat, Matrix.of_apply, Matrix.of_apply, threadEV, reindex_apply]
    simp only [Equiv.symm_apply_apply]
    congr 1
    funext d
    refine Fin.cases ?_ (fun d' => ?_) d
    · rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [Fin.val_cast]
      conv_lhs => rw [show Fin.cast (show n + 1 + 1 = ((k.val + 1) + (n - k.val)) + 1 from by omega)
            (0 : Fin (n + 1 + 1)) = (0 : Fin (((k.val + 1) + (n - k.val)) + 1))
          from by apply Fin.ext; simp]
      simp [Fin.cons_zero]
    · rw [castEquiv_symm_apply]
      apply Fin.ext
      simp only [Fin.val_cast]
      conv_lhs => rw [show Fin.cast (show n + 1 + 1 = ((k.val + 1) + (n - k.val)) + 1 from by omega)
            (Fin.succ d') = Fin.succ (Fin.cast hp.symm d')
          from by apply Fin.ext; simp]
      simp [Fin.cons_succ, hg_val]
  -- (ii): the W-side mode split matches under the same column regrouping `g`.
  have hii : ∀ (ρ : Fin (∏ d, (![L, out 0] : Fin 2 → ℕ) d))
      (cj : Fin (∏ d, combinedDims (threadDM out k) (threadDN out k) d)),
      modeSplit (reindex (castEquiv (Nat.add_comm 2 (n + 1))
          (fun i => chainPeel_hd L out i)).symm W) ρ
        (multiIndexEquiv (fun j => out j.succ)
          (g ((multiIndexEquiv (combinedDims (threadDM out k)
            (threadDN out k))).symm cj)))
        = modeSplit (threadEX W k) ρ cj := by
    intro ρ cj
    rw [modeSplit_apply, modeSplit_apply, reindex_apply, threadEX, reindex_apply]
    simp only [Equiv.symm_symm, Equiv.symm_apply_apply]
    congr 1
    funext d
    apply Fin.ext
    rw [castEquiv_symm_apply, castEquiv_apply]
    simp only [Fin.val_cast, combineIndex, Fin.addCases]
    by_cases hlt : (d : ℕ) < 2
    · rw [dif_pos (show (d : ℕ) < 2 from hlt), dif_pos (show (d : ℕ) < 2 from hlt)]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (((multiIndexEquiv (![L, out 0] : Fin 2 → ℕ)).symm ρ) a).val)
        (Fin.ext (by simp))
    · rw [dif_neg (show ¬ (d : ℕ) < 2 from hlt), dif_neg (show ¬ (d : ℕ) < 2 from hlt)]
      rw [eqRec_eq_cast, eqRec_eq_cast, hcv, hcv]
      simp only [Fin.val_cast]
      rw [hg_val]
      exact congrArg (fun a => (((multiIndexEquiv (combinedDims (threadDM out k)
        (threadDN out k))).symm cj) a).val)
        (Fin.ext (by simp only [Fin.val_subNat, Fin.val_cast]))
  -- assemble: rewrite columns by (i), apply `hVrel`, then match summands by (ii).
  ext s cj
  rw [hi s cj, hVrel, Matrix.mul_apply, Matrix.mul_apply]
  exact Finset.sum_congr rfl fun ρ _ => by rw [hii ρ cj]

/-- **Carried-residual threading.** Each deeper split of the carried residual
    `V` (from peeling `W`'s first core at position 2) has best-rank error at
    most the next split of `W`. The induction-hypothesis feeder of the error
    bound: combines the per-step reshape inequality
    `deepSplit_bestRankApproxErrorSq_le` with the `kSplitErr ↔ deepSplit`
    alignment. -/
theorem kSplitErr_carried_le {n : ℕ} {out : Fin (n + 2) → ℕ} {L : ℕ} {S : ℕ}
    (W : Tensor 𝕜 (n + 3) (Fin.cons L out)) (ranks : Fin (n + 2) → ℕ)
    (A : Tensor 𝕜 (2 + 1) (Fin.snoc (![L, out 0] : Fin 2 → ℕ) S))
    (V : Tensor 𝕜 (n + 1 + 1) (Fin.cons S (fun k => out k.succ)))
    (hAorth : (ctrLeftMat A)ᴴ * ctrLeftMat A = 1)
    (hVrel : ctrRightMat V = (ctrLeftMat A)ᴴ *
      modeSplit (reindex (castEquiv (Nat.add_comm 2 (n + 1))
        (fun i => chainPeel_hd L out i)).symm W)) (k : Fin (n + 1)) :
    kSplitErr V (fun j => ranks j.succ) k ≤ kSplitErr W ranks k.succ := by
  rw [kSplitErr_eq_deepSplitR V ranks k, kSplitErr_succ_eq_deepSplitL W ranks k]
  by_cases hR' : ranks k.succ
      ≤ ∏ d, (Fin.cons S (threadDM out k) : Fin ((k.val + 1) + 1) → ℕ) d
  · have hGOAL : threadEV V k
        = ctrRightFold ((ctrLeftMat A)ᴴ * modeSplit (threadEX W k)) := by
      rw [← ctrRightMat_threadEV_eq W A V hVrel k, ctrRightFold_ctrRightMat]
    rw [hGOAL]
    exact deepSplit_bestRankApproxErrorSq_le (threadEX W k) (ctrLeftMat A) hAorth
      (ranks k.succ) hR'
  · rw [show Matrix.bestRankApproxErrorSq (deepSplitR (threadEV V k)) (ranks k.succ) = 0
        from by rw [Matrix.bestRankApproxErrorSq, dif_neg hR']]
    exact Matrix.bestRankApproxErrorSq_nonneg _ _

end Threading

/-! ### The truncated TT-SVD output as an explicit witness -/

/-- **The truncated TT-SVD output** of a leading-bond tensor `W` at target ranks
    `ranks`, as an explicit function (not merely an existential). It peels the
    first core off the mode-`2` split with the best-rank-`(min (ranks 0) …)`
    truncation (`truncLeft`/`truncRight`), recurses on the carried residual, and
    reassembles through the chain-peel reindex — exactly the construction the
    error bound's induction performs. The leading bond is exposed so the
    recursion can carry the bond produced by each truncation. -/
noncomputable def ttApproxOpen : (n : ℕ) → (out : Fin (n + 1) → ℕ) → (L : ℕ) →
    Tensor 𝕜 (n + 2) (Fin.cons L out) → (Fin (n + 1) → ℕ) → Tensor 𝕜 (n + 2) (Fin.cons L out)
  | 0, out, L, W, ranks =>
      let E := castEquiv (Nat.add_comm 2 0) (fun i => chainPeel_hd L out i)
      let R0 := min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)
      reindex E (truncLeft (reindex E.symm W) R0 (min_le_right _ _)
               ⋈ₜ truncRight (reindex E.symm W) R0 (min_le_right _ _))
  | n + 1, out, L, W, ranks =>
      let E := castEquiv (Nat.add_comm 2 (n + 1)) (fun i => chainPeel_hd L out i)
      let R0 := min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)
      reindex E (truncLeft (reindex E.symm W) R0 (min_le_right _ _)
               ⋈ₜ ttApproxOpen n (fun k => out k.succ) R0
                    (truncRight (reindex E.symm W) R0 (min_le_right _ _))
                    (fun j => ranks j.succ))

/-- **`Y` is a truncated TT-SVD of the leading-bond tensor `W` at ranks `ranks`.**
    The predicate capturing *every* valid output of the left-to-right truncated
    sweep: at each step the left factor `A` is an orthonormal-column best rank-`R₀`
    truncation of the current mode-`2` split (`R₀ = min (ranks 0) (∏ …)`), the
    carried residual `V` is the exact remainder, and the rest of `Y` is a truncated
    TT-SVD of `V`. The only freedom is the (non-unique) per-step best-rank
    projector — the canonical choice is `ttApproxOpen`. Mirrors
    `IsTruncatedHOSVD` for the TT setting (here the sweep order is fixed by the
    train topology, so there is no analogue of HOSVD's mode permutation `σ`). -/
def IsTruncatedTTSVDOpen : (n : ℕ) → (out : Fin (n + 1) → ℕ) → (L : ℕ) →
    Tensor 𝕜 (n + 2) (Fin.cons L out) → (Fin (n + 1) → ℕ) →
    Tensor 𝕜 (n + 2) (Fin.cons L out) → Prop
  | 0, out, L, W, ranks, Y =>
      let E := castEquiv (Nat.add_comm 2 0) (fun i => chainPeel_hd L out i)
      let R0 := min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)
      ∃ (A : Tensor 𝕜 (2 + 1) (Fin.snoc (![L, out 0] : Fin 2 → ℕ) R0))
        (V : Tensor 𝕜 (0 + 1) (Fin.cons R0 (fun k => out k.succ))),
        (ctrLeftMat A)ᴴ * ctrLeftMat A = 1 ∧
        ctrRightMat V = (ctrLeftMat A)ᴴ * modeSplit (reindex E.symm W) ∧
        ‖reindex E.symm W - A ⋈ₜ V‖_F²
          = Matrix.bestRankApproxErrorSq (modeSplit (reindex E.symm W)) R0 ∧
        Y = reindex E (A ⋈ₜ V)
  | n + 1, out, L, W, ranks, Y =>
      let E := castEquiv (Nat.add_comm 2 (n + 1)) (fun i => chainPeel_hd L out i)
      let R0 := min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)
      ∃ (A : Tensor 𝕜 (2 + 1) (Fin.snoc (![L, out 0] : Fin 2 → ℕ) R0))
        (V : Tensor 𝕜 (n + 1 + 1) (Fin.cons R0 (fun k => out k.succ)))
        (Y' : Tensor 𝕜 (n + 1 + 1) (Fin.cons R0 (fun k => out k.succ))),
        (ctrLeftMat A)ᴴ * ctrLeftMat A = 1 ∧
        ctrRightMat V = (ctrLeftMat A)ᴴ * modeSplit (reindex E.symm W) ∧
        ‖reindex E.symm W - A ⋈ₜ V‖_F²
          = Matrix.bestRankApproxErrorSq (modeSplit (reindex E.symm W)) R0 ∧
        IsTruncatedTTSVDOpen n (fun k => out k.succ) R0 V (fun j => ranks j.succ) Y' ∧
        Y = reindex E (A ⋈ₜ Y')

/-! ### The full per-split error bound (`thm:tt-svd-error`) and the uniform form -/

set_option maxHeartbeats 1000000 in
-- The induction recurses through dependent `Fin.cons`/`combinedDims` reshapes
-- whose defeq checking (the `∑ kSplitErr` step) exceeds the default budget.
/-- **Truncated TT-SVD error bound (`thm:tt-svd-error`, `eq:tt-svd-error`),
    predicate form.** *Every* truncated TT-SVD `Y` of the leading-bond tensor `W`
    satisfies the exact per-split sum bound

      `‖W − Y‖² ≤ ∑ₖ bestRankApproxErrorSq (W's mode-(k+2) split) (ranks k)`,

    i.e. `‖X − X̂‖ ≤ √(∑ₙ ‖X_{[n]} − [X_{[n]}]_{Rₙ}‖²)` after taking square roots.
    Proof by induction on the number of output modes, reading the per-step facts
    off the predicate `IsTruncatedTTSVDOpen`: the orthonormal best-rank step
    factor and exact remainder give the step error (`truncStep_err`-conditioned),
    added Pythagoreanly (`tt_truncation_step_error`), with the carried residual's
    deeper split errors dominated by the originals (`kSplitErr_carried_le`); the
    accumulated sum telescopes into `∑ₖ kSplitErr W ranks k`. -/
theorem isTruncatedTTSVDOpen_error_bound : ∀ {n : ℕ} (out : Fin (n + 1) → ℕ) (L : ℕ)
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ)
    (Y : Tensor 𝕜 (n + 2) (Fin.cons L out)),
      IsTruncatedTTSVDOpen n out L W ranks Y → ‖W - Y‖_F² ≤ ∑ k, kSplitErr W ranks k := by
  intro n
  induction n with
  | zero =>
    intro out L W ranks Y hY
    obtain ⟨A, V, hAorth, hVrel, herr, hYeq⟩ := hY
    set E := castEquiv (Nat.add_comm 2 0) (fun i => chainPeel_hd L out i) with hEdef
    rw [hYeq]
    refine le_of_eq ?_
    have hround : reindex E.symm W - A ⋈ₜ V = reindex E.symm (W - reindex E (A ⋈ₜ V)) := by
      rw [reindex_sub, reindex_comp, Equiv.self_trans_symm, reindex_id]
    have hnorm : ‖W - reindex E (A ⋈ₜ V)‖_F² = ‖reindex E.symm W - A ⋈ₜ V‖_F² := by
      rw [hround, frobNormSq_reindex]
    rw [hnorm, herr, Matrix.bestRankApproxErrorSq_min, ← kSplitErr_zero_eq_peel,
      Fin.sum_univ_one]
  | succ n ih =>
    intro out L W ranks Y hY
    obtain ⟨A, V, Y', hAorth, hVrel, herr, hrec, hYeq⟩ := hY
    set E := castEquiv (Nat.add_comm 2 (n + 1)) (fun i => chainPeel_hd L out i) with hEdef
    have hthread : ∀ k : Fin (n + 1),
        kSplitErr V (fun j => ranks j.succ) k ≤ kSplitErr W ranks k.succ :=
      kSplitErr_carried_le W ranks A V hAorth hVrel
    have hVhat : ‖V - Y'‖_F² ≤ ∑ k, kSplitErr V (fun j => ranks j.succ) k :=
      ih (fun k => out k.succ) _ V (fun j => ranks j.succ) Y' hrec
    rw [hYeq]
    have hround : reindex E.symm W - A ⋈ₜ Y' = reindex E.symm (W - reindex E (A ⋈ₜ Y')) := by
      rw [reindex_sub, reindex_comp, Equiv.self_trans_symm, reindex_id]
    have hnorm : ‖W - reindex E (A ⋈ₜ Y')‖_F² = ‖reindex E.symm W - A ⋈ₜ Y'‖_F² := by
      rw [hround, frobNormSq_reindex]
    rw [hnorm, tt_truncation_step_error (reindex E.symm W) A V hAorth hVrel Y', herr,
      Matrix.bestRankApproxErrorSq_min, ← kSplitErr_zero_eq_peel, Fin.sum_univ_succ]
    have htail : ‖V - Y'‖_F² ≤ ∑ k : Fin (n + 1), kSplitErr W ranks k.succ :=
      le_trans hVhat (Finset.sum_le_sum fun k _ => hthread k)
    linarith [htail]

/-- `ttApproxOpen` is a truncated TT-SVD: the canonical instance of the predicate
    `IsTruncatedTTSVDOpen` (the natural left-to-right sweep with the Eckart–Young
    best-rank factor at each step). -/
theorem isTruncatedTTSVDOpen_ttApproxOpen : ∀ {n : ℕ} (out : Fin (n + 1) → ℕ) (L : ℕ)
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ),
      IsTruncatedTTSVDOpen n out L W ranks (ttApproxOpen n out L W ranks) := by
  intro n
  induction n with
  | zero =>
    intro out L W ranks
    exact ⟨truncLeft (reindex (castEquiv (Nat.add_comm 2 0)
              (fun i => chainPeel_hd L out i)).symm W)
            (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)) (min_le_right _ _),
          truncRight (reindex (castEquiv (Nat.add_comm 2 0)
              (fun i => chainPeel_hd L out i)).symm W)
            (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)) (min_le_right _ _),
          truncLeft_orthonormal _ _ _, ctrRightMat_truncRight _ _ _, truncStep_err _ _ _, rfl⟩
  | succ n ih =>
    intro out L W ranks
    exact ⟨truncLeft (reindex (castEquiv (Nat.add_comm 2 (n + 1))
              (fun i => chainPeel_hd L out i)).symm W)
            (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)) (min_le_right _ _),
          truncRight (reindex (castEquiv (Nat.add_comm 2 (n + 1))
              (fun i => chainPeel_hd L out i)).symm W)
            (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)) (min_le_right _ _),
          ttApproxOpen n (fun k => out k.succ)
            (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d))
            (truncRight (reindex (castEquiv (Nat.add_comm 2 (n + 1))
                (fun i => chainPeel_hd L out i)).symm W)
              (min (ranks 0) (∏ d, (![L, out 0] : Fin 2 → ℕ) d)) (min_le_right _ _))
            (fun j => ranks j.succ),
          truncLeft_orthonormal _ _ _, ctrRightMat_truncRight _ _ _, truncStep_err _ _ _,
          ih (fun k => out k.succ) _ _ (fun j => ranks j.succ), rfl⟩

/-- **Function-form corollary** of the per-split error bound: the canonical
    output `ttApproxOpen` satisfies it (instance of the predicate theorem). -/
theorem ttApproxOpen_error_bound {n : ℕ} (out : Fin (n + 1) → ℕ) (L : ℕ)
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ) :
      ‖W - ttApproxOpen n out L W ranks‖_F² ≤ ∑ k, kSplitErr W ranks k :=
  isTruncatedTTSVDOpen_error_bound out L W ranks _
    (isTruncatedTTSVDOpen_ttApproxOpen out L W ranks)

/-- **Per-split competitor bound for `kSplitErr`.** If a competitor `Y`'s
    `k`-th split has rank `≤ ranks k` (and `ranks k` does not exceed the split's
    row count), then `kSplitErr W ranks k ≤ ‖W − Y‖²`. The Eckart–Young
    optimality `bestRankApproxErrorSq_le` transported to the `kSplitMat`. -/
theorem kSplitErr_le_competitor {n : ℕ} {out : Fin (n + 1) → ℕ} {L : ℕ}
    (W Y : Tensor 𝕜 (n + 2) (Fin.cons L out)) (ranks : Fin (n + 1) → ℕ) (k : Fin (n + 1))
    (hrk : ranks k ≤ ∏ i : Fin (k.val + 2), (Fin.cons L out : Fin (n + 2) → ℕ)
        (Fin.cast (show n + 2 = (k.val + 2) + (n - k.val) from by omega).symm
          (Fin.castAdd (n - k.val) i)))
    (hY : (kSplitMat Y k).rank ≤ ranks k) :
    kSplitErr W ranks k ≤ ‖W - Y‖_F² := by
  rw [kSplitErr]
  refine (Matrix.bestRankApproxErrorSq_le (kSplitMat W k) (ranks k) hrk
    (kSplitMat Y k) hY).trans_eq ?_
  rw [kSplitMat, kSplitMat, ← modeSplitAt_sub, ← reindex_sub, frobNormSq_modeSplitAt,
    frobNormSq_reindex]

/-! ### Order-`D` form (`thm:tt-svd-error` for a plain tensor `X`)

The recursion above is phrased for a tensor with an exposed leading bond
(`W : Tensor (n+2) (cons L out)`). The lecture's theorem is about a plain tensor
`X` of order `D` with no distinguished mode. We recover it as the `L = 1`
specialization: prepend a unit leading mode (an isometry, `unitConsEquiv`), run
the recursion, and read the per-split errors back as `X`'s own mode splits
`X_{[n]}`. This mirrors how `exists_hasTTDecomp` recovers plain TT existence from
the leading-bond `exists_ttOpen_decomp`. -/

/-- Prepend a unit (`Fin 1`) leading mode to an order-`(D+1)` tensor, giving an
    order-`(D+2)` `Fin.cons 1`-shaped tensor with the same entries (the new
    leading coordinate is forced). A bijective, hence isometric, reshape. -/
def unitConsEquiv {D : ℕ} {I : Fin (D + 1) → ℕ} :
    ((d : Fin (D + 1)) → Fin (I d)) ≃
      ((d : Fin (D + 2)) → Fin ((Fin.cons 1 I : Fin (D + 2) → ℕ) d)) where
  toFun idx := Fin.cons (α := fun d => Fin ((Fin.cons 1 I : Fin (D + 2) → ℕ) d))
    (Fin.cast (by simp) (0 : Fin 1))
    (fun d => Fin.cast (@Fin.cons_succ (D + 1) (fun _ => ℕ) 1 I d).symm (idx d))
  invFun jdx := fun d => Fin.cast (@Fin.cons_succ (D + 1) (fun _ => ℕ) 1 I d) (jdx d.succ)
  left_inv idx := by
    funext d; simp only [Fin.cons_succ]; apply Fin.ext; simp
  right_inv jdx := by
    funext d
    refine Fin.cases ?_ (fun i => ?_) d
    · haveI : Subsingleton (Fin ((Fin.cons 1 I : Fin (D + 2) → ℕ) 0)) := by
        rw [show (Fin.cons 1 I : Fin (D + 2) → ℕ) 0 = 1 from by simp]; infer_instance
      exact Subsingleton.elim _ _
    · simp only [Fin.cons_succ]; apply Fin.ext; simp

/-- The mode-`(n+1)` split matrix of an order-`D` tensor `X : Tensor (D+1) I` (rows:
    the first `n+1` modes; columns: the rest) — the lecture's `X_{[n+1]}`. -/
noncomputable def xSplitMat {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (n : Fin D) :=
  modeSplitAt (p := n.val + 1) (q := D - n.val)
    (reindex (castEquiv (show D + 1 = (n.val + 1) + (D - n.val) from by omega)
      (fun _ => rfl)) X)

/-- The mode-`(n+1)` split error of `X` at truncation rank `R n`:
    `‖X_{[n+1]} − [X_{[n+1]}]_{R n}‖²`. -/
noncomputable def xSplitErr {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) (n : Fin D) : ℝ :=
  Matrix.bestRankApproxErrorSq (xSplitMat X n) (R n)

/-- Drop a unit (`Fin 1`) leading component from a block multi-index: when
    `I₁ 0 = 1` and `J₁ i = I₁ i.succ`, the index sets over `I₁` (arity `m+1`)
    and `J₁` (arity `m`) are equivalent (the head is forced). The block-level
    analogue of `unitConsEquiv`, used as the row equiv in the reshape. -/
def consUnitDropEquiv {m : ℕ} {I₁ : Fin (m + 1) → ℕ} {J₁ : Fin m → ℕ}
    (h0 : I₁ 0 = 1) (hs : ∀ i, J₁ i = I₁ i.succ) :
    ((d : Fin (m + 1)) → Fin (I₁ d)) ≃ ((d : Fin m) → Fin (J₁ d)) where
  toFun f := fun i => Fin.cast (hs i).symm (f i.succ)
  invFun g := Fin.cons (Fin.cast h0.symm (0 : Fin 1)) (fun i => Fin.cast (hs i) (g i))
  left_inv f := by
    funext d
    refine Fin.cases ?_ (fun i => ?_) d
    · simp only [Fin.cons_zero]
      haveI : Subsingleton (Fin (I₁ 0)) := by rw [h0]; infer_instance
      exact Subsingleton.elim _ _
    · simp only [Fin.cons_succ]; apply Fin.ext; simp
  right_inv g := by
    funext i; simp only [Fin.cons_succ]; apply Fin.ext; simp

/-- **Reshape: a unit-bonded split error is the plain mode-split error.** For the
    unit-bonded tensor `W = reindex unitConsEquiv X`, the `(k+2)`-split error of
    `W` (at rank `R k`) equals the plain mode-`(k+1)` split error of `X`. The
    unit leading mode contributes only a trivial row factor, handled by the
    block-aligned reindex invariance. -/
theorem kSplitErr_unitCons_castSucc {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) (k : Fin D) :
    kSplitErr (reindex unitConsEquiv X) (Fin.snoc R 1) k.castSucc = xSplitErr X R k := by
  rw [kSplitErr_eq_modeSplit, Fin.snoc_castSucc, reindex_comp, xSplitErr, xSplitMat,
    bestRankApproxErrorSq_modeSplitAt, reindex_comp]
  refine bestRankApproxErrorSq_reindex_block_aligned X _ _
    (consUnitDropEquiv (m := k.val + 1) ?_ ?_) (dimsCastEquiv ?_) ?_ (R k)
  · simp only [Fin.val_castSucc]
    rw [show (Fin.cast (by have := k.isLt; omega : (↑k + 1 + 1) + (D - ↑k) = D + 2)
          (Fin.castAdd (D - ↑k) (0 : Fin (↑k + 1 + 1)))) = (0 : Fin (D + 2))
        from by apply Fin.ext; simp, Fin.cons_zero]
  · intro i
    simp only [Fin.val_castSucc]
    rw [show (Fin.cast (by have := k.isLt; omega : (↑k + 1 + 1) + (D - ↑k) = D + 2)
            (Fin.castAdd (D - ↑k) (i.succ : Fin (↑k + 1 + 1))))
          = Fin.succ (Fin.cast (by have := k.isLt; omega : (↑k + 1) + (D - ↑k) = D + 1)
              (Fin.castAdd (D - ↑k) i))
        from by apply Fin.ext; simp, Fin.cons_succ]
  · intro i
    simp only [Fin.val_castSucc] at i ⊢
    rw [show (Fin.cast (by have := k.isLt; omega : (↑k + 1 + 1) + (D - ↑k) = D + 2)
            (Fin.natAdd (↑k + 2) i))
          = Fin.succ (Fin.cast (by have := k.isLt; omega : (↑k + 1) + (D - ↑k) = D + 1)
              (Fin.natAdd (↑k + 1) i))
        from by apply Fin.ext; simp; omega, Fin.cons_succ]
  · intro f g
    funext d
    apply Fin.ext
    have hcv : ∀ {x y : ℕ} (heq : Fin x = Fin y) (z : Fin x), (cast heq z).val = z.val := by
      intro x y heq z
      have : x = y := by have := Fintype.card_eq.mpr ⟨(Equiv.cast heq)⟩; simpa using this
      subst this; rfl
    simp only [unitConsEquiv, consUnitDropEquiv, dimsCastEquiv, splitRegroup, castEquiv,
      Equiv.trans, Equiv.symm, Equiv.coe_fn_mk, combineIndex, Function.comp_apply,
      Fin.val_cast, Fin.addCases]
    by_cases hlt : (↑d : ℕ) < ↑k + 1
    · rw [dif_pos (show (↑d.succ : ℕ) < ↑k + 1 + 1 from by rw [Fin.val_succ]; omega), dif_pos hlt]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (f a).val) (Fin.ext (by simp [Fin.val_succ]))
    · rw [dif_neg (show ¬ (↑d.succ : ℕ) < ↑k + 1 + 1 from by rw [Fin.val_succ]; omega), dif_neg hlt,
        eqRec_eq_cast, eqRec_eq_cast, hcv, hcv]
      simp only [Fin.val_cast]
      exact congrArg (fun a => (g a).val) (Fin.ext (by simp [Fin.val_subNat, Fin.val_succ]))

/-- The trailing split of the unit-bonded tensor is degenerate (a single
    column), so its rank-`1` truncation error is `0`. -/
theorem kSplitErr_unitCons_last {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) :
    kSplitErr (reindex unitConsEquiv X) (Fin.snoc R 1) (Fin.last D) = 0 := by
  rw [kSplitErr, Fin.snoc_last]
  apply Matrix.bestRankApproxErrorSq_eq_zero_of_rank_le
  refine (Matrix.rank_le_width _).trans ?_
  haveI : IsEmpty (Fin (D - (Fin.last D).val)) := by
    rw [Fin.val_last, Nat.sub_self]; infer_instance
  exact le_of_eq (Finset.prod_of_isEmpty _)

/-- **The truncated TT-SVD output of an order-`(D+1)` tensor `X`** at target
    TT-ranks `R`, as an explicit witness: prepend a unit leading mode, run the
    leading-bond `ttApproxOpen`, and squeeze the unit bond back out. The `L = 1`
    specialization of `ttApproxOpen`. -/
noncomputable def ttApprox {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) : Tensor 𝕜 (D + 1) I :=
  reindex unitConsEquiv.symm
    (ttApproxOpen D I 1 (reindex unitConsEquiv X) (Fin.snoc R 1))

/-- **`Y` is a truncated TT-SVD of an order-`(D+1)` tensor `X` at TT-ranks `R`.**
    The order-`D` predicate (`L = 1` specialization of `IsTruncatedTTSVDOpen`):
    `Y`'s unit-bonded form is a truncated TT-SVD of `X`'s. Captures every valid
    output of the truncated sweep; `ttApprox` is the canonical instance. Mirrors
    `IsTruncatedHOSVD`. -/
def IsTruncatedTTSVD {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) (Y : Tensor 𝕜 (D + 1) I) : Prop :=
  IsTruncatedTTSVDOpen D I 1 (reindex unitConsEquiv X) (Fin.snoc R 1) (reindex unitConsEquiv Y)

/-- `ttApprox` is a truncated TT-SVD: the canonical instance of `IsTruncatedTTSVD`. -/
theorem isTruncatedTTSVD_ttApprox {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) : IsTruncatedTTSVD X R (ttApprox X R) := by
  rw [IsTruncatedTTSVD, ttApprox, reindex_comp, Equiv.symm_trans_self, reindex_id]
  exact isTruncatedTTSVDOpen_ttApproxOpen I 1 (reindex unitConsEquiv X) (Fin.snoc R 1)

/-- **`thm:tt-svd-error` (order-`D` form, predicate).** *Every* truncated TT-SVD
    `Y` of `X` satisfies

      `‖X − Y‖² ≤ ∑ₙ ‖X_{[n+1]} − [X_{[n+1]}]_{Rₙ}‖²`,

    the squared form of `eq:tt-svd-error`. The `L = 1` specialization of
    `isTruncatedTTSVDOpen_error_bound`, with the per-split errors read back as
    `X`'s own mode splits (`xSplitErr`). -/
theorem tt_svd_error_bound {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) {Y : Tensor 𝕜 (D + 1) I} (hY : IsTruncatedTTSVD X R Y) :
    ‖X - Y‖_F² ≤ ∑ n : Fin D, xSplitErr X R n := by
  have hbound : ‖reindex unitConsEquiv X - reindex unitConsEquiv Y‖_F²
      ≤ ∑ k, kSplitErr (reindex unitConsEquiv X) (Fin.snoc R 1) k :=
    isTruncatedTTSVDOpen_error_bound I 1 (reindex unitConsEquiv X) (Fin.snoc R 1)
      (reindex unitConsEquiv Y) hY
  have hiso : ‖X - Y‖_F² = ‖reindex unitConsEquiv X - reindex unitConsEquiv Y‖_F² := by
    rw [← reindex_sub, frobNormSq_reindex]
  rw [hiso]
  refine hbound.trans (le_of_eq ?_)
  rw [Fin.sum_univ_castSucc, kSplitErr_unitCons_last, add_zero]
  exact Finset.sum_congr rfl (fun k _ => kSplitErr_unitCons_castSucc X R k)

/-- **Function-form corollary**: the canonical output `ttApprox X R` satisfies
    the order-`D` per-split error bound. -/
theorem ttApprox_error_bound {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) :
    ‖X - ttApprox X R‖_F² ≤ ∑ n : Fin D, xSplitErr X R n :=
  tt_svd_error_bound X R (isTruncatedTTSVD_ttApprox X R)

/-- **Per-split competitor bound (order-`D` form).** If a competitor `Y`'s mode-`(n+1)`
    split has rank `≤ R n`, the best rank-`R n` residual of `X_{[n+1]}` is at most
    `‖X − Y‖²`. The order-`D` analogue of `kSplitErr_le_competitor`. -/
theorem xSplitErr_le_competitor {D : ℕ} {I : Fin (D + 1) → ℕ} (X Y : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ) (n : Fin D)
    (hrk : R n ≤ ∏ i : Fin (n.val + 1), I (Fin.cast
        (show D + 1 = (n.val + 1) + (D - n.val) from by omega).symm (Fin.castAdd (D - n.val) i)))
    (hY : (xSplitMat Y n).rank ≤ R n) :
    xSplitErr X R n ≤ ‖X - Y‖_F² := by
  rw [xSplitErr]
  refine (Matrix.bestRankApproxErrorSq_le (xSplitMat X n) (R n) hrk (xSplitMat Y n) hY).trans_eq ?_
  rw [xSplitMat, xSplitMat, ← modeSplitAt_sub, ← reindex_sub, frobNormSq_modeSplitAt,
    frobNormSq_reindex]

/-- **√D quasi-optimality of the truncated TT-SVD (order-`(D+1)` form,
    predicate).** *Any* truncated TT-SVD `Xhat` of `X` is within `√D` of *any*
    competitor `Y` whose mode splits all have rank within the truncation ranks:
    `‖X − Xhat‖ ≤ √D · ‖X − Y‖`. The lecture's quasi-optimality corollary
    (`√(D−1)` for an order-`D` tensor; here order `D+1`, so `√D`). Direct
    consequence of `tt_svd_error_bound` and `xSplitErr_le_competitor`. Mirrors
    `hosvd_quasi_optimal`. -/
theorem tt_svd_quasi_optimal {D : ℕ} {I : Fin (D + 1) → ℕ}
    (X Xhat Y : Tensor 𝕜 (D + 1) I) (R : Fin D → ℕ)
    (hXhat : IsTruncatedTTSVD X R Xhat)
    (hrk : ∀ n, R n ≤ ∏ i : Fin (n.val + 1), I (Fin.cast
        (show D + 1 = (n.val + 1) + (D - n.val) from by omega).symm (Fin.castAdd (D - n.val) i)))
    (hY : ∀ n, (xSplitMat Y n).rank ≤ R n) :
    ‖X - Xhat‖_F² ≤ (D : ℝ) * ‖X - Y‖_F² ∧
      ‖X - Xhat‖_F ≤ Real.sqrt D * ‖X - Y‖_F := by
  have hbound : ‖X - Xhat‖_F² ≤ (D : ℝ) * ‖X - Y‖_F² := by
    refine (tt_svd_error_bound X R hXhat).trans ?_
    refine (Finset.sum_le_sum
      (fun n _ => xSplitErr_le_competitor X Y R n (hrk n) (hY n))).trans_eq ?_
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine ⟨hbound, ?_⟩
  have hsq : ‖X - Xhat‖_F ^ 2 ≤ ((D : ℕ) : ℝ) * ‖X - Y‖_F ^ 2 := by
    rw [frobNorm_sq, frobNorm_sq]; exact hbound
  exact le_sqrt_natCast_mul_of_sq_le (N := D) (frobNorm_nonneg _) (frobNorm_nonneg _) hsq

/-- **Function-form corollary** of quasi-optimality: the canonical output
    `ttApprox X R` is within `√D` of any competitor `Y` of split rank `≤ R`. -/
theorem ttApprox_quasi_optimal {D : ℕ} {I : Fin (D + 1) → ℕ} (X Y : Tensor 𝕜 (D + 1) I)
    (R : Fin D → ℕ)
    (hrk : ∀ n, R n ≤ ∏ i : Fin (n.val + 1), I (Fin.cast
        (show D + 1 = (n.val + 1) + (D - n.val) from by omega).symm (Fin.castAdd (D - n.val) i)))
    (hY : ∀ n, (xSplitMat Y n).rank ≤ R n) :
    ‖X - ttApprox X R‖_F² ≤ (D : ℝ) * ‖X - Y‖_F² ∧
      ‖X - ttApprox X R‖_F ≤ Real.sqrt D * ‖X - Y‖_F :=
  tt_svd_quasi_optimal X (ttApprox X R) Y R (isTruncatedTTSVD_ttApprox X R) hrk hY

end Tensor
