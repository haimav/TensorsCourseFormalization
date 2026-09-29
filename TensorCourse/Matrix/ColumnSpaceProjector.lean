/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Subspace
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Column-space orthonormal projector (lemma B1)

For any matrix `M : Matrix (Fin I) (Fin C) 𝕜` there is a matrix
`U : Matrix (Fin I) (Fin M.rank) 𝕜` with orthonormal columns
(`Uᴴ * U = 1`) whose associated orthogonal projector `U Uᴴ` fixes the
columns of `M` (`U * Uᴴ * M = M`). Geometrically `U`'s columns are an
orthonormal basis of the column space `col(M)`, and `U Uᴴ` is the matrix of
the orthogonal projection onto `col(M)`.

This is the **provable lemma B1** driving the HOSVD existence theorem
`thm:hosvd` (see `TuckerDecomposition/HOSVD.lean`). It is *not* an axiom:
the orthogonal projection onto a finite-dimensional subspace exists
unconditionally in Mathlib (`Submodule.orthogonalProjection`), and we build
`U` from an `OrthonormalBasis` of `col(M)`. Reference: any SVD/QR account,
e.g. Bernstein, *Matrix Mathematics*, §5.6 / §6.

## Main results

* `Matrix.exists_colspace_orthonormal_projector` — lemma B1.
-/

noncomputable section

open scoped Matrix InnerProductSpace
open Module WithLp

namespace Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- **Lemma B1 (column-space orthonormal projector).** For any matrix
    `M : Matrix (Fin I) (Fin C) 𝕜`, there is a matrix
    `U : Matrix (Fin I) (Fin M.rank) 𝕜` with orthonormal columns
    (`Uᴴ * U = 1`) whose projector `U Uᴴ` fixes `M`'s columns
    (`U * Uᴴ * M = M`).

    Construction: `col(M) = range M.mulVecLin`, transported into the inner
    product space `EuclideanSpace 𝕜 (Fin I)`, is a finite-dimensional
    subspace `K` of `finrank = M.rank`. An `OrthonormalBasis (Fin M.rank) 𝕜 K`
    `b` exists; stack its (ambient-coerced) vectors as the columns of `U`.
    Then `Uᴴ U = 1` is orthonormality of `b`, and `U Uᴴ M = M` is the fact
    that the orthogonal projection onto `K` fixes each column of `M`
    (the columns lie in `K`), read off coordinatewise via the projection sum
    formula `OrthonormalBasis.orthogonalProjection_apply_eq_sum`. -/
theorem exists_colspace_orthonormal_projector {I C : ℕ}
    (M : Matrix (Fin I) (Fin C) 𝕜) :
    ∃ U : Matrix (Fin I) (Fin M.rank) 𝕜, Uᴴ * U = 1 ∧ U * Uᴴ * M = M := by
  classical
  -- The module isomorphism `EuclideanSpace 𝕜 (Fin I) ≃ₗ[𝕜] (Fin I → 𝕜)`.
  set e : EuclideanSpace 𝕜 (Fin I) ≃ₗ[𝕜] (Fin I → 𝕜) :=
    WithLp.linearEquiv 2 𝕜 (Fin I → 𝕜) with he
  -- The column space of `M`, transported into `EuclideanSpace`.
  set K : Submodule 𝕜 (EuclideanSpace 𝕜 (Fin I)) :=
    (LinearMap.range M.mulVecLin).map e.symm.toLinearMap with hK
  -- `finrank K = M.rank`.
  have hrank : finrank 𝕜 K = M.rank := by
    rw [hK, LinearEquiv.finrank_map_eq]
    rfl
  -- An orthonormal basis of `K`, reindexed to `Fin M.rank`.
  set b : OrthonormalBasis (Fin M.rank) 𝕜 K :=
    (stdOrthonormalBasis 𝕜 K).reindex (finCongr hrank) with hb
  -- The factor matrix: column `j` is the ambient vector `b j`, read off
  -- coordinatewise.
  set U : Matrix (Fin I) (Fin M.rank) 𝕜 :=
    Matrix.of (fun i j => ((b j : EuclideanSpace 𝕜 (Fin I)) i)) with hU
  refine ⟨U, ?_, ?_⟩
  · -- Orthonormality: `(Uᴴ * U) j k = ⟪b j, b k⟫ = δ_{jk}`.
    funext j k
    rw [Matrix.mul_apply, Matrix.one_apply]
    -- The entry `∑ i, Uᴴ j i * U i k` is the inner product `⟪b j, b k⟫`.
    have hinner : (∑ i : Fin I, Uᴴ j i * U i k)
        = ⟪(b j : EuclideanSpace 𝕜 (Fin I)), (b k : EuclideanSpace 𝕜 (Fin I))⟫_𝕜 := by
      rw [EuclideanSpace.inner_eq_star_dotProduct, dotProduct]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      simp only [hU, Matrix.conjTranspose_apply, Matrix.of_apply, Pi.star_apply,
        RCLike.star_def]
      ring
    rw [hinner]
    -- `⟪b j, b k⟫ = ⟪(b j : K), (b k : K)⟫` then orthonormality of `b`.
    rw [← Submodule.coe_inner, orthonormal_iff_ite.mp b.orthonormal j k]
  · -- `U Uᴴ M = M`: column `k` is fixed because `M.col k ∈ K`.
    funext i k
    -- The `k`-th column of `M` as a Euclidean vector.
    set colk : EuclideanSpace 𝕜 (Fin I) := e.symm (fun p => M p k) with hcolk
    -- `colk ∈ K`.
    have hmem : colk ∈ K := by
      rw [hcolk, hK]
      apply Submodule.mem_map_of_mem
      rw [LinearMap.mem_range]
      refine ⟨Pi.single k 1, ?_⟩
      funext p
      rw [Matrix.mulVecLin_apply, Matrix.mulVec_single_one]
      rfl
    -- The orthogonal projection onto `K` fixes `colk`.
    have hproj : K.orthogonalProjection colk = (⟨colk, hmem⟩ : K) :=
      Submodule.orthogonalProjection_mem_subspace_eq_self ⟨colk, hmem⟩
    -- The projection sum formula evaluated coordinatewise at `i`.
    have hsum : (colk i) = ∑ j : Fin M.rank,
        ⟪(b j : EuclideanSpace 𝕜 (Fin I)), colk⟫_𝕜 * (b j : EuclideanSpace 𝕜 (Fin I)) i := by
      have hpsum := b.orthogonalProjection_apply_eq_sum colk
      rw [hproj] at hpsum
      -- `hpsum : (⟨colk, hmem⟩ : K) = ∑ j, ⟪b j, colk⟫ • b j` (in `K`).
      -- Apply the ambient-coercion and read coordinate `i`.
      have hcoe : colk = ((∑ j : Fin M.rank,
          ⟪(b j : EuclideanSpace 𝕜 (Fin I)), colk⟫_𝕜 • b j : K) :
            EuclideanSpace 𝕜 (Fin I)) := by
        rw [← hpsum]
      -- Push the coercion through the sum, then read coordinate `i`. Rewrite
      -- only the left-hand `colk i` (not the `colk` inside the inner product).
      conv_lhs => rw [hcoe, Submodule.coe_sum]
      -- `((∑ j, c j • b j : K) : E) i = ∑ j, c j • (b j : E) i`. Read the
      -- coordinate via the linear equiv `e` (= `ofLp`), whose coercion to a
      -- function is the `PiLp` coercion definitionally.
      have hcoord : ∀ (v : EuclideanSpace 𝕜 (Fin I)), v i = e v i := fun v => rfl
      rw [hcoord, map_sum, Finset.sum_apply]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      rw [Submodule.coe_smul, map_smul, Pi.smul_apply, smul_eq_mul]
      rfl
    -- Now compute `(U Uᴴ M) i k` and match.
    rw [Matrix.mul_apply]
    -- `(U * Uᴴ * M) i k = ∑ p, (U * Uᴴ) i p * M p k`.
    have hUUH : ∀ p : Fin I, (U * Uᴴ) i p =
        ∑ j : Fin M.rank, (b j : EuclideanSpace 𝕜 (Fin I)) i *
          (starRingEnd 𝕜) ((b j : EuclideanSpace 𝕜 (Fin I)) p) := by
      intro p
      rw [Matrix.mul_apply]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      simp only [hU, Matrix.conjTranspose_apply, Matrix.of_apply, RCLike.star_def]
    -- Show `M i k = colk i` and rewrite via `hsum`.
    have hMcolk : M i k = colk i := by
      rw [hcolk]
      rfl
    rw [show M i k = colk i from hMcolk, hsum]
    -- LHS: `∑ p, (∑ j, (b j) i * conj((b j) p)) * M p k`.
    -- RHS: `∑ j, ⟪b j, colk⟫ * (b j) i`.
    simp only [hUUH, Finset.sum_mul]
    -- LHS: `∑ p, ∑ j, ((b j) i * conj((b j) p)) * M p k`.
    -- Swap the sums to `∑ j, ∑ p, …`.
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    -- inner term: `∑ p, ((b j) i * conj((b j) p)) * M p k = ⟪b j, colk⟫ * (b j) i`.
    rw [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Finset.sum_mul]
    refine Finset.sum_congr rfl (fun p _ => ?_)
    -- `colk p = M p k` and `ofLp/star` bookkeeping.
    have hcolkp : colk p = M p k := by rw [hcolk]; rfl
    simp only [Pi.star_apply, RCLike.star_def, hcolkp]
    ring

end Matrix

end
