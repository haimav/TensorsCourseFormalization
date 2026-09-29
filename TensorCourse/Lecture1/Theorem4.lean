/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.CPDecomposition.Defs
import TensorCourse.CPDecomposition.OuterProd
import TensorCourse.CPDecomposition.ReshapeRank
import TensorCourse.MatMulTensor.EncodesMatMul
import TensorCourse.MatMulTensor.OuterProdIso

/-!
# Theorem 4 (Lecture 1): CP decomposition of the matrix multiplication tensor
# yields a bilinear algorithm, and its "power" corollary

This file proves the main algebraic content of Theorem 4 from Lecture 1 and
the corollary combining Results 1–3: any rank-`R` CP decomposition of the
matrix multiplication tensor `matMulTensor 𝕜 M` gives a bilinear algorithm
that computes the product of two `M × M` matrices using `R` scalar
multiplications of bilinear forms, and raising the construction to the
`K`-th "power" via iterated outer products and mode fusion gives a bilinear
algorithm for `M^K × M^K` matrix multiplication using `R^K` bilinear
products.

## Main results

* `matMul_via_bilinear_forms` — for a CP decomposition `cp` of
  `matMulTensor 𝕜 M` of rank `R`, every entry of `vec (A * B)` can be
  written as a sum of `R` terms, each a product of one entry of the
  "output" factor with a linear form in `Matrix.vectorize A` and a linear form in
  `Matrix.vectorize B`.
* `matMulTensor_one_cp` — the trivial rank-`1` CP decomposition of
  `matMulTensor 𝕜 1` (the base case for the inductive construction).
* `exists_matMulTensor_pow_cp` — from a rank-`R` CP decomposition of
  `matMulTensor 𝕜 M`, construct a rank-`R^K` CP decomposition of
  `matMulTensor 𝕜 (M^K)` for every `K`, by iterated outer product and
  mode fusion.
* `cpRank_matMulTensor_pow_le` — the resulting CP rank bound
  `cpRank (matMulTensor 𝕜 (M^K)) ≤ R^K`.
* `matMul_pow_via_bilinear_forms` — the full content of Theorem 4: a
  bilinear algorithm for `M^K × M^K` matrix multiplication using `R^K`
  bilinear products, obtained by combining `exists_matMulTensor_pow_cp`
  with `matMul_via_bilinear_forms`.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- **Matrix multiplication via a CP decomposition of the matrix
    multiplication tensor.**

    If `cp` is a rank-`R` CP decomposition of `matMulTensor 𝕜 M`, then for
    any two `M × M` matrices `A`, `B` and any output coordinate
    `l : Fin (M * M)`:

    `vec (A * B) l = ∑ r, cp.factors r 2 l
                         * (∑ i, cp.factors r 0 i * Matrix.vectorize A i)
                         * (∑ j, cp.factors r 1 j * Matrix.vectorize B j)`

    In words: the `l`-th entry of `vec (A * B)` is computed as a sum of
    `R` terms, where each term is a product of three scalars:
    1. an entry of the "output" factor (mode 2),
    2. a linear combination of entries of `A` with coefficients from the
       "A-side" factor (mode 0), and
    3. a linear combination of entries of `B` with coefficients from the
       "B-side" factor (mode 1).

    The key algorithmic content is that only `R` scalar multiplications of
    pairs of linear forms (the "bilinear products") are needed to compute
    all of `A * B`. Each such bilinear product is shared across all output
    coordinates `l`; only the coefficient `cp.factors r 2 l` depends on `l`. -/
theorem matMul_via_bilinear_forms {M : ℕ} {R : ℕ}
    (cp : CPDecomp 𝕜 3 (matMulDims M) R)
    (hcp : cp.tensor = matMulTensor 𝕜 M)
    (A B : Matrix (Fin M) (Fin M) 𝕜) (l : Fin (M * M)) :
    Matrix.vectorize (A * B) l =
      ∑ r : Fin R,
        cp.factors r 2 l *
        (∑ i : Fin (M * M), cp.factors r 0 i * Matrix.vectorize A i) *
        (∑ j : Fin (M * M), cp.factors r 1 j * Matrix.vectorize B j) := by
  -- Start from the encoding-correctness theorem, rewritten backwards.
  rw [← matMulTensor_encodes_matMul A B l]
  -- Replace `matMulTensor 𝕜 M` by `cp.tensor` using the hypothesis.
  rw [← hcp]
  -- Unfold `cp.tensor` and expand the product over `Fin 3`.
  simp only [CPDecomp.tensor, Fin.prod_univ_three, triIdx_zero, triIdx_one,
    triIdx_two]
  -- The goal now reads:
  --   ∑ i, ∑ j, (∑ r, cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l)
  --              * Matrix.vectorize A i * Matrix.vectorize B j
  --   = ∑ r, cp.factors r 2 l
  --         * (∑ i, cp.factors r 0 i * Matrix.vectorize A i)
  --         * (∑ j, cp.factors r 1 j * Matrix.vectorize B j)
  -- Push the two leading scalars `Matrix.vectorize A i`, `Matrix.vectorize B j` into the inner sum
  -- over `r`, producing a triple sum on the LHS.
  simp_rw [Finset.sum_mul]
  -- LHS: ∑ i, ∑ j, ∑ r, (cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l)
  --                     * Matrix.vectorize A i * Matrix.vectorize B j
  -- Pull ∑ r out to the outermost position by two applications of sum_comm:
  --   first swap the inner pair ∑ j, ∑ r (inside the scope of i) to get
  --     ∑ i, ∑ r, ∑ j, ...
  --   then swap the now-adjacent outer pair ∑ i, ∑ r to get
  --     ∑ r, ∑ i, ∑ j, ...
  rw [show (∑ i : Fin (M * M), ∑ j : Fin (M * M), ∑ r : Fin R,
        cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l *
          Matrix.vectorize A i * Matrix.vectorize B j)
      = (∑ i : Fin (M * M), ∑ r : Fin R, ∑ j : Fin (M * M),
        cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l *
          Matrix.vectorize A i * Matrix.vectorize B j) from
    Finset.sum_congr rfl (fun i _ => Finset.sum_comm)]
  rw [Finset.sum_comm]
  -- LHS is now: ∑ r, ∑ i, ∑ j,
  --   cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l
  --     * Matrix.vectorize A i * Matrix.vectorize B j
  refine Finset.sum_congr rfl (fun r _ => ?_)
  -- For a fixed `r`, we need:
  --   ∑ i, ∑ j, cp.factors r 0 i * cp.factors r 1 j * cp.factors r 2 l
  --              * Matrix.vectorize A i * Matrix.vectorize B j
  --   = cp.factors r 2 l
  --       * (∑ i, cp.factors r 0 i * Matrix.vectorize A i)
  --       * (∑ j, cp.factors r 1 j * Matrix.vectorize B j)
  -- Expand the RHS double product-of-sums into a double sum of products.
  rw [mul_assoc, Finset.sum_mul_sum]
  -- Distribute `cp.factors r 2 l` into the inner double sum.
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  ring

/-! ### Base case for the inductive construction: `matMulTensor 𝕜 1` -/

/-- The trivial rank-`1` CP decomposition of `matMulTensor 𝕜 1`.

    When `M = 1`, each mode of `matMulTensor 𝕜 1` has dimension `1 * 1 = 1`,
    so all multi-indices are equal (there is only one) and the tensor is the
    constant-`1` tensor on a singleton shape. The rank-one decomposition
    whose single factor vector along every mode is the constant `1`
    reconstructs it. -/
def matMulTensor_one_cp : CPDecomp 𝕜 3 (matMulDims 1) 1 where
  factors := fun _ _ _ => 1

/-- The trivial rank-`1` CP decomposition reconstructs `matMulTensor 𝕜 1`. -/
theorem matMulTensor_one_cp_tensor :
    (matMulTensor_one_cp (𝕜 := 𝕜)).tensor = matMulTensor 𝕜 1 := by
  ext idx
  -- Both sides evaluate to `1`:
  -- LHS is the constant-1 tensor (single term, single factor per mode).
  -- RHS is 1 because the Kronecker-delta condition is trivially true — all
  -- components live in `Fin (1 * 1) = Fin 1`, a subsingleton.
  have hp0 : (finProdFinEquiv.symm (idx 0)).2 = (finProdFinEquiv.symm (idx 2)).2 :=
    Subsingleton.elim _ _
  have hp1 : (finProdFinEquiv.symm (idx 1)).1 = (finProdFinEquiv.symm (idx 2)).1 :=
    Subsingleton.elim _ _
  have hp2 : (finProdFinEquiv.symm (idx 0)).1 = (finProdFinEquiv.symm (idx 1)).2 :=
    Subsingleton.elim _ _
  simp [CPDecomp.tensor, matMulTensor, matMulTensor_one_cp, hp0, hp1, hp2]

/-! ### The inductive construction of a CP decomposition of `matMulTensor 𝕜 (M^K)` -/

/-- **Inductive construction.** Given a rank-`R` CP decomposition `cp` of
    `matMulTensor 𝕜 M`, for every `K : ℕ` there exists a rank-`R^K` CP
    decomposition of `matMulTensor 𝕜 (M^K)`.

    The construction is by induction on `K`:

    * `K = 0`: use `matMulTensor_one_cp`.
    * `K → K+1`: form the outer-product CP decomposition
      `cp_K ◯ₜ cp : CPDecomp 𝕜 6 _ (R^K * R)` (via `CPDecomp.outerProdCP`),
      then mode-fuse it along `fuseSigma₃` with the per-fiber equivalences
      `outerDimsEquiv (M^K) M` (via `CPDecomp.fuse`). By
      `matMulTensor_eq_fuse_outerProd`, the resulting fused tensor equals
      `matMulTensor 𝕜 (M^K * M) = matMulTensor 𝕜 (M^(K+1))` (the last
      equality is definitional because `Nat.pow_succ` reduces to `rfl`). -/
theorem exists_matMulTensor_pow_cp {M R : ℕ}
    (cp : CPDecomp 𝕜 3 (matMulDims M) R)
    (hcp : cp.tensor = matMulTensor 𝕜 M)
    (K : ℕ) :
    ∃ cp' : CPDecomp 𝕜 3 (matMulDims (M ^ K)) (R ^ K),
      cp'.tensor = matMulTensor 𝕜 (M ^ K) := by
  induction K with
  | zero =>
    -- `M ^ 0 = 1` and `R ^ 0 = 1` are both definitional.
    exact ⟨matMulTensor_one_cp, matMulTensor_one_cp_tensor⟩
  | succ K ih =>
    -- Extract the induction hypothesis witness.
    obtain ⟨cp_K, hcp_K⟩ := ih
    -- Form the outer-product CP decomposition of the order-6 tensor
    -- `matMulTensor 𝕜 (M^K) ◯ₜ matMulTensor 𝕜 M`, then fuse along
    -- `fuseSigma₃` with the per-fiber equivalences `outerDimsEquiv (M^K) M`.
    -- The result is an order-3 CP decomposition with shape
    -- `matMulDims (M^K * M)` and rank `R^K * R`. Since `M^(K+1) = M^K * M`
    -- and `R^(K+1) = R^K * R` are definitional (`Nat.pow_succ` is `rfl`),
    -- this is exactly what we want.
    refine ⟨(cp_K.outerProdCP cp).fuse fuseSigma₃ (outerDimsEquiv (M ^ K) M), ?_⟩
    -- The fused decomposition's tensor equals the reindex of the outer
    -- product tensor, which by `matMulTensor_eq_fuse_outerProd` equals
    -- `matMulTensor 𝕜 (M^K * M) = matMulTensor 𝕜 (M^(K+1))`.
    rw [CPDecomp.fuse_tensor]
    -- The fused tensor on the LHS unfolds from `(fuse ...).tensor` to
    -- `reindex ... (cp_K ◯ₜ cp).tensor`, but the pretty-printed `◯ₜ` is
    -- ambiguous: Lean stores the left factor as `cp_K.outerProdCP cp`
    -- using the `CPDecomp.outerProdCP` definition rather than via the
    -- scoped notation. `change` forces the display to match the underlying
    -- term so that the next `rw` can find `CPDecomp.outerProd_tensor`.
    change Tensor.reindex (fuseEquiv fuseSigma₃ (outerDimsEquiv (M ^ K) M))
             (cp_K.outerProdCP cp).tensor = matMulTensor 𝕜 (M ^ (K + 1))
    rw [CPDecomp.outerProd_tensor, hcp_K, hcp, ← matMulTensor_eq_fuse_outerProd]
    -- Target: `matMulTensor 𝕜 (M^K * M) = matMulTensor 𝕜 (M^(K+1))`.
    -- `M^(K+1) = M^K * M` is definitional (`Nat.pow_succ` is `rfl`), but
    -- the goal state needs a nudge to see it.
    rfl

/-! ### Rank bound corollary -/

/-- **Rank bound corollary.** If `matMulTensor 𝕜 M` has a rank-`R` CP
    decomposition, then `cpRank (matMulTensor 𝕜 (M^K)) ≤ R^K` for every
    `K : ℕ`.

    Immediate from `exists_matMulTensor_pow_cp` and `cpRank_le_of_cpDecomp`. -/
theorem cpRank_matMulTensor_pow_le {M R : ℕ}
    (cp : CPDecomp 𝕜 3 (matMulDims M) R)
    (hcp : cp.tensor = matMulTensor 𝕜 M)
    (K : ℕ) :
    cpRank (matMulTensor 𝕜 (M ^ K)) ≤ R ^ K := by
  obtain ⟨cp', hcp'⟩ := exists_matMulTensor_pow_cp cp hcp K
  exact cpRank_le_of_cpDecomp cp' hcp'

/-! ### Full content of Theorem 4: bilinear algorithm for `M^K × M^K` matrix
    multiplication from a rank-`R` decomposition of `matMulTensor 𝕜 M` -/

/-- **Matrix multiplication of `M^K × M^K` matrices via a CP decomposition of
    `matMulTensor 𝕜 M`.**

    Given a rank-`R` CP decomposition `cp` of `matMulTensor 𝕜 M`, there exists
    a rank-`R^K` CP decomposition `cp'` of `matMulTensor 𝕜 (M^K)` such that
    for every pair of `M^K × M^K` matrices `A`, `B` and every output
    coordinate `l : Fin (M^K * M^K)`:

    `vec (A * B) l = ∑ r, cp'.factors r 2 l
                         * (∑ i, cp'.factors r 0 i * Matrix.vectorize A i)
                         * (∑ j, cp'.factors r 1 j * Matrix.vectorize B j)`

    In words: the product `A * B` of two `M^K × M^K` matrices can be computed
    by `R^K` scalar multiplications of bilinear forms in the entries of `A`
    and `B`. This is the full algorithmic content of Theorem 4 from Lecture 1,
    obtained by combining Results 1–3 of the algebraic approach.

    The `R^K` decomposition is constructed explicitly by `exists_matMulTensor_pow_cp`,
    and the bilinear-form statement then follows from `matMul_via_bilinear_forms`
    applied to the constructed decomposition. -/
theorem matMul_pow_via_bilinear_forms {M R : ℕ}
    (cp : CPDecomp 𝕜 3 (matMulDims M) R)
    (hcp : cp.tensor = matMulTensor 𝕜 M)
    (K : ℕ)
    (A B : Matrix (Fin (M ^ K)) (Fin (M ^ K)) 𝕜)
    (l : Fin (M ^ K * M ^ K)) :
    ∃ cp' : CPDecomp 𝕜 3 (matMulDims (M ^ K)) (R ^ K),
      Matrix.vectorize (A * B) l =
        ∑ r : Fin (R ^ K),
          cp'.factors r 2 l *
          (∑ i : Fin (M ^ K * M ^ K), cp'.factors r 0 i * Matrix.vectorize A i) *
          (∑ j : Fin (M ^ K * M ^ K), cp'.factors r 1 j * Matrix.vectorize B j) := by
  obtain ⟨cp', hcp'⟩ := exists_matMulTensor_pow_cp cp hcp K
  exact ⟨cp', matMul_via_bilinear_forms cp' hcp' A B l⟩

end Tensor
