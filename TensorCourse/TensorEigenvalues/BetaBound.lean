/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.Transpose
import TensorCourse.TensorEigenvalues.Apply
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Analysis.Normed.Algebra.Spectrum
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# A User-Friendly Bound on the Convergence Shift β(A) (Lecture 8, lem:beta-bound)

The shifted higher-order power method converges once the shift exceeds

  β(A) := (order−1) · max_{‖x‖₂=1} ρ( A contracted with x on all but its first two modes ),

where ρ is the spectral radius of the resulting symmetric `N × N` matrix. Lecture 8
gives the user-friendly bound

  β(A) ≤ (order−1) · ∑_{i₁,…,i_D} |a_{i₁…i_D}| .

Formalised over `ℝ` (the lemma is intrinsically about real *symmetric* matrices).
For a cubic tensor of Lean-order `D+2`, contracting the last `D` modes with `x`
leaves the `N × N` matrix `applyMatrix A x`; `β` is `(D+1)` times the supremum of
its spectral radius over the unit sphere. Each eigenvalue of the symmetric matrix
is a Rayleigh quotient at a unit eigenvector (`Matrix.IsHermitian.eigenvalues_eq`),
which the triangle inequality bounds by `∑|a|`.

## Main definitions

* `CubicTensor.applyMatrix A x` — the `N × N` matrix contracting the last `D` modes.
* `CubicTensor.beta A` — the convergence shift `β(A)` (ℝ≥0∞-valued).

## Main results

* `CubicTensor.beta_le` — `β(A) ≤ (D+1) · ∑ |a|` for symmetric `A`.
-/

noncomputable section

open Finset BigOperators Matrix
open scoped ENNReal NNReal

namespace CubicTensor

variable {D N : ℕ}

/-! ### Symmetry helpers (local copies of the reindexing used elsewhere) -/

/-- Cubic-tensor transpose acts by precomposition with `σ.symm` (the `▸` cast in
    `Tensor.transpose` is along a constant motive, hence the identity). -/
private theorem cubicTranspose_apply {O : ℕ} (σ : Equiv.Perm (Fin O))
    (T : CubicTensor ℝ O N) (idx : Fin O → Fin N) :
    T.transpose σ idx = T (fun d => idx (σ.symm d)) := by
  simp only [Tensor.transpose_apply]
  congr 1
  funext d
  apply eq_of_heq
  exact eqRec_heq_iff_heq.mpr HEq.rfl

/-- A symmetric tensor is invariant under precomposing its index with any permutation. -/
private theorem applyPerm {O : ℕ} {A : CubicTensor ℝ O N} (hA : A.IsSymmetric)
    (σ : Equiv.Perm (Fin O)) (idx : Fin O → Fin N) :
    A (fun d => idx (σ d)) = A idx := by
  have h : A.transpose σ.symm idx = A idx := by rw [hA σ.symm]
  rw [cubicTranspose_apply] at h
  simpa only [Equiv.symm_symm] using h

/-- Reindexing `cons b (cons a r)` by `swap 0 1` swaps the first two entries. -/
private theorem cons_swap01 (a b : Fin N) (r : Fin D → Fin N) :
    (fun d : Fin (D + 2) =>
        (Fin.cons b (Fin.cons a r) : Fin (D + 2) → Fin N) ((Equiv.swap (0 : Fin (D + 2)) 1) d))
      = (Fin.cons a (Fin.cons b r) : Fin (D + 2) → Fin N) := by
  have h10 : (Fin.succ (0 : Fin (D + 1))) = (1 : Fin (D + 2)) := Fin.succ_zero_eq_one
  funext d
  rcases Fin.eq_zero_or_eq_succ d with h0 | ⟨d1, rfl⟩
  · subst h0
    rw [Equiv.swap_apply_left, ← h10, Fin.cons_succ, Fin.cons_zero, Fin.cons_zero]
  · rcases Fin.eq_zero_or_eq_succ d1 with h1 | ⟨d2, rfl⟩
    · subst h1
      rw [h10, Equiv.swap_apply_right, Fin.cons_zero, ← h10, Fin.cons_succ, Fin.cons_zero]
    · have hne0 : (Fin.succ (Fin.succ d2)) ≠ (0 : Fin (D + 2)) := Fin.succ_ne_zero _
      have hne1 : (Fin.succ (Fin.succ d2)) ≠ (1 : Fin (D + 2)) := by
        rw [← h10]; exact fun h => Fin.succ_ne_zero _ (Fin.succ_injective _ h)
      rw [Equiv.swap_apply_of_ne_of_ne hne0 hne1, Fin.cons_succ, Fin.cons_succ,
        Fin.cons_succ, Fin.cons_succ]

/-! ### The contracted matrix -/

/-- Contract the last `D` modes of an order-`(D+2)` cubic tensor `A` with `x`,
    leaving the `N × N` matrix `(applyMatrix A x) i j = ∑_rest A(i,j,rest) ∏ x(rest)`. -/
def applyMatrix (A : CubicTensor ℝ (D + 2) N) (x : Fin N → ℝ) :
    Matrix (Fin N) (Fin N) ℝ :=
  fun i j => ∑ rest : Fin D → Fin N,
    A (Fin.cons i (Fin.cons j rest)) * ∏ d : Fin D, x (rest d)

/-- **The contracted matrix of a symmetric tensor is symmetric (Hermitian over ℝ).** -/
theorem applyMatrix_isHermitian (A : CubicTensor ℝ (D + 2) N)
    (hA : A.IsSymmetric) (x : Fin N → ℝ) :
    (applyMatrix A x).IsHermitian := by
  rw [Matrix.IsHermitian]
  ext i j
  simp only [Matrix.conjTranspose_apply, star_trivial]
  change (∑ rest : Fin D → Fin N, A (Fin.cons j (Fin.cons i rest)) * ∏ d, x (rest d))
     = ∑ rest : Fin D → Fin N, A (Fin.cons i (Fin.cons j rest)) * ∏ d, x (rest d)
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  congr 1
  have key := applyPerm hA (Equiv.swap (0 : Fin (D + 2)) 1) (Fin.cons i (Fin.cons j rest))
  rw [cons_swap01] at key
  exact key

/-! ### The β quantity -/

/-- The convergence shift `β(A)` (ℝ≥0∞-valued): `(D+1)` times the supremum over the
    unit sphere of the spectral radius of the contracted matrix. -/
noncomputable def beta (A : CubicTensor ℝ (D + 2) N) : ℝ≥0∞ :=
  ((D : ℝ≥0∞) + 1) *
    ⨆ x : {x : Fin N → ℝ // ∑ i, x i ^ 2 = 1}, spectralRadius ℝ (applyMatrix A x.1)

/-! ### Bounds -/

/-- Reindex a sum over full multi-indices `Fin (D+2) → Fin N` as a triple sum over
    the first index `i`, the second index `j`, and the remaining `D` indices `rest`,
    via a double application of `Fin.consEquiv`. -/
private theorem sum_cons2 {M : Type*} [AddCommMonoid M]
    (f : (Fin (D + 2) → Fin N) → M) :
    ∑ idx : Fin (D + 2) → Fin N, f idx
      = ∑ i : Fin N, ∑ j : Fin N, ∑ rest : Fin D → Fin N,
          f (Fin.cons i (Fin.cons j rest)) := by
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) f,
      Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have step : ∀ tail : Fin (D + 1) → Fin N,
      (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) (i, tail) = Fin.cons i tail := by
    intro tail; funext d; exact Fin.consEquiv_apply _ _ _
  simp only [step]
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N))
        (fun tail => f (Fin.cons i tail)), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  have hc : (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) (j, rest) = Fin.cons j rest := by
    funext d; exact Fin.consEquiv_apply _ _ _
  rw [hc]

/-- The sum of the contracted matrix's entrywise absolute values is at most `∑|a|`
    (for a unit vector `x`). Pure algebra: push `‖·‖` through the sum/product and
    bound `∏‖x‖ ≤ 1`. -/
theorem sum_nnnorm_applyMatrix_le (A : CubicTensor ℝ (D + 2) N) {x : Fin N → ℝ}
    (hx : ∑ i, x i ^ 2 = 1) :
    ∑ i, ∑ j, ‖applyMatrix A x i j‖₊ ≤ ∑ idx : Fin (D + 2) → Fin N, ‖A idx‖₊ := by
  -- Each coordinate of the unit vector has `nnnorm ≤ 1`.
  have hxle : ∀ k : Fin N, ‖x k‖₊ ≤ 1 := by
    intro k
    have hk2 : x k ^ 2 ≤ 1 := by
      rw [← hx]
      exact Finset.single_le_sum (fun l _ => sq_nonneg (x l)) (mem_univ k)
    have habs : |x k| ≤ 1 := (sq_le_one_iff_abs_le_one _).mp hk2
    have hcoe : (‖x k‖₊ : ℝ) ≤ ((1 : ℝ≥0) : ℝ) := by
      rw [coe_nnnorm, Real.norm_eq_abs, NNReal.coe_one]; exact habs
    exact NNReal.coe_le_coe.mp hcoe
  -- Per-entry bound: `‖(applyMatrix A x) i j‖₊ ≤ ∑ rest, ‖A (cons i (cons j rest))‖₊`.
  have hentry : ∀ i j : Fin N,
      ‖applyMatrix A x i j‖₊
        ≤ ∑ rest : Fin D → Fin N, ‖A (Fin.cons i (Fin.cons j rest))‖₊ := by
    intro i j
    change ‖∑ rest : Fin D → Fin N, A (Fin.cons i (Fin.cons j rest)) * ∏ d : Fin D, x (rest d)‖₊
        ≤ ∑ rest : Fin D → Fin N, ‖A (Fin.cons i (Fin.cons j rest))‖₊
    refine le_trans (nnnorm_sum_le _ _) ?_
    refine Finset.sum_le_sum (fun rest _ => ?_)
    rw [nnnorm_mul]
    have hp1 : ‖∏ d : Fin D, x (rest d)‖₊ ≤ 1 := by
      rw [nnnorm_prod]; exact Finset.prod_le_one' (fun d _ => hxle (rest d))
    exact mul_le_of_le_one_right (zero_le _) hp1
  calc ∑ i, ∑ j, ‖applyMatrix A x i j‖₊
      ≤ ∑ i, ∑ j, ∑ rest : Fin D → Fin N, ‖A (Fin.cons i (Fin.cons j rest))‖₊ :=
        Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hentry i j))
    _ = ∑ idx : Fin (D + 2) → Fin N, ‖A idx‖₊ :=
        (sum_cons2 (fun idx => ‖A idx‖₊)).symm

/-- **Each eigenvalue of the contracted matrix is bounded by `∑|a|`.** -/
theorem abs_eigenvalue_le (A : CubicTensor ℝ (D + 2) N) (hA : A.IsSymmetric)
    {x : Fin N → ℝ} (hx : ∑ i, x i ^ 2 = 1) (i : Fin N) :
    |(applyMatrix_isHermitian A hA x).eigenvalues i|
      ≤ (∑ idx : Fin (D + 2) → Fin N, ‖A idx‖ : ℝ) := by
  set hM := applyMatrix_isHermitian A hA x with hMdef
  -- Each coordinate of the unit eigenvector has absolute value `≤ 1`.
  have hvle : ∀ k : Fin N, |(hM.eigenvectorBasis i) k| ≤ 1 := by
    intro k
    have hsum : ∑ l, (hM.eigenvectorBasis i) l ^ 2 = 1 := by
      have h := EuclideanSpace.real_norm_sq_eq (hM.eigenvectorBasis i)
      rw [hM.eigenvectorBasis.orthonormal.1 i, one_pow] at h
      exact h.symm
    have hk2 : (hM.eigenvectorBasis i) k ^ 2 ≤ 1 := by
      rw [← hsum]
      exact Finset.single_le_sum (fun l _ => sq_nonneg _) (mem_univ k)
    exact (sq_le_one_iff_abs_le_one _).mp hk2
  -- The eigenvalue is the Rayleigh quotient at its unit eigenvector.
  have heig : hM.eigenvalues i
      = ∑ k, ∑ l,
          (hM.eigenvectorBasis i) k * ((applyMatrix A x) k l * (hM.eigenvectorBasis i) l) := by
    rw [Matrix.IsHermitian.eigenvalues_eq hM i, RCLike.re_to_real]
    simp only [dotProduct, Matrix.mulVec, Pi.star_apply, star_trivial, Finset.mul_sum]
  rw [heig]
  -- Per-term bound: `|v_k · M_{kl} · v_l| ≤ |M_{kl}|`.
  have bound : ∀ k l : Fin N,
      |(hM.eigenvectorBasis i) k * ((applyMatrix A x) k l * (hM.eigenvectorBasis i) l)|
        ≤ |(applyMatrix A x) k l| := by
    intro k l
    rw [abs_mul, abs_mul]
    calc |(hM.eigenvectorBasis i) k| * (|(applyMatrix A x) k l| * |(hM.eigenvectorBasis i) l|)
        ≤ 1 * (|(applyMatrix A x) k l| * 1) := by gcongr <;> apply hvle
      _ = |(applyMatrix A x) k l| := by ring
  calc |∑ k, ∑ l,
            (hM.eigenvectorBasis i) k * ((applyMatrix A x) k l * (hM.eigenvectorBasis i) l)|
      ≤ ∑ k, |∑ l,
            (hM.eigenvectorBasis i) k * ((applyMatrix A x) k l * (hM.eigenvectorBasis i) l)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k, ∑ l,
            |(hM.eigenvectorBasis i) k * ((applyMatrix A x) k l * (hM.eigenvectorBasis i) l)| :=
        Finset.sum_le_sum (fun k _ => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ k, ∑ l, |(applyMatrix A x) k l| :=
        Finset.sum_le_sum (fun k _ => Finset.sum_le_sum (fun l _ => bound k l))
    _ ≤ (∑ idx : Fin (D + 2) → Fin N, ‖A idx‖ : ℝ) := by
        have key := sum_nnnorm_applyMatrix_le A hx
        have h1 : (∑ k, ∑ l, |(applyMatrix A x) k l| : ℝ)
            = ((∑ k, ∑ l, ‖(applyMatrix A x) k l‖₊ : ℝ≥0) : ℝ) := by
          rw [NNReal.coe_sum]
          refine Finset.sum_congr rfl (fun k _ => ?_)
          rw [NNReal.coe_sum]
          refine Finset.sum_congr rfl (fun l _ => ?_)
          rw [coe_nnnorm, Real.norm_eq_abs]
        have h2 : (∑ idx : Fin (D + 2) → Fin N, ‖A idx‖ : ℝ)
            = ((∑ idx : Fin (D + 2) → Fin N, ‖A idx‖₊ : ℝ≥0) : ℝ) := by
          rw [NNReal.coe_sum]
          refine Finset.sum_congr rfl (fun idx _ => ?_)
          rw [coe_nnnorm]
        rw [h1, h2, NNReal.coe_le_coe]
        exact key

/-- **Spectral radius of the contracted matrix is bounded by `∑|a|`.** -/
theorem spectralRadius_applyMatrix_le (A : CubicTensor ℝ (D + 2) N) (hA : A.IsSymmetric)
    {x : Fin N → ℝ} (hx : ∑ i, x i ^ 2 = 1) :
    spectralRadius ℝ (applyMatrix A x)
      ≤ ∑ idx : Fin (D + 2) → Fin N, (‖A idx‖₊ : ℝ≥0∞) := by
  unfold spectralRadius
  refine iSup₂_le (fun k hk => ?_)
  -- Real spectrum of a Hermitian matrix is the range of its eigenvalues.
  rw [(applyMatrix_isHermitian A hA x).spectrum_real_eq_range_eigenvalues] at hk
  obtain ⟨i, rfl⟩ := hk
  rw [← ENNReal.coe_finset_sum, ENNReal.coe_le_coe, ← NNReal.coe_le_coe,
      coe_nnnorm, Real.norm_eq_abs, NNReal.coe_sum]
  refine le_trans (abs_eigenvalue_le A hA hx i) ?_
  exact le_of_eq (Finset.sum_congr rfl (fun idx _ => (coe_nnnorm _).symm))

/-- **Lecture 8, lem:beta-bound.** For a symmetric tensor, `β(A) ≤ (D+1) · ∑ |a|`. -/
theorem beta_le (A : CubicTensor ℝ (D + 2) N) (hA : A.IsSymmetric) :
    beta A ≤ ((D : ℝ≥0∞) + 1) * ∑ idx : Fin (D + 2) → Fin N, (‖A idx‖₊ : ℝ≥0∞) := by
  rw [beta]
  gcongr
  exact iSup_le (fun x => spectralRadius_applyMatrix_le A hA x.2)

end CubicTensor
