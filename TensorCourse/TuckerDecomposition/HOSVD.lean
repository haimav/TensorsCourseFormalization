/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TuckerDecomposition.Defs
import TensorCourse.TuckerDecomposition.Matricized
import TensorCourse.TuckerDecomposition.NRank
import TensorCourse.Matrix.ColumnSpaceProjector
import TensorCourse.CPDecomposition.KhatriRao
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Higher-Order SVD: existence of an exact Tucker decomposition (`thm:hosvd`)

This file proves the existence half of the Tucker theory:

* `Tensor.exists_tucker_of_multirank` (**`thm:hosvd`**) — a tensor of
  multilinear rank `R` admits an exact Tucker decomposition with core of
  shape `R` and factor matrices with orthonormal columns.
* `Tensor.exists_orthTuckerDecomp_of_multirank` — the same existence
  statement, packaged as a `TuckerDecomp` object (`thm:hosvd` in the lecture's
  noun form, "`X` admits an orthonormal Tucker decomposition").
* `Tensor.ne_tuckerTensor_of_core_too_small` — the converse rank
  obstruction: if some core dimension is below the corresponding n-rank, no
  Tucker reconstruction of that shape can recover the tensor.
* `Tensor.nRank_le_cpRank_le_prod` — the rank sandwich
  `maxₙ rankₙ(X) ≤ cpRank X ≤ ∏ₙ rankₙ(X)`.

The existence theorem rests on the column-space orthonormal-projector lemma
B1 (`Matrix.exists_colspace_orthonormal_projector`), the Tucker composition
lemma `tuckerTensor_tuckerTensor`, and the matricized form
`tuckerTensor_modeUnfold` (all already established). No axioms beyond B1's
own (which is itself axiom-free).
-/

noncomputable section

open Finset BigOperators Matrix

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-! ### Helper: a Kronecker product of identities is the identity -/

/-- If every off-mode factor is the identity (and the row/column shapes
    agree, `I = J = dims`), then `kroneckerExcept` is the identity matrix.
    The product over modes `d ≠ n` of Kronecker deltas
    `δ((rem dims).symm ri d, (rem dims).symm cj d)` collapses to `δ(ri, cj)`
    since `remainingModesEquiv` is a bijection. -/
theorem kroneckerExcept_eq_one_of_off_one (n : Fin D)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (dims d)) 𝕜)
    (hU : ∀ d, d ≠ n → U d = 1) :
    kroneckerExcept U n = 1 := by
  funext ri cj
  rw [kroneckerExcept_apply, Matrix.one_apply]
  by_cases hrc : ri = cj
  · -- Diagonal: every factor is `1`, so the product is `1`.
    subst hrc
    rw [if_pos rfl]
    refine Finset.prod_eq_one (fun d hd => ?_)
    have hdn : d ≠ n := Finset.ne_of_mem_erase hd
    rw [dif_neg hdn, hU d hdn, Matrix.one_apply_eq]
  · -- Off-diagonal: `(rem dims).symm ri ≠ (rem dims).symm cj`, so some mode
    -- `d ≠ n` gives a zero factor.
    rw [if_neg hrc]
    obtain ⟨d₀, hd₀⟩ : ∃ d : {k : Fin D // k ≠ n},
        (remainingModesEquiv n dims).symm ri d ≠ (remainingModesEquiv n dims).symm cj d := by
      by_contra hc
      push Not at hc
      exact hrc ((remainingModesEquiv n dims).symm.injective (funext hc))
    obtain ⟨d₀, hd₀ne⟩ := d₀
    refine Finset.prod_eq_zero (Finset.mem_erase.mpr ⟨hd₀ne, Finset.mem_univ d₀⟩) ?_
    rw [dif_neg hd₀ne, hU d₀ hd₀ne, Matrix.one_apply_ne hd₀]

/-! ### A single square factor that fixes `X` leaves `X` unchanged -/

/-- The square single-mode factor list: `P` at mode `n`, identity elsewhere,
    all of shape `Matrix (Fin (dims d)) (Fin (dims d)) 𝕜`. Unlike
    `singleModeFactor`, this stays *square* (no shape change), so the Tucker
    product against it lands back in `Tensor 𝕜 D dims` with no cast. -/
def squareSingleFactor (n : Fin D)
    (P : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜) :
    (d : Fin D) → Matrix (Fin (dims d)) (Fin (dims d)) 𝕜 :=
  fun d => if h : d = n then h ▸ P else 1

@[simp] theorem squareSingleFactor_self (n : Fin D)
    (P : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜) :
    squareSingleFactor n P n = P := by
  simp only [squareSingleFactor, dif_pos]

theorem squareSingleFactor_of_ne (n : Fin D)
    (P : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜) {d : Fin D} (hd : d ≠ n) :
    squareSingleFactor n P d = 1 := by
  simp only [squareSingleFactor, dif_neg hd]

/-- **One square factor fixing `X` leaves the Tucker reconstruction equal to
    `X`.** If `(P n) * X_(n) = X_(n)` (i.e. `P n` fixes the mode-`n`
    matricization), then the Tucker product with `P n` on mode `n` and
    identities elsewhere equals `X`. Proved by matricization: the mode-`n`
    unfolding of the result is `P n * X_(n) * 1 = X_(n)`. -/
theorem tuckerTensor_squareSingleFactor_eq_self (n : Fin D)
    (P : Matrix (Fin (dims n)) (Fin (dims n)) 𝕜) (X : Tensor 𝕜 D dims)
    (hP : P * X.modeUnfold n = X.modeUnfold n) :
    tuckerTensor X (squareSingleFactor n P) = X := by
  apply modeUnfold_injective n
  rw [tuckerTensor_modeUnfold, squareSingleFactor_self,
    kroneckerExcept_eq_one_of_off_one n _ (fun d hd => squareSingleFactor_of_ne n P hd),
    Matrix.transpose_one, Matrix.mul_one, hP]

/-! ### The inductive lemma: many square factors fixing `X` leave `X` -/

/-- **All square factors fixing `X` leave the Tucker reconstruction equal to
    `X`.** If each square factor `P n` fixes the mode-`n` matricization of
    `X` (`(P n) * X_(n) = X_(n)`), then `tuckerTensor X P = X`.

    Proved by induction over the finset of *active* modes (where `P d ≠ 1`),
    peeling one mode at a time via the composition lemma
    `tuckerTensor_tuckerTensor` and `tuckerTensor_squareSingleFactor_eq_self`.
    We phrase the induction over an arbitrary finset `s` of modes, replacing
    `P d` by `1` off `s`. -/
theorem tuckerTensor_eq_self_of_fix
    (P : (d : Fin D) → Matrix (Fin (dims d)) (Fin (dims d)) 𝕜)
    (X : Tensor 𝕜 D dims)
    (hP : ∀ n, (P n) * X.modeUnfold n = X.modeUnfold n) :
    tuckerTensor X P = X := by
  -- Generalize over the active-mode finset `s`; the goal is the `s = univ` case.
  suffices h : ∀ s : Finset (Fin D),
      tuckerTensor X (fun d => if d ∈ s then P d else 1) = X by
    have := h Finset.univ
    simpa using this
  intro s
  induction s using Finset.induction with
  | empty =>
    -- No active modes: all factors are `1`.
    simp only [Finset.notMem_empty, if_false]
    exact tuckerTensor_one X
  | @insert m s' hm ih =>
    -- The factor list for `insert m s'` is `(squareSingleFactor m (P m)) · (list for s')`.
    have hsplit : (fun d => if d ∈ insert m s' then P d else 1) =
        (fun d => squareSingleFactor m (P m) d *
          (fun d => if d ∈ s' then P d else (1 : Matrix (Fin (dims d)) (Fin (dims d)) 𝕜)) d) := by
      funext d
      by_cases hdm : d = m
      · subst hdm
        simp only [squareSingleFactor_self, Finset.mem_insert, true_or, if_true,
          if_neg hm, Matrix.mul_one]
      · simp only [squareSingleFactor_of_ne m (P m) hdm, Matrix.one_mul,
          Finset.mem_insert, hdm, false_or]
    rw [hsplit, ← tuckerTensor_tuckerTensor, ih]
    -- Now `tuckerTensor X (squareSingleFactor m (P m)) = X`.
    exact tuckerTensor_squareSingleFactor_eq_self m (P m) X (hP m)

/-! ### `thm:hosvd`: existence of an exact Tucker decomposition -/

/-- **Theorem `thm:hosvd` (existence).** A tensor `X` of multilinear rank `R`
    admits an exact Tucker decomposition with a core `S` of shape `R` and
    factor matrices `U` with orthonormal columns.

    Proof: column-space lemma B1 gives, for each mode `n`, a matrix `U n`
    with `R n = nRank n X` columns, orthonormal columns `(U n)ᴴ (U n) = 1`,
    and projector `U n (U n)ᴴ` fixing `X_(n)`. Set the core
    `S := tuckerTensor X (fun d => (U d)ᴴ)`. Then by the composition lemma
    `tuckerTensor S U = tuckerTensor X (fun d => (U d) (U d)ᴴ)`, and the
    inductive fixedness lemma `tuckerTensor_eq_self_of_fix` collapses this to
    `X` since each projector `U n (U n)ᴴ` fixes the mode-`n` matricization. -/
theorem exists_tucker_of_multirank {D : ℕ} {dims : Fin D → ℕ}
    (X : Tensor 𝕜 D dims) (R : Fin D → ℕ) (hX : X.HasMultirank R) :
    ∃ (S : Tensor 𝕜 D R)
      (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (R d)) 𝕜),
      (∀ d, (U d)ᴴ * (U d) = 1) ∧ X = tuckerTensor S U := by
  -- For each mode, B1 gives a factor `U n` with `R n` columns.
  -- First rewrite `M.rank = R n` so B1's output lands in the right column type.
  have hB1 : ∀ n : Fin D, ∃ U : Matrix (Fin (dims n)) (Fin (R n)) 𝕜,
      Uᴴ * U = 1 ∧ U * Uᴴ * X.modeUnfold n = X.modeUnfold n := by
    intro n
    have hrk : (X.modeUnfold n).rank = R n := hX n
    obtain ⟨U, hU1, hU2⟩ := Matrix.exists_colspace_orthonormal_projector (X.modeUnfold n)
    -- Transport `U`'s columns along `(X.modeUnfold n).rank = R n` via `finCongr`.
    set c : Fin ((X.modeUnfold n).rank) ≃ Fin (R n) := finCongr hrk with hc
    refine ⟨U.submatrix id c.symm, ?_, ?_⟩
    · -- `(U')ᴴ U' = (Uᴴ U).submatrix c.symm c.symm = (1).submatrix c.symm c.symm = 1`.
      ext a b
      rw [Matrix.mul_apply, Matrix.one_apply]
      have : (∑ i : Fin (dims n), (U.submatrix id c.symm)ᴴ a i * (U.submatrix id c.symm) i b)
          = (Uᴴ * U) (c.symm a) (c.symm b) := by
        rw [Matrix.mul_apply]
        refine Finset.sum_congr rfl (fun i _ => ?_)
        simp only [Matrix.conjTranspose_submatrix, Matrix.submatrix_apply, id]
      rw [this, hU1, Matrix.one_apply]
      by_cases hab : a = b
      · simp [hab]
      · rw [if_neg hab, if_neg (fun h => hab (c.symm.injective h))]
    · -- `U' (U')ᴴ X_(n) = (U Uᴴ).submatrix id id X_(n) = U Uᴴ X_(n) = X_(n)`.
      have hUUH : (U.submatrix id c.symm) * (U.submatrix id c.symm)ᴴ = U * Uᴴ := by
        rw [Matrix.conjTranspose_submatrix,
          Matrix.submatrix_mul_equiv U Uᴴ id c.symm id, Matrix.submatrix_id_id]
      rw [hUUH, hU2]
  -- Choose the factor matrices.
  choose U hUorth hUfix using hB1
  -- The HOSVD core.
  refine ⟨tuckerTensor X (fun d => (U d)ᴴ), U, hUorth, ?_⟩
  -- `tuckerTensor (core) U = tuckerTensor X (fun d => U d * (U d)ᴴ)`.
  rw [tuckerTensor_tuckerTensor]
  -- The product of projectors fixes `X` (inductive lemma).
  symm
  exact tuckerTensor_eq_self_of_fix (fun d => U d * (U d)ᴴ) X (fun n => hUfix n)

/-- **`thm:hosvd`, packaged.** Restates `exists_tucker_of_multirank` with the
    decomposition bundled as a `TuckerDecomp` object: a tensor of multilinear
    rank `R` admits an orthonormal-factor Tucker decomposition `tk` (core of
    shape `R`, factors with orthonormal columns) whose reconstruction is `X`.
    This is the lecture's existence claim in its noun form — "`X` admits a
    Tucker decomposition of size `(R₁,…,R_D)` with orthogonal factor matrices."
    A one-line wrapper over `exists_tucker_of_multirank`. -/
theorem exists_orthTuckerDecomp_of_multirank {D : ℕ} {dims : Fin D → ℕ}
    (X : Tensor 𝕜 D dims) (R : Fin D → ℕ) (hX : X.HasMultirank R) :
    ∃ tk : TuckerDecomp 𝕜 D dims R, tk.IsOrthonormal ∧ tk.tensor = X := by
  obtain ⟨S, U, hU, hXSU⟩ := exists_tucker_of_multirank X R hX
  exact ⟨⟨S, U⟩, hU, hXSU.symm⟩

/-! ### Converse: a too-small core cannot recover `X` -/

/-- **Converse rank obstruction.** If `X` has multilinear rank `R` and some
    core dimension `J n` is strictly below `R n`, then no Tucker
    reconstruction `tuckerTensor S U` with core shape `J` can equal `X`.

    Proof: were `X = tuckerTensor S U`, the matricized form gives
    `X_(n) = U n * S_(n) * (kroneckerExcept U n)ᵀ`, whose rank is at most
    `J n` (the inner factor `S_(n)` has `J n` rows), contradicting
    `rank X_(n) = R n > J n`. -/
theorem ne_tuckerTensor_of_core_too_small {D : ℕ} {dims J : Fin D → ℕ}
    (X : Tensor 𝕜 D dims) (R : Fin D → ℕ) (hX : X.HasMultirank R)
    (S : Tensor 𝕜 D J)
    (U : (d : Fin D) → Matrix (Fin (dims d)) (Fin (J d)) 𝕜) (n : Fin D)
    (h : J n < R n) :
    X ≠ tuckerTensor S U := by
  intro heq
  -- The matricization has rank ≤ J n.
  have hmat : X.modeUnfold n = U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ := by
    rw [heq, tuckerTensor_modeUnfold]
  have hrank_le : (X.modeUnfold n).rank ≤ J n := by
    rw [hmat]
    calc (U n * S.modeUnfold n * (kroneckerExcept U n)ᵀ).rank
        ≤ (U n * S.modeUnfold n).rank := Matrix.rank_mul_le_left _ _
      _ ≤ (S.modeUnfold n).rank := Matrix.rank_mul_le_right _ _
      _ ≤ J n := Matrix.rank_le_height _
  -- But `rank X_(n) = R n > J n`.
  have hrk : (X.modeUnfold n).rank = R n := hX n
  rw [hrk] at hrank_le
  exact absurd hrank_le (not_le.mpr h)

/-! ### Rank sandwich `maxₙ rankₙ ≤ cpRank ≤ ∏ₙ rankₙ` -/

/-- **Rank sandwich.** For an order-`D` tensor with `0 < D`,
    `rankₙ(X) ≤ cpRank X` for every mode `n`, and `cpRank X ≤ ∏ₙ rankₙ(X)`.

    Lower bound: a rank-`r` CP decomposition matricizes (via
    `cp_modeUnfold_eq_khatriRao`) to a product whose mode-`n` unfolding has
    rank `≤ r`, so `rankₙ(X) ≤ r`; minimizing over `r` gives
    `rankₙ(X) ≤ cpRank X`. Upper bound: HOSVD existence
    (`exists_tucker_of_multirank` at `R = nRank`) produces a Tucker
    decomposition of core shape `nRank`, whose CP-rank is at most
    `∏ₙ rankₙ(X)` by `cpRank_tuckerTensor_le`. -/
theorem nRank_le_cpRank_le_prod {D : ℕ} {dims : Fin D → ℕ}
    (X : Tensor 𝕜 D dims) (hpos : 0 < D) :
    (∀ n, X.nRank n ≤ cpRank X) ∧ cpRank X ≤ ∏ n, X.nRank n := by
  constructor
  · -- Lower bound: `rankₙ(X) ≤ cpRank X`.
    intro n
    -- Unfold `cpRank` as an infimum; bound it below by `rankₙ`.
    -- It suffices to show: for any CP decomposition of rank `r` of `X`,
    -- `rankₙ(X) ≤ r`. Then `rankₙ(X)` is a lower bound for the infimum set,
    -- hence `≤ sInf = cpRank X` (which is `0` if the set is empty, but the
    -- set is nonempty — `X` always has *some* CP decomposition).
    rw [nRank_def, cpRank]
    apply le_csInf
    · -- The CP-rank set is nonempty: the full decomposition exists.
      obtain ⟨r, cp, hcp⟩ := exists_cpDecomp hpos X
      exact ⟨r, cp, hcp⟩
    · rintro r ⟨cp, hcp⟩
      -- `X_(n) = cp.factorMatrix n * (khatriRaoExcept …)ᵀ`, rank ≤ r.
      have hmat : X.modeUnfold n =
          cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ := by
        rw [← hcp, cp_modeUnfold_eq_khatriRao]
      rw [hmat]
      calc (cp.factorMatrix n * (khatriRaoExcept cp.factorMatrix n)ᵀ).rank
          ≤ (cp.factorMatrix n).rank := Matrix.rank_mul_le_left _ _
        _ ≤ r := Matrix.rank_le_width _
  · -- Upper bound: `cpRank X ≤ ∏ₙ rankₙ(X)`.
    -- HOSVD existence at `R = nRank` gives a Tucker decomposition; bound its
    -- CP-rank via `cpRank_tuckerTensor_le`.
    obtain ⟨S, U, _, hXSU⟩ := exists_tucker_of_multirank X X.nRank (fun n => rfl)
    -- Rewrite only the left-hand `cpRank X` (not the `nRank` on the right).
    conv_lhs => rw [hXSU]
    exact cpRank_tuckerTensor_le hpos S U

end Tensor

end
