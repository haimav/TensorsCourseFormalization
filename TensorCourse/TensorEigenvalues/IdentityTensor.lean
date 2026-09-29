/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorEigenvalues.Shift
import TensorCourse.Tensor.Transpose
import Mathlib.GroupTheory.Perm.Fin
import Mathlib.Data.Fintype.Perm

/-!
# Existence of an Identity-on-Sphere Tensor for Even Orders (Lecture 8)

Lecture 8 introduces, for an **even** order `D` and dimension `N`, a tensor
`E_{D,N}` that acts like the identity on the unit sphere:
`E_{D,N} 𝐱 = 𝐱` whenever `‖𝐱‖₂ = 1`. This file constructs such a tensor over
`ℝ` and proves the existence statement.

Working over `ℝ` is essential: the pairing of two modes by a Kronecker delta
contracts to `∑ᵢ xᵢ²`, which equals `∑ᵢ ‖xᵢ‖² = 1` on the unit sphere only for
real vectors. Over `ℂ` no identity-on-sphere tensor exists.

## Construction

We take the *pairing* tensor `pairId m` of order `2m + 2`, pairing consecutive
modes `(0,1), (2,3), …, (2m, 2m+1)` by Kronecker deltas:

  `pairId m (i₁, …, i_{2m+2}) = ∏_{pairs} δ`.

It is defined recursively: `pairId 0` is the `N × N` identity matrix (as an
order-2 tensor), and `pairId (m+1)` multiplies the delta on the first pair by
`pairId m` on the remaining modes. (This is one term — a single perfect
matching — of the lecture's symmetrised `E_{D,N}`; every perfect matching is
identity-on-sphere, so it suffices.)

## Main results

* `CubicTensor.pairId` — the pairing tensor.
* `CubicTensor.fullContract_eq` — the complete contraction of `pairId m` with a
  vector `x` equals `(∑ᵢ xᵢ²)^{m+1}`.
* `CubicTensor.apply_pairId` — `pairId m ⬝ₜ x = x · (∑ᵢ xᵢ²)^m` pointwise.
* `CubicTensor.identityOnSphere_pairId` — `pairId m` is identity-on-sphere.
* `CubicTensor.exists_identityOnSphere_even` — the existence statement: for
  every even order `2m + 2` there is an identity-on-sphere tensor.
-/

noncomputable section

open Finset BigOperators

namespace CubicTensor

variable {N : ℕ}

/-! ### Reindexing helpers -/

/-- Reindex a sum over `Fin (D+1) → Fin N` as a double sum over the first index
    and the remaining `D` indices, via `Fin.consEquiv`. -/
private theorem sum_cons1 {M : Type*} [AddCommMonoid M] {D : ℕ}
    (f : (Fin (D + 1) → Fin N) → M) :
    ∑ rest : Fin (D + 1) → Fin N, f rest
      = ∑ a : Fin N, ∑ t : Fin D → Fin N, f (Fin.cons a t) := by
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) f,
      Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  refine Finset.sum_congr rfl (fun t _ => ?_)
  have hc : (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) (a, t) = Fin.cons a t := by
    funext d; exact Fin.consEquiv_apply _ _ _
  rw [hc]

/-- Reindex a sum over `Fin (D+2) → Fin N` as a triple sum over the first two
    indices and the remaining `D` indices, via a double `Fin.consEquiv`. -/
private theorem sum_cons2 {M : Type*} [AddCommMonoid M] {D : ℕ}
    (f : (Fin (D + 2) → Fin N) → M) :
    ∑ idx : Fin (D + 2) → Fin N, f idx
      = ∑ a : Fin N, ∑ b : Fin N, ∑ rest : Fin D → Fin N,
          f (Fin.cons a (Fin.cons b rest)) := by
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) f,
      Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  have step : ∀ tail : Fin (D + 1) → Fin N,
      (Fin.consEquiv (fun _ : Fin (D + 2) => Fin N)) (a, tail) = Fin.cons a tail := by
    intro tail; funext d; exact Fin.consEquiv_apply _ _ _
  simp only [step]
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N))
        (fun tail => f (Fin.cons a tail)), Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun b _ => ?_)
  refine Finset.sum_congr rfl (fun rest _ => ?_)
  have hc : (Fin.consEquiv (fun _ : Fin (D + 1) => Fin N)) (b, rest) = Fin.cons b rest := by
    funext d; exact Fin.consEquiv_apply _ _ _
  rw [hc]

/-- Collapse a Kronecker-delta pair sum: `∑ₐ ∑_b δ_{ab} · xₐ · x_b = ∑ₐ xₐ²`. -/
private theorem sum_pair_delta (x : Fin N → ℝ) :
    ∑ a : Fin N, ∑ b : Fin N, (if a = b then (1 : ℝ) else 0) * (x a * x b)
      = ∑ a, x a ^ 2 := by
  refine Finset.sum_congr rfl (fun a _ => ?_)
  have h : ∀ b, (if a = b then (1 : ℝ) else 0) * (x a * x b)
      = if a = b then x a * x b else 0 := by
    intro b; rw [ite_mul, one_mul, zero_mul]
  simp only [h]
  rw [Finset.sum_ite_eq univ a (fun b => x a * x b)]
  simp [sq]

/-- Collapse a single Kronecker-delta sum: `∑ₐ δ_{ia} · xₐ = xᵢ`. -/
private theorem sum_single_delta (x : Fin N → ℝ) (i : Fin N) :
    ∑ a : Fin N, (if i = a then (1 : ℝ) else 0) * x a = x i := by
  have h : ∀ a, (if i = a then (1 : ℝ) else 0) * x a = if i = a then x a else 0 := by
    intro a; rw [ite_mul, one_mul, zero_mul]
  simp only [h]
  rw [Finset.sum_ite_eq univ i (fun a => x a)]
  simp

/-! ### The pairing tensor -/

/-- **The pairing tensor `pairId m`** of order `2m + 2`. It pairs consecutive
    modes `(0,1), (2,3), …, (2m, 2m+1)` by Kronecker deltas. Defined recursively:
    `pairId 0` is the `N × N` identity (order-2 tensor); `pairId (m+1)` is the
    delta on the first pair times `pairId m` on the remaining modes. Here
    `Fin.tail idx 0 = idx 1` gives the partner of mode `0`. -/
def pairId : (m : ℕ) → CubicTensor ℝ (2 * m + 2) N
  | 0 => fun idx => if idx 0 = Fin.tail idx 0 then (1 : ℝ) else 0
  | (m + 1) => fun idx =>
      (if idx 0 = Fin.tail idx 0 then (1 : ℝ) else 0) *
        pairId m (Fin.tail (Fin.tail idx))

theorem pairId_zero_apply (idx : Fin 2 → Fin N) :
    (pairId 0 : CubicTensor ℝ 2 N) idx = if idx 0 = Fin.tail idx 0 then (1 : ℝ) else 0 :=
  rfl

theorem pairId_succ_apply (m : ℕ) (idx : Fin (2 * (m + 1) + 2) → Fin N) :
    pairId (m + 1) idx
      = (if idx 0 = Fin.tail idx 0 then (1 : ℝ) else 0) *
          pairId m (Fin.tail (Fin.tail idx)) :=
  rfl

/-! ### The complete contraction -/

/-- **Complete contraction of `pairId m`.** Contracting `pairId m` with `x` in
    all `2m + 2` modes yields `(∑ᵢ xᵢ²)^{m+1}`. Proved by induction on `m`,
    peeling the first pair of modes at each step. -/
theorem fullContract_eq (m : ℕ) (x : Fin N → ℝ) :
    (∑ idx : Fin (2 * m + 2) → Fin N,
        pairId m idx * ∏ d : Fin (2 * m + 2), x (idx d))
      = (∑ a, x a ^ 2) ^ (m + 1) := by
  induction m with
  | zero =>
      rw [sum_cons2 (fun idx => (pairId 0 : CubicTensor ℝ 2 N) idx
              * ∏ d : Fin 2, x (idx d))]
      have key : ∀ a b : Fin N, ∀ rest : Fin 0 → Fin N,
          (pairId 0 : CubicTensor ℝ 2 N) (Fin.cons a (Fin.cons b rest))
              * ∏ d : Fin 2, x ((Fin.cons a (Fin.cons b rest) : Fin 2 → Fin N) d)
            = (if a = b then (1 : ℝ) else 0) * (x a * x b) := by
        intro a b rest
        rw [pairId_zero_apply]
        simp only [Fin.cons_zero, Fin.tail_cons, Fin.prod_univ_succ, Fin.prod_univ_zero,
          Fin.cons_zero, Fin.cons_succ]
        ring
      simp only [key, Finset.sum_const, Finset.card_univ, mul_zero, Fintype.card_unique,
        one_smul]
      rw [sum_pair_delta]; ring
  | succ m ih =>
      rw [sum_cons2 (fun idx => pairId (m + 1) idx
              * ∏ d : Fin (2 * (m + 1) + 2), x (idx d))]
      have key : ∀ a b : Fin N, ∀ rest : Fin (2 * (m + 1)) → Fin N,
          pairId (m + 1) (Fin.cons a (Fin.cons b rest))
              * ∏ d : Fin (2 * (m + 1) + 2),
                  x ((Fin.cons a (Fin.cons b rest) : Fin (2 * (m + 1) + 2) → Fin N) d)
            = ((if a = b then (1 : ℝ) else 0) * (x a * x b)) *
                (pairId m rest * ∏ d : Fin (2 * (m + 1)), x (rest d)) := by
        intro a b rest
        rw [pairId_succ_apply]
        simp only [Fin.cons_zero, Fin.tail_cons, Fin.prod_univ_succ, Fin.cons_zero,
          Fin.cons_succ]
        ring
      simp only [key]
      have collapse : ∀ a b : Fin N,
          (∑ rest : Fin (2 * (m + 1)) → Fin N,
              ((if a = b then (1 : ℝ) else 0) * (x a * x b)) *
                (pairId m rest * ∏ d : Fin (2 * (m + 1)), x (rest d)))
            = ((if a = b then (1 : ℝ) else 0) * (x a * x b)) * (∑ c, x c ^ 2) ^ (m + 1) := by
        intro a b
        have hih : (∑ rest : Fin (2 * (m + 1)) → Fin N,
            pairId m rest * ∏ d : Fin (2 * (m + 1)), x (rest d))
              = (∑ c, x c ^ 2) ^ (m + 1) := ih
        rw [← Finset.mul_sum, hih]
      simp only [collapse, ← Finset.sum_mul]
      rw [sum_pair_delta, ← pow_succ']

/-! ### Applying the pairing tensor -/

/-- **`pairId m ⬝ₜ x = x · (∑ᵢ xᵢ²)^m`.** Applying `pairId m` to `x` (contracting
    the last `2m + 1` modes) leaves the free index paired to the contracted mode 1,
    forcing it to equal the free index and contributing `xᵢ`, while the remaining
    `m` pairs contract to `(∑ᵢ xᵢ²)^m` (via `fullContract_eq`). -/
theorem apply_pairId (m : ℕ) (x : Fin N → ℝ) (i : Fin N) :
    (pairId m ⬝ₜ x) i = x i * (∑ a, x a ^ 2) ^ m := by
  cases m with
  | zero =>
      rw [apply_apply,
          sum_cons1 (fun rest => (pairId 0 : CubicTensor ℝ 2 N) (Fin.cons i rest)
              * ∏ d : Fin 1, x (rest d))]
      have key : ∀ a : Fin N, ∀ t : Fin 0 → Fin N,
          (pairId 0 : CubicTensor ℝ 2 N) (Fin.cons i (Fin.cons a t))
              * ∏ d : Fin 1, x ((Fin.cons a t : Fin 1 → Fin N) d)
            = (if i = a then (1 : ℝ) else 0) * x a := by
        intro a t
        rw [pairId_zero_apply]
        simp only [Fin.cons_zero, Fin.tail_cons, Fin.prod_univ_succ, Fin.prod_univ_zero,
          Fin.cons_zero]
        ring
      simp only [key, Finset.sum_const, Finset.card_univ, Fintype.card_unique, one_smul]
      rw [sum_single_delta, pow_zero, mul_one]
  | succ m =>
      rw [apply_apply,
          sum_cons1 (fun rest => pairId (m + 1) (Fin.cons i rest)
              * ∏ d : Fin (2 * (m + 1) + 1), x (rest d))]
      have key : ∀ a : Fin N, ∀ t : Fin (2 * (m + 1)) → Fin N,
          pairId (m + 1) (Fin.cons i (Fin.cons a t))
              * ∏ d : Fin (2 * (m + 1) + 1),
                  x ((Fin.cons a t : Fin (2 * (m + 1) + 1) → Fin N) d)
            = ((if i = a then (1 : ℝ) else 0) * x a) *
                (pairId m t * ∏ d : Fin (2 * (m + 1)), x (t d)) := by
        intro a t
        rw [pairId_succ_apply]
        simp only [Fin.cons_zero, Fin.tail_cons, Fin.prod_univ_succ, Fin.cons_zero,
          Fin.cons_succ]
        ring
      simp only [key]
      have collapse : ∀ a : Fin N,
          (∑ t : Fin (2 * (m + 1)) → Fin N,
              ((if i = a then (1 : ℝ) else 0) * x a) *
                (pairId m t * ∏ d : Fin (2 * (m + 1)), x (t d)))
            = ((if i = a then (1 : ℝ) else 0) * x a) * (∑ c, x c ^ 2) ^ (m + 1) := by
        intro a
        rw [← Finset.mul_sum]
        congr 1
        exact fullContract_eq m x
      simp only [collapse, ← Finset.sum_mul]
      rw [sum_single_delta]

/-! ### Identity on the sphere -/

/-- **`pairId m` is identity-on-sphere.** On the real unit sphere `∑ᵢ ‖xᵢ‖² = 1`,
    so `∑ᵢ xᵢ² = 1` and `pairId m ⬝ₜ x = x`. -/
theorem identityOnSphere_pairId (m : ℕ) :
    IdentityOnSphere (pairId m : CubicTensor ℝ (2 * m + 2) N) := by
  intro x hx
  have hsum : (∑ a, x a ^ 2) = 1 := by
    rw [← hx]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Real.norm_eq_abs, sq_abs]
  funext i
  rw [apply_pairId, hsum, one_pow, mul_one]

/-- **Existence of an identity-on-sphere tensor for even orders (Lecture 8).**
    For every even order `2m + 2` (which is `≥ 2`) there is a real tensor acting
    as the identity on the unit sphere — the lecture's `E_{D,N}`. -/
theorem exists_identityOnSphere_even (m : ℕ) :
    ∃ E : CubicTensor ℝ (2 * m + 2) N, IdentityOnSphere E :=
  ⟨pairId m, identityOnSphere_pairId m⟩

/-! ### The symmetric identity tensor `E_{D,N}`

The lecture's `E_{D,N}` is the *symmetrisation* of the single perfect matching
`pairId m`: the average over all mode permutations of `pairId m`. We show it is
still identity-on-sphere. The heart of the argument is that *every* permuted
copy `(pairId m).transpose σ` is individually identity-on-sphere (every perfect
matching contracts the free mode's partner to the free index and the rest to
`(∑ᵢ xᵢ²)ᵐ`), so the average is too. -/

/-! #### Linearity of tensor apply and transpose over finite sums -/

/-- Tensor apply distributes over a finite sum of tensors. -/
theorem apply_sum_T {ι : Type*} {E : ℕ} (s : Finset ι)
    (T : ι → CubicTensor ℝ (E + 1) N) (x : Fin N → ℝ) :
    (∑ j ∈ s, T j) ⬝ₜ x = ∑ j ∈ s, (T j ⬝ₜ x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert _ _ hj ih => rw [Finset.sum_insert hj, apply_add_T, ih, Finset.sum_insert hj]

/-- Transpose distributes over a finite sum of cubic tensors. -/
theorem transpose_sum {ι : Type*} {O : ℕ} (s : Finset ι)
    (U : ι → CubicTensor ℝ O N) (τ : Equiv.Perm (Fin O)) :
    (∑ j ∈ s, U j).transpose τ = ∑ j ∈ s, (U j).transpose τ := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert _ _ hj ih =>
      rw [Finset.sum_insert hj, Tensor.transpose_add, ih, Finset.sum_insert hj]

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

/-! #### Invariance of the free-mode contraction under permuting contracted modes -/

/-- **Contracting the non-free modes is invariant under any permutation fixing
    `0`.** For a cubic tensor `T` of order `E + 1` and a permutation `κ` of the
    modes with `κ 0 = 0`, precomposing the index by `κ` before contracting the
    last `E` modes with `x` leaves the sum unchanged. Proof: `κ` restricts to a
    permutation `ρ` of the contracted modes (via `Equiv.Perm.decomposeFin`), and
    both the tensor value and the product `∏ x` are invariant under reindexing
    the contracted modes by `ρ`. -/
private theorem sum_free_perm0 {E : ℕ} (T : CubicTensor ℝ (E + 1) N)
    (x : Fin N → ℝ) (i : Fin N) (κ : Equiv.Perm (Fin (E + 1))) (hκ : κ 0 = 0) :
    (∑ g : Fin E → Fin N,
        T (fun d => (Fin.cons i g : Fin (E + 1) → Fin N) (κ d)) * ∏ d : Fin E, x (g d))
      = ∑ g : Fin E → Fin N, T (Fin.cons i g) * ∏ d : Fin E, x (g d) := by
  set ρ : Equiv.Perm (Fin E) := (Equiv.Perm.decomposeFin κ).2 with hρ
  have hfst : (Equiv.Perm.decomposeFin κ).1 = κ 0 := by
    have h := Equiv.Perm.decomposeFin_symm_apply_zero
        (Equiv.Perm.decomposeFin κ).1 (Equiv.Perm.decomposeFin κ).2
    rw [Prod.mk.eta, Equiv.symm_apply_apply] at h
    exact h.symm
  have hκρ : κ = Equiv.Perm.decomposeFin.symm (0, ρ) := by
    have h1 : Equiv.Perm.decomposeFin κ = (0, ρ) := by
      rw [hρ]
      have : (Equiv.Perm.decomposeFin κ).1 = (0 : Fin (E + 1)) := by rw [hfst, hκ]
      rw [← this]
    rw [← h1, Equiv.symm_apply_apply]
  have hcons : ∀ g : Fin E → Fin N,
      (fun d => (Fin.cons i g : Fin (E + 1) → Fin N) (κ d))
        = (Fin.cons i (fun k => g (ρ k)) : Fin (E + 1) → Fin N) := by
    intro g
    funext d
    induction d using Fin.cases with
    | zero => rw [hκ]; simp
    | succ e =>
        rw [hκρ, Equiv.Perm.decomposeFin_symm_apply_succ, Equiv.swap_self]
        simp
  have hterm : ∀ g : Fin E → Fin N,
      T (fun d => (Fin.cons i g : Fin (E + 1) → Fin N) (κ d)) * ∏ d : Fin E, x (g d)
        = T (Fin.cons i (fun k => g (ρ k))) * ∏ d : Fin E, x ((fun k => g (ρ k)) d) := by
    intro g
    rw [hcons g]
    congr 1
    exact (Equiv.prod_comp ρ (fun k => x (g k))).symm
  simp_rw [hterm]
  exact Equiv.sum_comp (Equiv.arrowCongr ρ.symm (Equiv.refl (Fin N)))
    (fun h => T (Fin.cons i h) * ∏ d : Fin E, x (h d))

/-! #### The generalized free-mode contraction of `pairId` -/

/-- Collapse a single Kronecker-delta sum with the delta in the form `a = i`:
    `∑ₐ δ_{ai} · xₐ = xᵢ`. -/
private theorem sum_single_delta' (x : Fin N → ℝ) (i : Fin N) :
    ∑ a : Fin N, (if a = i then (1 : ℝ) else 0) * x a = x i := by
  have h : ∀ a, (if a = i then (1 : ℝ) else 0) * x a = if a = i then x a else 0 := by
    intro a; rw [ite_mul, one_mul, zero_mul]
  simp only [h]
  rw [Finset.sum_ite_eq' univ i (fun a => x a)]
  simp

/-- Peeling the first mode of a complete contraction of `pairId m`:
    `∑_b ∑_rest pairId m (b :: rest) · (x_b · ∏ x_rest) = (∑ xⱼ²)^{m+1}`. This is
    `fullContract_eq` written with the first summed mode split off. -/
private theorem tailContract (x : Fin N → ℝ) (m : ℕ) :
    (∑ b : Fin N, ∑ rest : Fin (2 * m + 1) → Fin N,
        pairId m (Fin.cons b rest) * (x b * ∏ d : Fin (2 * m + 1), x (rest d)))
      = (∑ a, x a ^ 2) ^ (m + 1) := by
  have key : ∀ (b : Fin N) (rest : Fin (2 * m + 1) → Fin N),
      pairId m (Fin.cons b rest) * (x b * ∏ d : Fin (2 * m + 1), x (rest d))
        = pairId m (Fin.cons b rest)
            * ∏ d : Fin (2 * m + 1 + 1), x ((Fin.cons b rest : Fin (2 * m + 1 + 1) → Fin N) d) := by
    intro b rest
    conv_rhs => rw [Fin.prod_univ_succ, Fin.cons_zero]
    refine congrArg (fun P => pairId m (Fin.cons b rest) * (x b * P)) ?_
    exact Finset.prod_congr rfl (fun d _ => by rw [Fin.cons_succ])
  simp_rw [key]
  exact (sum_cons1 (fun g => pairId m g * ∏ d : Fin (2 * m + 1 + 1), x (g d))).symm.trans
    (fullContract_eq m x)

/-- **Free-mode contraction of `pairId m` at an arbitrary mode `p`.** Contracting
    `pairId m` with `x` in all modes except mode `p` (which is fixed to the free
    index `i`) yields `xᵢ · (∑ⱼ xⱼ²)ᵐ`, independently of which mode `p` is free.
    This generalises `apply_pairId` (the `p = 0` case). Proved by induction on
    `m`, splitting on whether the free mode lies in the first Kronecker pair
    (`p ∈ {0, 1}`) or in the tail (`p ≥ 2`). -/
theorem contractAt_pairId (x : Fin N → ℝ) :
    ∀ (m : ℕ) (p : Fin (2 * m + 2)) (i : Fin N),
      (∑ g : Fin (2 * m + 1) → Fin N,
          pairId m (Fin.insertNth p i g) * ∏ d : Fin (2 * m + 1), x (g d))
        = x i * (∑ a, x a ^ 2) ^ m := by
  intro m
  induction m with
  | zero =>
      intro p i
      induction p using Fin.cases with
      | zero =>
          -- `p = 0`: `insertNth 0 i g = cons i g`, so this is `apply_pairId`.
          simp only [Fin.insertNth_zero']
          rw [← apply_apply]
          exact apply_pairId _ x i
      | succ q =>
          induction q using Fin.cases with
          | zero =>
              -- `p = 1`, `m = 0`.
              rw [sum_cons1 (fun g => pairId 0 (Fin.insertNth (Fin.succ 0) i g)
                    * ∏ d : Fin 1, x (g d))]
              have key : ∀ (a : Fin N) (t : Fin 0 → Fin N),
                  pairId 0 (Fin.insertNth (Fin.succ 0) i (Fin.cons a t))
                      * ∏ d : Fin 1, x ((Fin.cons a t : Fin 1 → Fin N) d)
                    = (if a = i then (1 : ℝ) else 0) * x a := by
                intro a t
                rw [Fin.insertNth_succ_cons, Fin.insertNth_zero', pairId_zero_apply]
                simp only [Fin.cons_zero, Fin.tail_cons, Fin.prod_univ_succ,
                  Fin.prod_univ_zero, Fin.cons_zero]
                ring
              simp only [key, Finset.sum_const, Finset.card_univ, Fintype.card_unique, one_smul]
              rw [sum_single_delta', pow_zero, mul_one]
          | succ r => exact r.elim0
  | succ m ih =>
      intro p i
      induction p using Fin.cases with
      | zero =>
          -- `p = 0`: `insertNth 0 i g = cons i g`, so this is `apply_pairId`.
          simp only [Fin.insertNth_zero']
          rw [← apply_apply]
          exact apply_pairId _ x i
      | succ q =>
          induction q using Fin.cases with
          | zero =>
              -- `p = 1`: the free index sits in the first Kronecker pair (as its
              -- partner), the rest fully contracts.
              refine Eq.trans (sum_cons2 (fun g => pairId (m + 1) (Fin.insertNth (Fin.succ 0) i g)
                    * ∏ d : Fin (2 * m + 1 + 2), x (g d))) ?_
              have key : ∀ (a b : Fin N) (rest : Fin (2 * m + 1) → Fin N),
                  pairId (m + 1) (Fin.insertNth (Fin.succ 0) i
                      (Fin.cons a (Fin.cons b rest)))
                    * ∏ d : Fin (2 * m + 1 + 2),
                        x ((Fin.cons a (Fin.cons b rest) :
                          Fin (2 * m + 1 + 2) → Fin N) d)
                  = ((if a = i then (1 : ℝ) else 0) * x a) *
                      (pairId m (Fin.cons b rest) *
                        (x b * ∏ d : Fin (2 * m + 1), x (rest d))) := by
                intro a b rest
                rw [Fin.insertNth_succ_cons, Fin.insertNth_zero', pairId_succ_apply,
                  Fin.prod_univ_succ, Fin.prod_univ_succ]
                simp only [Fin.cons_zero, Fin.tail_cons, Fin.cons_succ]
                ring
              simp_rw [key]
              have step : ∀ a : Fin N,
                  (∑ b : Fin N, ∑ rest : Fin (2 * m + 1) → Fin N,
                      ((if a = i then (1 : ℝ) else 0) * x a) *
                        (pairId m (Fin.cons b rest) *
                          (x b * ∏ d : Fin (2 * m + 1), x (rest d))))
                    = ((if a = i then (1 : ℝ) else 0) * x a) * (∑ a, x a ^ 2) ^ (m + 1) := by
                intro a
                simp_rw [← Finset.mul_sum]
                rw [tailContract x m]
              refine Eq.trans (Finset.sum_congr rfl (fun a _ => step a)) ?_
              rw [← Finset.sum_mul, sum_single_delta']
          | succ r =>
              -- `p ≥ 2`: the free index sits in the tail; the first pair
              -- contracts to `∑ xⱼ²` and the tail reduces via the IH at `r`.
              refine Eq.trans (sum_cons2 (fun g => pairId (m + 1) (Fin.insertNth r.succ.succ i g)
                    * ∏ d : Fin (2 * m + 1 + 2), x (g d))) ?_
              have key : ∀ (a b : Fin N) (rest : Fin (2 * m + 1) → Fin N),
                  pairId (m + 1) (Fin.insertNth r.succ.succ i
                      (Fin.cons a (Fin.cons b rest)))
                    * ∏ d : Fin (2 * m + 1 + 2),
                        x ((Fin.cons a (Fin.cons b rest) :
                          Fin (2 * m + 1 + 2) → Fin N) d)
                  = ((if a = b then (1 : ℝ) else 0) * (x a * x b)) *
                      (pairId m (Fin.insertNth r i rest) *
                        ∏ d : Fin (2 * m + 1), x (rest d)) := by
                intro a b rest
                rw [Fin.insertNth_succ_cons, Fin.insertNth_succ_cons, pairId_succ_apply,
                  Fin.prod_univ_succ, Fin.prod_univ_succ]
                simp only [Fin.cons_zero, Fin.tail_cons, Fin.cons_succ]
                ring
              simp_rw [key]
              have step : ∀ a b : Fin N,
                  (∑ rest : Fin (2 * m + 1) → Fin N,
                      ((if a = b then (1 : ℝ) else 0) * (x a * x b)) *
                        (pairId m (Fin.insertNth r i rest) *
                          ∏ d : Fin (2 * m + 1), x (rest d)))
                    = ((if a = b then (1 : ℝ) else 0) * (x a * x b)) *
                        (x i * (∑ a, x a ^ 2) ^ m) := by
                intro a b
                rw [← Finset.mul_sum, ih r i]
              refine Eq.trans
                (Finset.sum_congr rfl (fun a _ => Finset.sum_congr rfl (fun b _ => step a b))) ?_
              simp_rw [← Finset.sum_mul]
              rw [sum_pair_delta]
              ring

/-! #### Every permuted matching is identity-on-sphere -/

/-- **Applying a permuted `pairId` gives `xᵢ · (∑ⱼ xⱼ²)ᵐ`.** For every mode
    permutation `σ`, `(pairId m).transpose σ ⬝ₜ x` equals `x · (∑ⱼ xⱼ²)ᵐ`
    pointwise. The transpose moves the free mode's partner but not the value of
    the contraction: reindexing the contracted modes reduces it to
    `contractAt_pairId` at the mode `σ 0`. -/
theorem apply_transpose_pairId (m : ℕ) (σ : Equiv.Perm (Fin (2 * m + 2)))
    (x : Fin N → ℝ) (i : Fin N) :
    ((pairId m).transpose σ ⬝ₜ x) i = x i * (∑ a, x a ^ 2) ^ m := by
  haveI : NeZero (2 * m + 2) := ⟨by omega⟩
  set A0 : CubicTensor ℝ (2 * m + 2) N := (pairId m).transpose σ with hA0
  set cr : Equiv.Perm (Fin (2 * m + 2)) := (σ 0).cycleRange with hcr
  set T' : CubicTensor ℝ (2 * m + 2) N := (pairId m).transpose cr.symm with hT'
  set β : Equiv.Perm (Fin (2 * m + 2)) := σ.symm * cr.symm with hβ
  have hβ0 : β 0 = 0 := by
    rw [hβ, Equiv.Perm.mul_apply, hcr, Fin.cycleRange_symm_zero, Equiv.symm_apply_apply]
  have hA : ∀ g : Fin (2 * m + 1) → Fin N,
      A0 (Fin.cons i g)
        = T' (fun d => (Fin.cons i g : Fin (2 * m + 2) → Fin N) (β d)) := by
    intro g
    rw [hA0, hT', cubicTranspose_apply cr.symm (pairId m), cubicTranspose_apply σ (pairId m)]
    congr 1
    funext e
    simp only [Equiv.symm_symm]
    congr 1
    rw [hβ, Equiv.Perm.mul_apply, Equiv.symm_apply_apply]
  have hB : ∀ g : Fin (2 * m + 1) → Fin N,
      T' (Fin.cons i g) = pairId m (Fin.insertNth (σ 0) i g) := by
    intro g
    rw [hT', cubicTranspose_apply cr.symm (pairId m)]
    congr 1
    simp only [Equiv.symm_symm, hcr]
    exact Fin.cons_comp_cycleRange i g (σ 0)
  rw [apply_apply]
  simp_rw [hA]
  rw [sum_free_perm0 T' x i β hβ0]
  simp_rw [hB]
  exact contractAt_pairId x m (σ 0) i

/-! #### The symmetric identity tensor and its properties -/

/-- **The symmetric identity tensor `E_{D,N}` (Lecture 8).** The average over all
    mode permutations of the single perfect matching `pairId m`. Explicitly,
    `E_{D,N}(idx) = (1/D!) ∑_{σ} ∏_j δ(idx_{σ(2j)}, idx_{σ(2j+1)})` — the fully
    symmetrised pairing tensor. -/
noncomputable def symId (m : ℕ) : CubicTensor ℝ (2 * m + 2) N :=
  (((2 * m + 2).factorial : ℝ)⁻¹) •
    ∑ σ : Equiv.Perm (Fin (2 * m + 2)), (pairId m).transpose σ

/-- **Pointwise formula for `E_{D,N}`.** `symId m idx` is the average over all
    mode permutations `σ` of `pairId m` evaluated at the permuted index
    `idx ∘ σ⁻¹` — matching the lecture's symmetrised Kronecker-pairing sum. -/
theorem symId_apply (m : ℕ) (idx : Fin (2 * m + 2) → Fin N) :
    symId m idx
      = (((2 * m + 2).factorial : ℝ)⁻¹) *
          ∑ σ : Equiv.Perm (Fin (2 * m + 2)), pairId m (fun d => idx (σ.symm d)) := by
  rw [symId]
  have h1 : ((((((2 * m + 2).factorial : ℝ)⁻¹) •
        ∑ σ : Equiv.Perm (Fin (2 * m + 2)), (pairId m).transpose σ) : CubicTensor ℝ (2 * m + 2) N)
          idx)
      = (((2 * m + 2).factorial : ℝ)⁻¹) *
          ∑ σ : Equiv.Perm (Fin (2 * m + 2)), ((pairId m).transpose σ) idx := by
    rw [Pi.smul_apply, smul_eq_mul,
      Finset.sum_apply idx Finset.univ (fun σ => (pairId m).transpose σ)]
  rw [h1]
  refine congrArg (fun t => (((2 * m + 2).factorial : ℝ)⁻¹) * t) ?_
  exact Finset.sum_congr rfl (fun σ _ => cubicTranspose_apply σ (pairId m) idx)

/-- **`E_{D,N}` is a symmetric tensor.** It is a scalar multiple of the sum of
    all mode-permutations of `pairId m`, and that sum is invariant under any
    further permutation (reindexing `σ ↦ σ τ`). -/
theorem symId_isSymmetric (m : ℕ) : (symId m : CubicTensor ℝ (2 * m + 2) N).IsSymmetric := by
  have hsum : IsSymmetric
      ((∑ σ : Equiv.Perm (Fin (2 * m + 2)), (pairId m).transpose σ) :
        CubicTensor ℝ (2 * m + 2) N) := by
    intro τ
    rw [transpose_sum, Finset.sum_congr rfl (fun σ _ => Tensor.transpose_mul σ τ (pairId m))]
    exact Fintype.sum_equiv (Equiv.mulRight τ) _ _ (fun σ => rfl)
  exact hsum.smul _

/-- **`E_{D,N}` acts as the identity on the unit sphere (Lecture 8).** For every
    unit vector `x` (`∑ᵢ ‖xᵢ‖² = 1`), `symId m ⬝ₜ x = x`. Since every permuted
    matching `(pairId m).transpose σ` is identity-on-sphere, so is their average:
    `symId m ⬝ₜ x = (1/D!) ∑_σ x = (1/D!)(D! · x) = x`. -/
theorem identityOnSphere_symId (m : ℕ) :
    IdentityOnSphere (symId m : CubicTensor ℝ (2 * m + 2) N) := by
  intro x hx
  have hsum : (∑ a, x a ^ 2) = 1 := by
    rw [← hx]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Real.norm_eq_abs, sq_abs]
  have hEach : ∀ σ : Equiv.Perm (Fin (2 * m + 2)),
      (pairId m).transpose σ ⬝ₜ x = x := by
    intro σ
    funext j
    rw [apply_transpose_pairId, hsum, one_pow, mul_one]
  rw [symId, apply_smul_T, apply_sum_T,
      Finset.sum_congr rfl (fun σ _ => hEach σ)]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_perm, Fintype.card_fin]
  rw [← Nat.cast_smul_eq_nsmul ℝ, ← smul_assoc, smul_eq_mul,
      inv_mul_cancel₀ (by exact_mod_cast (Nat.factorial_pos (2 * m + 2)).ne'), one_smul]

end CubicTensor
