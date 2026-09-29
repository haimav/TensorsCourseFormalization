/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.TensorTrain.ChainCtrProd
import TensorCourse.TensorTrain.Existence

/-!
# Full TT existence (`thm:tt-existence`), general order `D`

The lecture's `thm:tt-existence`: every tensor admits a TT decomposition. We
prove the structural core by induction on the number of output modes, using
the one-step split factorization `exists_ctrProd_of_splitAt` (peel the first
core off the mode-2 split) and the chain peel `ttOpen_eq_ctrProd_peel`
(`ttOpen` = first core `⋈ₜ` rest), bottoming out at a single core.

* `Tensor.exists_ttOpen_decomp` — every `W : Tensor 𝕜 (n+2) (cons L out)`
  (leading bond `L` exposed) is `ttOpen R T` for some bonds `R` with
  `R 0 = L`, `R last = 1`.
* `Tensor.exists_hasTTDecomp` — every `X` has a TT decomposition
  (`HasTTDecomp X R`), the lecture's `thm:tt-existence`.
-/

noncomputable section

open Finset BigOperators

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The open TT reconstruction with **no cores** is the constant-`1` vector:
    the only bond multi-index over `Fin 1` matching the exposed coordinate
    contributes the empty product `1`. -/
theorem ttOpen_nil {out : Fin 0 → ℕ} (R : Fin 1 → ℕ)
    (T : (k : Fin 0) → Fin (R k.castSucc) → Fin (out k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin 1) → Fin ((Fin.cons (R 0) out : Fin 1 → ℕ) d)) :
    ttOpen R T idx = 1 := by
  rw [ttOpen_apply]
  rw [Finset.sum_eq_single_of_mem
        (Fin.cons (Fin.cast (by simp) (idx 0)) (fun i => i.elim0) : (j : Fin 1) → Fin (R j))
        (Finset.mem_univ _)]
  · rw [if_pos (by simp), Fin.prod_univ_zero]
  · intro r _ hne
    rw [if_neg]
    intro hc
    apply hne
    funext j
    refine Fin.cases ?_ (fun i => i.elim0) j
    rw [Fin.cons_zero]
    apply Fin.ext
    simpa using hc

/-- **Strengthened TT existence (`ttOpen` form).** Every tensor with an
    exposed leading bond `L` is the open TT reconstruction of some cores, with
    leading bond `L` and unit trailing bond. Induction on the number of output
    modes; base case = one core. -/
theorem exists_ttOpen_decomp : ∀ {n : ℕ} (out : Fin (n + 1) → ℕ) (L : ℕ)
    (W : Tensor 𝕜 (n + 2) (Fin.cons L out)),
    ∃ (Rtail : Fin (n + 1) → ℕ)
      (T : (k : Fin (n + 1)) → Fin ((Fin.cons L Rtail : Fin (n + 2) → ℕ) k.castSucc) →
            Fin (out k) → Fin ((Fin.cons L Rtail : Fin (n + 2) → ℕ) k.succ) → 𝕜),
      (Fin.cons L Rtail : Fin (n + 2) → ℕ) (Fin.last (n + 1)) = 1 ∧
        W = ttOpen (Fin.cons L Rtail : Fin (n + 2) → ℕ) T := by
  intro n
  induction n with
  | zero =>
    intro out L W
    refine ⟨fun _ => 1,
      fun k => Fin.cases (fun a b _ => W (Fin.cons a (Fin.cons b (fun i => i.elim0))))
        (fun i => i.elim0) k,
      by simp, ?_⟩
    funext idx
    rw [ttOpen_peel]
    simp only [Fin.cases_zero, ttOpen_nil, mul_one]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      show ((Fin.cons L (fun _ => 1) : Fin 2 → ℕ) 1) = 1 from by simp, one_smul]
    congr 1
    funext d
    refine Fin.cases ?_ (fun i => ?_) d
    · simp only [Fin.cons_zero]; apply Fin.ext; simp
    · simp only [Fin.cons_succ]
      refine Fin.cases ?_ (fun j => j.elim0) i
      simp only [Fin.cons_zero]; apply Fin.ext; simp
  | succ n ih =>
    intro out L W
    -- Reindex `W` into `combinedDims` form using the same equiv `E` that
    -- `ttOpen_eq_ctrProd_peel` uses, factor off the first core, recurse.
    set E := castEquiv (Nat.add_comm 2 (n + 1)) (fun i => chainPeel_hd L out i) with hEdef
    obtain ⟨A, V, hAorth, hW'⟩ := exists_ctrProd_of_split (reindex E.symm W)
    obtain ⟨Rtail'', T'', hlast'', hV⟩ := ih (fun k => out k.succ) _ V
    refine ⟨Fin.cons _ Rtail'',
      fun k => Fin.cases (uncurryCore A) (fun k' => T'' k') k, ?_, ?_⟩
    · -- trailing bond: `(cons L (cons S Rtail'')) (last) = (cons S Rtail'') (last) = 1`
      rw [← Fin.succ_last, Fin.cons_succ]
      exact hlast''
    · -- `W = ttOpen (cons L (cons S Rtail'')) T`
      have hround : reindex E (A ⋈ₜ V) = W := by
        rw [← hW', reindex_comp, Equiv.symm_trans_self, reindex_id]
      refine hround.symm.trans ?_
      rw [ttOpen_eq_ctrProd_peel]
      congr 1
      congr 1
      -- first core: `A = chainCore (T 0)` (the tail `V = ttOpen tail` is closed by
      -- `congr` via `hV`, the bonds/cores matching definitionally)
      simp only [Fin.cases_zero]
      exact (chainCore_uncurryCore A).symm

/-- **Leading-unit-bond squeeze.** When the leading bond is `1`, the open
    reconstruction with the leading coordinate fixed to any `s : Fin (R 0)`
    equals the (fully summed) `ttTensor` reconstruction: the leading bond sum
    is trivial. -/
theorem ttOpen_cons_eq_ttTensor {D : ℕ} {I : Fin D → ℕ} (R : Fin (D + 1) → ℕ)
    (hR0 : R 0 = 1)
    (T : (k : Fin D) → Fin (R k.castSucc) → Fin (I k) → Fin (R k.succ) → 𝕜)
    (idx : (d : Fin D) → Fin (I d)) (s : Fin (R 0)) :
    ttOpen R T (Fin.cons s idx) = ttTensor R T idx := by
  rw [ttOpen_apply, ttTensor_apply]
  haveI : Subsingleton (Fin (R 0)) := by rw [hR0]; infer_instance
  refine Finset.sum_congr rfl (fun r _ => ?_)
  rw [if_pos (by simp only [Fin.cons_zero]; rw [Subsingleton.elim (r 0) s])]
  exact Finset.prod_congr rfl (fun k _ => by congr 1)

/-- **`thm:tt-existence` (general order).** Every tensor admits a TT
    decomposition (with unit boundary bonds). Obtained from
    `exists_ttOpen_decomp` applied to `X` with a unit leading mode prepended,
    squeezed back via `ttOpen_cons_eq_ttTensor`. -/
theorem exists_hasTTDecomp {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I) :
    ∃ R, HasTTDecomp X R := by
  obtain ⟨Rtail, T, hlast, hW⟩ := exists_ttOpen_decomp (n := D) I 1
    (fun jdx => X (fun d => Fin.cast (by simp) (jdx d.succ)))
  refine ⟨Fin.cons 1 Rtail, by simp, hlast, T, ?_⟩
  funext idx
  rw [← ttOpen_cons_eq_ttTensor (Fin.cons 1 Rtail) (by simp) T idx
        (Fin.cast (by simp) (0 : Fin 1)), ← hW]
  congr 1

/-- **`thm:tt-existence` (packaged noun form).** Every tensor admits a *proper*
    packaged TT decomposition. The `TTDecomp` repackaging of `exists_hasTTDecomp`,
    mirroring `exists_orthTuckerDecomp_of_multirank` for Tucker. -/
theorem exists_ttDecomp {D : ℕ} {I : Fin (D + 1) → ℕ} (X : Tensor 𝕜 (D + 1) I) :
    ∃ (R : Fin (D + 2) → ℕ) (tt : TTDecomp 𝕜 (D + 1) I R),
      tt.IsProper ∧ tt.tensor = X := by
  obtain ⟨R, hR⟩ := exists_hasTTDecomp X
  rw [hasTTDecomp_iff_ttDecomp] at hR
  obtain ⟨⟨h0, hlast⟩, tt, hX⟩ := hR
  exact ⟨R, tt, ⟨h0, hlast⟩, hX⟩

end Tensor
