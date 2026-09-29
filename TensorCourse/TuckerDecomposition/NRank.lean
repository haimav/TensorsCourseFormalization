/-
Copyright (c) 2026 Haim Avron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import TensorCourse.Tensor.ModeUnfold
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# n-Ranks of a Tensor

The *n-rank* of an order-`D` tensor is the rank of its mode-`n` matricization,
`rankₙ(X) = rank(X_(n))`. A tensor *has multilinear rank* `(R₀, …, R_{D-1})`
when its n-rank equals `R n` at every mode `n`. These are the basic
invariants governing the Tucker decomposition: the existence theorem
`thm:hosvd` produces an exact Tucker decomposition of size equal to the
multilinear rank (see `TuckerDecomposition/HOSVD.lean`).

## Main definitions

* `Tensor.nRank n T` — the n-rank `(T.modeUnfold n).rank`.
* `Tensor.HasMultirank T R` — the proposition `∀ n, T.nRank n = R n`.

## Main results

* `Tensor.nRank_def` — `T.nRank n = (T.modeUnfold n).rank` (definitional).
* `Tensor.hasMultirank_iff` — `T.HasMultirank R ↔ ∀ n, T.nRank n = R n`.
* `Tensor.HasMultirank.nRank_eq` — extract `T.nRank n = R n` from the
  multilinear-rank hypothesis.
-/

noncomputable section

namespace Tensor

variable {𝕜 : Type*} [RCLike 𝕜]
variable {D : ℕ} {dims : Fin D → ℕ}

/-- The **n-rank** of a tensor: the rank of its mode-`n` matricization,
    `rankₙ(X) = rank(X_(n))`. -/
def nRank (n : Fin D) (T : Tensor 𝕜 D dims) : ℕ := (T.modeUnfold n).rank

/-- `nRank` unfolds to the rank of the mode-`n` matricization (definitional). -/
@[simp] theorem nRank_def (n : Fin D) (T : Tensor 𝕜 D dims) :
    T.nRank n = (T.modeUnfold n).rank := rfl

/-- `T` *has multilinear rank* `(R₀, …, R_{D-1})` when its n-rank equals
    `R n` at every mode `n`. -/
def HasMultirank (T : Tensor 𝕜 D dims) (R : Fin D → ℕ) : Prop :=
  ∀ n, T.nRank n = R n

/-- `HasMultirank T R` unfolds to `∀ n, T.nRank n = R n` (definitional). -/
theorem hasMultirank_iff (T : Tensor 𝕜 D dims) (R : Fin D → ℕ) :
    T.HasMultirank R ↔ ∀ n, T.nRank n = R n := Iff.rfl

/-- Extract the n-rank value from a multilinear-rank hypothesis. -/
theorem HasMultirank.nRank_eq {T : Tensor 𝕜 D dims} {R : Fin D → ℕ}
    (hT : T.HasMultirank R) (n : Fin D) : T.nRank n = R n := hT n

end Tensor

end
