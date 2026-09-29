# Tensor Course Formalization

This repository contains Lean code formalizing theorems proved as part of TAU's course 0372-4033, *Tensor Computations and Analyzing High Dimensional Data*.

## How the Lean code was produced

The Lean code was AI-generated, with human supervision in two areas:

1. **Statement fidelity.** A human verified that each Lean theorem statement accurately reflects the corresponding statement in the lecture notes. *The Lean proofs themselves were not read or checked by a human.* The rationale is that Lean guarantees the correctness of a proof, so how a statement is proved matters less. What needs human verification is only that the lecture notes' statements were translated correctly into Lean.
2. **Structure and definitions.** The data structures and definitions received some level of review (possibly not enough) to keep the verification code clean and reusable.

## A note on the proofs

The proofs in the lecture notes were *not* AI-generated, and they were carefully verified by hand. The Lean formalization adds an extra layer of verification on top of that.

## License

Copyright 2026 Haim Avron, Tel Aviv University.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
