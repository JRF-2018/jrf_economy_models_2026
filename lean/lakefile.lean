import Lake
open Lake DSL

package jrf_economy_models_2026

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.34.1"

-- 各ノートが書き出す Lean のファイル(Notes/)と、模型 im の帳簿の恒等式(Notes/im.lean)
@[default_target]
lean_lib Notes where
  globs := #[.submodules `Notes]
