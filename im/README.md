# im: 統合模型(ノート 4「2026年の経済モデル統合」)

ノート 4(`notebooks/integrated_model.ipynb`)が取ってきて使う模型の一式。文責 JRF、模型と実験の補助 Claude Opus 5.5。

- `im_core.py`: 帳簿(Ledger)とパラメータの登録簿(Params)。
- `im1.py`: 一国の模型(三財の価格と生産、信用、重みつきの標本の家計、土地・知財・株、災害、学習と投票、危機時の生産、文化、年金、貸し手、プレイヤーの記入ほか。部品はすべてオプションで、既定では切ってある)。
- `im3.py`: 地域と通貨(共通・固定・変動・同盟)。`im_firms.py`: 部門の中の企業の集まり。`im_time.py`: 時間の層(粗い時間の進め方、平行世界)。`upgrade.py`: 格上げの別枠の模型。`im_gj.py`: 骨組みの上に書き直したノート M の gj。
- 比べる相手: `me_c.py`(micro_economy の現代化)、`v6a.py`・`v6b.py`(資産市場つきの信用循環モデルと、その標本の版)、`gj.py`(ノート M 第 2 章)。
- `test_im.py`: 段ごとの関門(帳簿が閉じる、部品を切れば前と同じ数)。`python3 test_im.py` で全部を確かめる(2 分ほど)。結果の例は `test_im_last.txt`。
- `results_stage*.py` と `results_stage*.txt`: 段ごとの実験の台本と記録(標本 2000 記入、種 8〜12 個)。重い台本は、並列の数を環境変数 `IM_WORKERS`(既定 4)で決める。`run_queue.sh` は順に回す(メモリの上限つき)。
- `im.lean`: 帳簿の恒等式とカレツキの式(Lean 4 + Mathlib)。
