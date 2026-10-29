# jrf_economy_models_2026

JRF の経済モデルの統合 2026年版

<!-- Time-stamp: "2026-10-29" -->

私(JRF)が AI 以前(2006〜2022 年)に作った経済のシミュレーションと、制度の設計・擬数学的な議論を、Claude Opus 5.5 さんと一つずつ読み直し、最後に一つの模型 **im**(integrated model)に組み上げました。八冊の IPYNB と、ブラウザで遊べる箱庭 **The JRF Economy Model 2026** からなります。

もとは、「ミクロ経済学の我流シミュレーション」(2018〜2020 年、[micro_economy](https://github.com/JRF-2018/micro_economy))の、「価格を最適化で決める」という発想を発展させたい、企業の数を連続にしたい、輸出入や、どの企業も生産したがらないときの「危機時の生産」を入れたい、という関心から始まりました。

## ページ

* **[The JRF Economy Model 2026](https://jrf-2018.github.io/jrf_economy_models_2026/im_sandbox.html)** … 三つの国の経済の箱庭(ブラウザの版。日本語と英語)。
* ノートの HTML 版: https://jrf-2018.github.io/jrf_economy_models_2026/ (この README の下のリンク)。

## ノート(`notebooks/`)

書いた順に並べます(読む順の勧めも同じ)。ノート O の O は Origin(始まり)で、私の経済のシミュレーションの始まりの 2011 年の模型を扱います。

| | ノート | 中身 |
|---|---|---|
| 1 | [価格を最適化で決める経済 ── 2020年までの micro_economy モデルの解剖](notebooks/micro_economy_anatomy.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/micro_economy_anatomy.html)) | 元のスコアの本当の最小点は、経済の崩壊点だった。局所的な最適化の「下手さ」が経済を支えていた。条件の数を未知数の数とそろえ(労働の需給 + 利潤率の均等化)、残差を最小にすればよい。「全体最適 + 力関係」は Negishi の方法 |
| O | [経済の中で経済が育つ ── 2011年の外作用モデルと『埋め込み』の遷移](notebooks/embedded_economy.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/embedded_economy.html)) | お金を「マイナスの負債」として書く 2011 年の Perl のモデル。ペニーオークションと無尽講。保存・評価・実現可能性・行動の四つの原理と、経済の圏。部分経済が全体に「格上げ」される遷移の模型と、縮約との関係 |
| 2 | [土地と知財と借金 ── 2019年の資産市場シミュレーションの解剖](notebooks/asset_market.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/asset_market.html)) | 借金の合計は、取り崩しと返済の二つの曲線の交点で釣り合う。贅沢品の需要は流れで決まる(ストック・フロー一貫)。知財の切り下げは内生的な減耗率。信用循環モデルに土地を担保として足す |
| P | [ポイントと年金と信用 ── ネットの小さな通貨の設計(2006〜2018)](notebooks/points_pension.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/points_pension.html)) | 投げ銭予約は部分準備の通貨。空スター、時刻の証明、PP ログイン。はてなポイントの分配のプログラム(Perl を無改変で実行)のペンションと ID の分割。BI 否認時の返し方。AI の利用料を書き手に配る雛形 |
| I | [必需品と贅沢品の宇宙的独立関係 ── 絶滅の確率、0 × ∞、ナイト的不確実性、そして学習](notebooks/cosmic_independence.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/cosmic_independence.html)) | 『宗教学雑考集』の擬数学的な議論を数学に。絶滅の確率 1 と、破局の確率の総和が有限なら生き延びる望み。「贅沢品の量は決められない」の正体は信念の区間(Bewley)。国の競争と「経済 = 学習」 |
| 3 | [誰が一家離散するのか ── シミュレーション仏教の経済と、裾の重い運](notebooks/simbd_economy.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/simbd_economy.html)) | [simbd](https://github.com/JRF-2018/simbd) の経済の部分。裾の重い運は規模に対する収穫逓増を生む。一家離散するのは誰か(帳簿の式と臨界の値)。死蔵の割合の折れ線と ρ_eff。災害と、記録から学ぶ社会 |
| M | [雑多なモデルの再検討 ── 必需品と贅沢品、縁故主義、投票、AI と弾圧](notebooks/misc_models.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/misc_models.html)) | [misc_ipynb](https://github.com/JRF-2018/misc_ipynb) の AI と作った小さな模型(gj_6 ほか)。結論が式に書き込まれていないか、外しても残るか |
| 4 | [2026年の経済モデル統合 ── 価格と信用と土地と災害、そして一人の暮らし](notebooks/integrated_model.ipynb) ([HTML](https://jrf-2018.github.io/jrf_economy_models_2026/integrated_model.html)) | 統合模型 im。年の中の釣り合いを均衡の残差の最小化で解き、帳簿を毎年閉じたまま、信用・土地・知財・地域と通貨・災害・危機時の生産・学習と投票・文化・年金・一人のプレイヤーまで積み上げる。ブラウザの版の説明は 16.5 節 |

どのノートも、上から順に実行すれば全部動きます(Python 3、numpy、scipy、matplotlib。ノート I は PyTorch、ノート O・P は Perl も使います)。元のコードは、決まった版を GitHub や作者のサイトから取ってきて、sha256 を確かめてから走らせます。ノート 4 は、このリポジトリの `im/` を使います(なければ `git clone` します)。並列の計算に fork を使うので、Linux か macOS を想定しています。

各ノートの Lean のセルは、最後にまとめて確かめます。Lean 4(v4.34.1)と Mathlib の入った lake のプロジェクトの場所を、ノートと同じディレクトリの `.leandir` に一行で書いておくと動きます(なければ飛ばします)。

## ブラウザの版: The JRF Economy Model 2026(`site/`)

三つの島(三つの国)の経済の箱庭です。模型は im そのもので(JS に写したもの)、ブラウザの中で一年ずつ価格を解きます。

* **箱庭**: 政策のつまみ、出来事のボタン(大災害、局地の災害、能力の半減、利上げ、熱狂、新しい工房)、指標のカード、新聞の見出し、効果音。島の上のものにマウスを載せると、説明とその年の数が出ます。
* **分析の画面**: 二つの試しの比べ、平行世界の扇(運だけを引き直した 12〜48 本の未来)、小さな政策の地図。
* **一人の暮らし**: 箱庭の中の一つの世帯として、暮らしの支出、何を削るか、働く度合い、子、選挙の票と呼びかけを選びます。代替わりと家系図。
* **時間のつまみ**: 戻る、戻って命令を出すと枝が分かれる、サイコロで先の運を引き直す、取り消し、セーブとロード、URL で渡す。
* 遊びの出来事として、革命(移り変わりの痛みが積もると、税と再分配の体制が入れ替わる)と、格上げの工房。勝ち負けはありません。

手元で見るには、`site/` で `python -m http.server` を起こし、`http://127.0.0.1:8000/im_sandbox.html` を開いてください(file:// では Worker が使えないので、画面の中で回す道になり、少し重くなります)。

### Python と JS の一致について

参照実装は Python の `im/` で、JS(`site/js/model/`)はその写しです。同じ数が出るように、im に持ち運べる乱数(xoshiro128\*\*)と持ち運べる最小化(Levenberg–Marquardt)の道(`rng='portable'`、`solver='portable'`)を足しました。`python/check_js_vs_py.py` が JS を quickjs で動かし、一国の場面 18、三国の場面 8 で、Python と年ごとに比べます。水準の量は各年の相対のずれ、差の量(物価上昇率、赤字、収支など)は系列の大きさに対するずれで、手元では 1e-9 以内で合います(機械によって numpy の `exp`・`log` の最後の桁が違うので、CI の合否は 1e-8 で決めています)。ただし:

* 価格は数値の微分を使って解くので、関数の値の丸めのずれが 1e8 倍ほどに増え、Python と JS の価格は 1e-11 ほどずれます。ふつうの道では、このずれは年を追って広がりません。
* **不安定な道では、丸めのずれが育ちます。** 変動の為替で一国に価格の上限を置くと、その国の通貨が下がり続け(80 年で 1/20)、ずれは 60 年目まで 1e-9 以内ですが、80 年目で 6e-8 になります。この場面だけ、許容を 1e-6 にしています(Python と JS のずれをそこまで厳密に見る必要はないと判断し、場面の長さは縮めていません)。
* **革命の場面では、預金のジニに意味のない年があります。** 再分配が入れ替わるたびに預金の総額が 0 の近くを通り、預金のジニ(分母が預金の和)が 82 のような値になります。この場面も許容を 1e-6 にしています。ほかの量は 1e-9 で合います。
* 預金のジニ(分母が預金の和)は、ほかの場面でも預金の総額が小さい年に桁が落ちやすいので、どの場面でも許容 1e-6 で数えています(GitHub Actions の機械では numpy の `exp`・`log` の最後の桁が手元と違い、投票と年金の三国の場面で 1.7e-9 になりました)。
* ブラウザの種類によって `Math.exp` などの最後の桁が違いうるので、URL で渡した試しは、別のブラウザでは長い年のあとに少しずれることがあります。同じブラウザでのやり直しとロードは、完全に同じ数になります。

## 中身

* `notebooks/` … 八冊の IPYNB(実行済みの出力つき)。Pages の配備のときに HTML に変換します(再実行はしません)。
* `im/` … 統合模型 im(ノート 4)の Python の一式と、段ごとの実験の台本と記録。詳しくは [im/README.md](https://github.com/JRF-2018/jrf_economy_models_2026/blob/master/im/README.md)。
* `site/` … ブラウザの版。GitHub Actions(`.github/workflows/pages.yml`)で GitHub Pages に配備します。
* `python/` … `check_js_vs_py.py`(JS と Python の一致)、`check_timeline.py`(時間のつまみ: 鍵の年から計算し直した枝が、通しの計算と同じ数になる)。CI は `.github/workflows/check.yml`(im の段ごとの関門 `im/test_im.py` も回します。10 分ほどかかるので、`im/`・`site/js/`・`python/` を変えたときと、手で起こしたときだけ回します)。
* `lean/` … 各ノートが書き出す Lean のファイル(`lean/Notes/`)と、im の帳簿の恒等式(`Notes/im.lean`)。Lean 4(v4.34.1)と Mathlib。CI は `.github/workflows/lean.yml`。

```bash
pip install numpy scipy quickjs
(cd im && python test_im.py)
python python/check_js_vs_py.py
python python/check_timeline.py
(cd lean && lake exe cache get && lake build)
```

### Lean による定理(`lean/Notes/`)

| ファイル | ノート | 定理の数 | 中身の例 |
|---|---|---|---|
| `micro_economy.lean` | 1 | 6 | 名目の水準は決まらない(同次性)、会計の誤りの形、残差の二乗和が 0 なら均衡、スラッファの取り換え |
| `embedded_economy.lean` | O | 11 | ペニーオークションの無差別、集計は保存を守る、一物一価の条件、区間ごとの縮約、一点へのまとめは格上げのリセットと可換にならない(根の候補ごとに一点なら可換) |
| `asset_market.lean` | 2 | 5 | 贅沢品の需要が 500 に決まる、落ち着き先の一意性、減耗する資産の定常の量と値打ち |
| `points_pension.lean` | P | 8 | ID を分けても取り分は変わらない、BI 否認時の返し方の閉じた式と負担の上限、複利の借入の上限、ポンジ・ゲーム |
| `independence.lean` | I | 21 | 破局の確率が下がらなければ絶滅は確実、総和が有限なら生存の確率は正、終末論法(SSA と SIA)、0 × ∞、信念が区間なら支配されない配分は区間、国の競争、危機の近傍で配分は一点に |
| `breakup.lean` | 3 | 16 | 裾の重い損失の平方根の法則と規模に対する収穫逓増、下に壁があると賭けが得になる、有限責任は期待値を上げる、死蔵の折れ線と ρ_eff、災害の守りと記憶 |
| `misc_models.lean` | M | 13 | gj の帳簿と資本家の所得(カレツキ)、ゴーマンの集計、縁故主義の尾根と情報の値打ち、相関のある票の分散、注意の予算 |
| `integrated_model.lean` | 4 | 13 | 標本の持ち上げ、帳簿の恒等式、一般のカレツキの式、利子生活者の地域の貿易赤字、準備の縛り、波及の上限、群れの相関と投票率 |
| `im.lean` | 4 | 4 | im の帳簿の恒等式とカレツキの式 |

`sorry` はありません。

## 関連

* 元のコード: [micro_economy](https://github.com/JRF-2018/micro_economy)(2018〜2020)、[simple_market_0](https://github.com/JRF-2018/simple_market_0)(2011、Perl)、[simple_market_a1](https://github.com/JRF-2018/simple_market_a1)(2019)、[simbd](https://github.com/JRF-2018/simbd)(2021)、[misc_ipynb](https://github.com/JRF-2018/misc_ipynb)。
* 信用循環モデル: [credit_simulation_v6](https://github.com/JRF-2018/credit_simulation_v6)(ノート「信用創造はどこで止まるのか」「信用の世界経済は一つか」)。im は、その信用と分配の流れと、micro_economy の価格の流れを合わせたものです。
* 元の記事(ブログ『JRF の私見:税・経済・法』): 《[ミクロ経済学の我流シミュレーション その1 基礎経済モデル](http://jrf.cocolog-nifty.com/society/2018/03/post.html)》(2018年3月)〜《[その5 大改良](http://jrf.cocolog-nifty.com/society/2020/02/post-3f22aa.html)》(2020年2月)、《[外作用的簡易経済シミュレーションのアイデアと Perl による実装](http://jrf.cocolog-nifty.com/society/2011/01/post.html)》(2011年1月)、《[資産市場の簡易シミュレーション](http://jrf.cocolog-nifty.com/society/2019/05/post-9da6e7.html)》(2019年5月)。各ノートの付録 C に、関連する記事の一覧があります。

## Author (文責)

JRF ( http://jrf.cocolog-nifty.com/statuses , Twitter (X): @jion_rockford )

元のモデル・制度の設計・議論は JRF のもの(AI 以前のもの)です。読み直し・模型の組み直し・プログラム・証明・ブラウザの版は Claude Opus 5.5 (Anthropic) によるものです。

## License

基本的に「学習用に特に Public Domain になることを望んでいるが、それが気になる場合は MIT License で。」です。

----
(This document is mainly written in Japanese/UTF8.)
