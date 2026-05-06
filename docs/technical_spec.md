# local-doc-search 技術仕様

## 概要

`local-doc-search` は、多言語のローカルテキストファイルを検索するCLI。
SQLiteを永続ストアとして使い、FTS5 trigramによる文字列検索とsqlite-vecによるベクトル検索を組み合わせる。

主なコマンド:

- `local-doc-search index`: 指定root配下のファイルをSQLite DBへindexする。
- `local-doc-search search`: FTS/vec/hybrid検索を実行する。
- `local-doc-search tui-search`: 検索結果をfzfで選択し、該当文書をpager/editorで開く。
- `local-doc-search info`: DB metadataと件数を表示する。
- `local-doc-search files`: DBにindex済みのファイル一覧を表示する。
- `local-doc-search server`: ローカル検索serverを起動し、queryごとのモデルロードを避ける。
- `local-doc-search codex-index`: Codex session履歴を固定DBへindexする。
- `local-doc-search codex-search`: Codex session履歴の固定DBを検索する。
- `local-doc-search codex-server`: Codex session履歴の固定DB用serverを起動する。

パッケージ管理と実行は `uv` を前提にする。

## DB構造

SQLite DBには以下を保存する。

- `metadata`
  - `schema_version`
  - `embedding_model`
  - `embedding_dim`
  - `embedding_backend`
  - `embedding_device`
  - `embedding_batch_size`
  - `embedding_prefix_policy`
  - `index_kind`
  - `created_at`
- `files`
  - 絶対path: `path`
  - index時のroot: `root_path`
  - rootからの相対path: `relative_path`
  - 差分判定用: `size`, `mtime_ns`, `content_hash`
- `chunks`
  - fileごとのchunk本文
  - `chunk_index`, `start_offset`, `end_offset`
  - `start_line`, `end_line`
  - Codex履歴用metadata: `session_id`, `cwd`, `role`, `turn_id`, `timestamp`, `session_path`, `line_no`
- `chunks_fts`
  - `fts5(path UNINDEXED, text, tokenize='trigram')`
- `chunk_vec`
  - `vec0(embedding float[N])`
  - `N` はindex時のembedding dimension

既存DBに `root_path` / `relative_path` / `start_line` / `end_line` / Codex履歴用metadata columns がない場合は、起動時に不足カラムを追加する。
古いDBへ追加した行番号は既定値になるため、正確な行番号が必要な場合は再indexする。

## Indexing

`local-doc-search index --db path.sqlite --root DIR [--root DIR...] [--ext .md ...] [--exclude REGEX ...]` でindexする。

index開始時はcron logや長時間実行時の識別のため、`command`, `db`, `roots`, `model`, `device`, `batch_size`, `rebuild`, `extensions`, `exclude` を標準出力に表示する。

処理内容:

- rootは複数指定できる。
- rootがファイルの場合は、そのファイルの親ディレクトリをroot扱いにする。
- 拡張子指定がある場合は、その拡張子だけ対象にする。
- デフォルト対象拡張子は `.txt`, `.md`, `.markdown`, `.rst`。
- `--exclude` はrootからの相対pathをPOSIX形式にした文字列に対してPython regex `re.search()` で判定する。
- `--exclude` は複数指定でき、1つでもmatchしたファイルはindex対象外にする。
- `.git`, `.venv`, `__pycache__`, dot directory は走査対象から除外する。
- UTF-8 / UTF-8 BOMとして読めないファイルはskipする。
- ファイル本文は空行区切りの段落を抽出し、既定600文字の上限以内で複数段落を1chunkへまとめる。
- 段落packingでchunk境界が発生する場合、直前の1段落を次chunkにも含める。
  - ただし、overlap段落と次段落だけで600文字上限を超える場合は上限を優先し、overlap段落を落とす。
- 単独で長すぎる段落は600文字上限で分割し、120文字overlapさせる。
- chunkingは拡張子ごとのstrategyで選択する構造にしている。
  - `.md`, `.markdown` は `markdown-section` strategyを使い、ATX見出し (`#` から `######`) のsection境界を越えてchunkをまとめない。
  - Markdown chunkには現在sectionの上位heading pathを検索用textのprefixとして付与する。
    - 例: `# aaa` > `## bbb` > `### ccc` 配下の本文chunkには、`# aaa`, `## bbb`, `### ccc` を前置する。
    - prefixは検索用textにのみ追加し、`start_offset`, `end_offset`, `start_line`, `end_line` は元本文内の位置を維持する。
  - Markdown section内の段落packingではheading contextを優先し、通常の1段落overlapは行わない。
  - fenced code block内の見出し風行はsection境界として扱わない。
  - fenced code block (3連バッククォートまたは `~~~`) 内の本文はchunk対象から除外する。
  - `.txt`, `.rst` は `paragraph-pack` strategyを使う。
  - 将来はreStructuredText構造単位、code block考慮などを拡張子別に追加できる。
- document chunkは `passage: ...` prefixでembeddingする。
  - 実際のprefixはmodelごとのprefix policyで決まる。
- index中は候補ファイル数、処理済みファイル数、現在の状態、embedding対象chunk数、embedding済みchunk数、累計処理chunk数、1秒あたりの処理chunk数をprogress表示する。
- embeddingは `--batch-size` ごとのchunk batch単位で進捗を更新する。処理済みファイル数はファイル全体のDB反映が完了した時点で進める。

exclude例:

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --exclude '^archive/' --exclude '\.tmp\.md$'
```

## 差分更新

indexは差分更新に対応している。

DBに保存済みの `root_path`, `relative_path`, `size`, `mtime_ns`, `content_hash` が現在のファイルと一致する場合、そのファイルは再indexしない。

ファイル状態ごとの挙動:

- 新規ファイル: chunk/FTS/vectorを新規作成する。
- 更新ファイル: そのファイルの既存chunk/FTS/vectorを削除し、ファイル単位で再作成する。
- 削除ファイル: DBから該当file/chunk/FTS/vectorを削除する。
- `--exclude` により今回のindex対象外になった既存ファイル: 削除ファイルと同じくDBから削除する。
- 変更なし: 再チャンク化、再embedding、FTS/vector更新をskipする。

現在の実装上の注意:

- 変更ファイル内の一部chunkだけを再embeddingする粒度ではなく、更新されたファイル全体を再indexする。
- index実行ごとにroot配下のファイル一覧走査は行う。
- 現状はhash計算のためにファイル本文を読む。大規模ディレクトリ向けには、`size + mtime_ns` が一致した時点でhash計算もskipする余地がある。

## Embedding

デフォルトモデルは `intfloat/multilingual-e5-small`。

index時:

- `--model` でembedding model名を指定できる。
- `--device auto|cpu|mps` でembedding実行deviceを指定できる。
- `auto` はPyTorch MPSが利用可能なら `mps`、不可なら `cpu` を使う。
- `--device mps` を明示してMPSが利用できない場合はエラーにする。
- `--batch-size` でindex時に一括処理するchunk数を指定できる。
- 指定モデル名、dimensionはDB metadataへ保存する。
- backend、resolved device、batch size、prefix policyもDB metadataへ保存する。
- `embedding_batch_size` は再現性・監査・性能比較用の記録であり、検索時の実行やDB互換性判定には必須ではない。

search時:

- `--model` は指定しない。
- vector系検索ではDB metadataの `embedding_model` を読み、そのモデルでquery embeddingを生成する。
- `--device auto|cpu|mps` でquery embeddingのdeviceだけ指定できる。
- queryは `query: ...` prefixでembeddingする。
  - 実際のquery prefixはDB metadataの `embedding_prefix_policy` から決める。
- DBにembedding metadataがない場合は、reindexを促すエラーにする。

benchmark時:

- `local-doc-search benchmark-embeddings` はindexを作成せず、embedding provider単体の速度を測る。
- `--model` は複数指定できる。省略時は `pfnet/plamo-embedding-1b` と `sbintuitions/sarashina-embedding-v2-1b` を比較する。
- `--device auto|cpu|mps`, `--batch-size`, `--documents`, `--warmup`, `--repeat`, `--task passage|query` を指定できる。
- 出力ではモデルロード時間と、warmup後のencode時間を分ける。
- `mps` の場合は計測前後に `torch.mps.synchronize()` を呼び、非同期実行の未完了分を測定から漏らさない。
- `--input-file` を指定するとUTF-8テキストを段落単位、段落が1つだけなら非空行単位でbenchmark文書として読む。

対象モデルとprefix policy:

- `intfloat/multilingual-e5-small`: query=`query: `, passage=`passage: `
- `cl-nagoya/ruri-v3-*`: query=`検索クエリ: `, passage=`検索文書: `
- `Qwen/Qwen3-Embedding-0.6B`: query=`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`, passageはprefixなし。既知dimensionは1024。
- `BAAI/bge-m3`: query/passageともprefixなし。既知dimensionは1024。
- `sbintuitions/sarashina-embedding-v2-*`: query=`task: 質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。\nquery: `, passage=`text: `
- `pfnet/plamo-embedding-1b`: `plamo-custom` backendでmodel card推奨の `AutoModel.encode_query` / `AutoModel.encode_document` を使う。prefixはPLaMo model側で扱うためlocal-doc-search側では付与しない。

推奨:

- 安定運用では `intfloat/multilingual-e5-small` を基本とする。
- 日本語精度を重視する場合は `cl-nagoya/ruri-v3-*` を優先候補とする。
- `Qwen/Qwen3-Embedding-0.6B` と `BAAI/bge-m3` は多言語候補としてSentenceTransformer backendで利用できる。Qwen3はqueryだけretrieval instructionを付与し、documentはprefixなしでindexする。bge-m3はquery/documentともlocal-doc-search側ではprefixを付与しない。
- `sbintuitions/sarashina-embedding-v2-1b` はSentenceTransformer backendで利用できる日本語重視の候補。公式model cardのRetrieval/Reranking用prefixを付与する。Sarashina Model NonCommercial License Agreementで配布されているため用途に注意する。
- `pfnet/plamo-embedding-1b` は `plamo-custom` backendで対応する。過去に確認した非有限値対策として検出・warning・retryは残すが、通常の失敗を前提にした扱いではない。

現在のbackend:

- `sentence-transformers`
  - `intfloat/multilingual-e5-small`, `cl-nagoya/ruri-v3-*`, `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, `sbintuitions/sarashina-embedding-v2-*` など。
- `plamo-custom`
  - `pfnet/plamo-embedding-1b` 専用。
  - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` と `AutoModel.from_pretrained(..., trust_remote_code=True)` を使う。
  - `AutoModel.from_pretrained()` では `dtype=torch.bfloat16` を明示する。sqlite-vecへ保存するembeddingは既存通り `float32` へ変換する。
  - PLaMoのcustom codeは `config.max_length` を参照するため、存在しない場合は `max_position_embeddings` から補完する。
  - PLaMoのRotaryEmbedding cacheは、Transformersのmeta tensor経由loadとnon-persistent bufferの組み合わせで不正な状態になり得るため、device移動後、dimension probe前に全layerのrotary cacheを現在device/dtypeで再生成する。
  - `--device auto` は他backendと同じく、PyTorch MPSが利用可能なら `mps`、不可なら `cpu` を使う。
  - PLaMoの `encode_document` / `encode_query` はCPUでも確率的に非有限値を返すことがあるため、非有限値の場合のみwarningを出して最大5回retryする。全試行失敗した場合はエラーにし、不正vectorは保存しない。
  - document chunkは `encode_document(texts, tokenizer)`、queryは `encode_query(text, tokenizer)` でembeddingする。
  - 公式要件として `sentencepiece` が必要。

利用例:

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run local-doc-search index --db notes.sqlite --root ~/notes --model cl-nagoya/ruri-v3-70m --device mps
uv run local-doc-search index --db notes.sqlite --root ~/notes --model Qwen/Qwen3-Embedding-0.6B --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model BAAI/bge-m3 --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model sbintuitions/sarashina-embedding-v2-1b --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model pfnet/plamo-embedding-1b --device auto
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --mode vec --device auto
```

## Search Modes

`local-doc-search search --db A.sqlite [--db B.sqlite...] --mode` は以下に対応する。

- `fts`
  - FTS5 trigramだけで検索する。
- `vec`
  - sqlite-vec `vec0` のKNNだけで検索する。
- `fts-vec`
  - FTS/LIKEで候補を取り、query vectorとの距離でrerankする。
- `vec-fts`
  - vector検索で候補を取り、FTS rankまたはLIKE一致でrerankする。

query入力:

- `--query`
  - semantic/vector検索用の自然文query。
  - FTSで使う場合はFTS5構文ではなくリテラルphraseとして扱う。
- `--pattern`
  - SQLite FTS5 `MATCH` に渡すpattern。
  - `AND`, `OR`, `NOT`, `NEAR`, prefixなどFTS5 query syntaxをそのまま使える。

mode省略時の推定:

- `--query` のみ: `vec`
- `--pattern` のみ: `fts`
- `--query` + `--pattern`: `vec-fts`

hybrid時の入力:

- vector側は `--query` を使う。
- FTS側は `--pattern` があれば `--pattern` を使い、無ければ `--query` をリテラルphraseとして使う。
- `--query` + `--pattern` + `--mode fts-vec` は、FTS候補を `--pattern` で取得し、`--query` のvectorでrerankする。

3文字未満のリテラルqueryはFTS5 trigramの `MATCH` では扱いづらいため、`LIKE` fallbackで候補取得または文字一致スコア計算を行う。`--pattern` 指定時はFTS5構文を尊重し、LIKE fallbackしない。

## Codex History Search

`local-doc-search codex-index` はCodex session履歴を専用DBへindexする。

- 固定DB pathは `~/.codex/local-doc-search/codex-history.sqlite`。
- 既定の入力rootは `~/.codex/sessions`。
- 既定モデルは `cl-nagoya/ruri-v3-310m`。
- `--root` で別のsession rootまたは `.jsonl` ファイルを指定できる。
- rootが存在しない場合、またはfile rootが `.jsonl` ではない場合は、DB作成や `--rebuild` の前にエラーで停止する。
- `--model`, `--device`, `--batch-size`, `--rebuild` は通常indexと同様に使える。
- DB metadataに `index_kind=codex-history` を保存する。

抽出対象:

- `session_meta.payload.id` を `session_id` として保存する。
- `session_meta.payload.cwd` を `cwd` として保存する。
- index対象messageのJSONL内1-origin行番号を `line_no` として保存する。
- `response_item.payload.type == "message"` のうち、`role=user` の実指示をindexする。
- `role=assistant` は `phase=final_answer` のみindexする。
- `phase=commentary` の途中経過、developer message、tool call/output、reasoning、subagent/guardian session、AGENTS/env初期contextはindexしない。
- chunkは原則1turn=1chunk。
- ただしturn textが長すぎる場合は、通常indexと同じ `MAX_CHARS=600` の段落packingで複数chunkへ分割する。
  - これにより長大な貼り付けや回答がMPS embedding時に過大なsequence lengthになることを避ける。
  - 分割後chunkにも同じ `session_id`, `cwd`, `role`, `turn_id`, `timestamp`, `session_path`, `line_no` を付与する。
  - `start_offset`, `end_offset`, `start_line`, `end_line` は元turn text内の位置を表す。

`local-doc-search codex-search` は固定DBを検索する。

- `--db` と `--model` は指定しない。
- vector系検索では固定DB metadataの `embedding_model` を使う。
- 出力には `session_id`, `cwd`, `role`, `timestamp`, `session_path`, `line_no` を含める。

利用例:

```bash
uv run local-doc-search codex-index --rebuild
uv run local-doc-search codex-server --device auto
uv run local-doc-search codex-search --query "以前相談した内容" --mode fts-vec
uv run local-doc-search codex-search --pattern "実装 OR エラー"
uv run local-doc-search codex-search --query "以前相談した内容" --json
```

## Output

検索結果には以下を含める。

- `score`
- 絶対path: `path`
- index時rootからの相対path: `relative_path`
- chunk番号: `chunk_index`
- 行番号範囲: `start_line`, `end_line`
- Codex履歴検索時: `session_id`, `cwd`, `role`, `timestamp`, `session_path`, `line_no`
- snippet
- `--explain` 指定時:
  - `fts_rank`
  - `vec_distance`

`--json` 指定時は `SearchResult` の内容をJSON配列として出力する。
他エージェントから利用する場合は、`path`, `relative_path`, `start_line`, `end_line`, `text` を参照すると、該当ファイル内の検索hit範囲を特定できる。

## TUI Search

`local-doc-search tui-search` は `local-doc-search search` と同じ検索引数を受け取り、結果を `fzf` で選択できるようにする。

```bash
uv run local-doc-search tui-search --db notes.sqlite --query "検索語" --mode vec
```

挙動:

- 検索実行は通常の `search` と同じ経路を使うため、server discovery、`--db` 省略、subset DB server利用、direct fallbackの挙動は同じ。
- 一覧表示は `score source relative_path:start_line-end_line` の短いカラム風表示を基本とする。
- `fzf` previewでは `start_line` / `end_line` 周辺の本文を行番号付きで表示する。
- hit chunkがpreview下部に寄りすぎないよう、hit前のcontextはpreview行数の約1/6に抑える。
- preview配置の既定は `down:60%` とし、上に選択肢、下にpreviewを置く。
- `--preview-window` で `right:70%`, `left:60%`, `down:50%` などfzfのlayout指定を渡せる。
- 選択後は既定で `less +<start_line> <path>` を起動する。
- `--pager` で `vim`, `nvim`, `bat` など別のpager/editorを指定できる。
- `fzf` が見つからない場合はエラーにする。

## Indexed File Listing

`local-doc-search files --db notes.sqlite` は、DBの `files` テーブルに保存されているindex済みファイル一覧を表示する。

通常出力は、terminal幅に合わせたtableではなく、1 file 1 lineの `key=value` 形式で出力する。
path系の値は空白を含む場合に備えてshell風にquoteする。
`--json` 指定時はJSON配列として出力する。

出力項目:

- `path`: 絶対path
- `root_path`: index時のroot
- `relative_path`: rootからの相対path
- `size`: index時のfile size
- `mtime_ns`: index時のmtime nanoseconds
- `content_hash`: index時のSHA-256 hash

表示順は `relative_path`, `path` の昇順。

## Multi DB Search

`local-doc-search search` は `--db` を複数指定できる。

```bash
uv run local-doc-search search --db notes.sqlite --db work.sqlite --query "検索語" --mode fts-vec
```

挙動:

- 各DBから候補を取得し、score順でglobal top-kへmergeする。
- 結果には `db_path` を含める。
- `db_path + chunk_id` で横断検索結果を一意に扱う。
- `fts` modeはembedding metadataの互換性を要求しない。
- `vec`, `fts-vec`, `vec-fts` は全DBのembedding互換性を要求する。

vector系検索で一致が必要なmetadata:

- `embedding_model`
- `embedding_dim`
- `embedding_backend`
- `embedding_prefix_policy`

一致不要なmetadata:

- `embedding_device`
- `embedding_batch_size`

## Server Mode

`local-doc-search server` はローカルHTTP serverをforegroundで起動する。

```bash
uv run local-doc-search server --db notes.sqlite --db work.sqlite --device auto
```

`local-doc-search codex-server` はCodex履歴固定DB用のserverをforegroundで起動する。

- 固定DB pathは `~/.codex/local-doc-search/codex-history.sqlite`。
- `--db` は指定しない。
- `--device`, `--host`, `--port` は通常serverと同様に使える。
- 起動前に `index_kind=codex-history` のDBであることを検証する。

```bash
uv run local-doc-search codex-server --device auto
```

`local-doc-search search` は同じDB集合のserverが起動中ならserverへ問い合わせる。
完全一致serverがない場合でも、指定DB集合が起動中serverのDB集合の部分集合であれば、そのserverへ問い合わせて対象DBだけ検索する。
server registryが存在する場合は、起動直後のraceを避けるため短時間 `/health` をretryする。
serverがない、または応答しない場合はdirect検索へfallbackする。
`--db` なしの場合は、liveなserverが1件だけならそのserverへ問い合わせる。
liveなserverが0件または複数件の場合は、曖昧または検索不能としてエラーにする。
`--db` なしではdirect検索へfallbackしない。

```bash
uv run local-doc-search search --db notes.sqlite --db work.sqlite --query "検索語" --mode fts-vec
uv run local-doc-search search --db notes.sqlite --query "検索語" --mode vec --no-server
uv run local-doc-search search --query "検索語" --mode vec
```

server discovery:

- `~/.cache/local-doc-search/servers/<db-set-hash>.json` にhost/port/DB fingerprintを保存する。
- serverは `/health` と `/search` を提供する。
- serverは `127.0.0.1` bindを既定とし、初期実装ではdaemon化しない。
- registry上のfingerprintが古くてもlive serverへ問い合わせる。server側でDB更新を検出してreload判定する。

注意:

- server起動時にDB metadataを読み、vector系検索用のembedderを1回だけloadする。
- serverは検索ごとにDB fileのmtime/sizeを確認し、起動後にDBが変わっていればmetadataを再読込する。
- embedding metadataが起動時と互換ならfingerprintを更新し、embedderを維持したまま検索を続ける。
- embedding metadataが変わり互換性が崩れた場合はserver再起動を要求する。
- `search --device cpu` のようにserverと異なるdeviceを明示した場合、そのserverは使わずdirect検索する。

## MCP Server

`local-doc-search mcp` はCodexなどのコーディングエージェント向けのstdio MCP serverとして起動する。

```bash
uv run local-doc-search mcp --db notes.sqlite --device auto
```

Codex設定例:

```toml
[mcp_servers.local-doc-search]
command = "uv"
args = ["run", "local-doc-search", "mcp", "--db", "/absolute/path/to/notes.sqlite", "--device", "auto"]
cwd = "/absolute/path/to/local_search"
```

仕様:

- transportはstdio。
- JSON-RPC messageはJSON Linesと `Content-Length` framingの両方を受け付ける。
- `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `prompts/list` に応答する。
- toolは `search`, `codex_session_search`, `roots`。
- `search` toolは `--db` で指定された通常index DBを検索する。
- `codex_session_search` toolは固定DB `~/.codex/local-doc-search/codex-history.sqlite` のCodex session履歴を検索する。
- MCP serverもHTTP serverと同様にDB更新を検出し、embedding metadata互換ならreloadして検索を続ける。
- `search` / `codex_session_search` toolの引数:
  - `query`: semantic/vector検索用文字列。
  - `pattern`: SQLite FTS5 `MATCH` に渡すpattern。`AND`, `OR`, `NOT`, `NEAR` などを使える。
  - `query` と `pattern` の少なくとも一方が必須。
  - `mode`: `fts`, `vec`, `fts-vec`, `vec-fts`。省略時はCLIと同じく、`query`のみで`vec`、`pattern`のみで`fts`、両方指定で`vec-fts`。
  - `limit`: 結果数。デフォルトは10。
  - `candidates`: rerank前候補数。デフォルトは50。
  - `explain`: `fts_rank` / `vec_distance` を出力に含めるか。デフォルトはfalse。
- tool結果はJSON文字列として返す。
- 各resultには `path`, `relative_path`, `start_line`, `end_line`, `chunk_index`, `start_offset`, `end_offset`, `score`, `source`, `text` を含める。
- `codex_session_search` のresultには、Codex履歴由来の `session_id`, `cwd`, `role`, `turn_id`, `timestamp`, `session_path`, `line_no` も含める。
- `explain=true` の場合は `fts_rank`, `vec_distance` も含める。
- `roots` toolは引数なし。`--db` で指定された各DBについて `db_path` と `roots` を返す。
- `roots` の各要素には `root_path`, `file_count`, `chunk_count` を含める。

## Test/Quality Gate

現時点の確認コマンド:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

`pyright` は `typeCheckingMode = "standard"` で `src/local_doc_search` と `tests` を対象にする。

テストで確認している主な内容:

- 複数rootと拡張子filter
- 日本語trigram検索
- 3文字未満queryのLIKE fallback
- `fts-vec` / `vec-fts`
- 更新ファイルの再index
- DB metadataのembedding modelをsearchで使うこと
- rootからの相対path保存と検索結果への反映
- 複数DB検索結果のmerge
- embedding metadata互換性チェック
