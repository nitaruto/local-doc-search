# tt-search 技術仕様

## 概要

`tt-search` は、日本語を含むローカルテキストファイルを検索するCLI。
SQLiteを永続ストアとして使い、FTS5 trigramによる文字列検索とsqlite-vecによるベクトル検索を組み合わせる。

主なコマンド:

- `tt-search index`: 指定root配下のファイルをSQLite DBへindexする。
- `tt-search search`: FTS/vec/hybrid検索を実行する。
- `tt-search info`: DB metadataと件数を表示する。
- `tt-search files`: DBにindex済みのファイル一覧を表示する。
- `tt-search server`: ローカル検索serverを起動し、queryごとのモデルロードを避ける。

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
- `chunks_fts`
  - `fts5(path UNINDEXED, text, tokenize='trigram')`
- `chunk_vec`
  - `vec0(embedding float[N])`
  - `N` はindex時のembedding dimension

既存DBに `root_path` / `relative_path` / `start_line` / `end_line` がない場合は、起動時に不足カラムを追加する。
古いDBへ追加した行番号は既定値になるため、正確な行番号が必要な場合は再indexする。

## Indexing

`tt-search index --db path.sqlite --root DIR [--root DIR...] [--ext .md ...] [--exclude REGEX ...]` でindexする。

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
  - fenced code block内の見出し風行はsection境界として扱わない。
  - `.txt`, `.rst` は `paragraph-pack` strategyを使う。
  - 将来はreStructuredText構造単位、code block考慮などを拡張子別に追加できる。
- document chunkは `passage: ...` prefixでembeddingする。
  - 実際のprefixはmodelごとのprefix policyで決まる。
- index中は候補ファイル数、処理済みファイル数、現在の状態、embedding対象chunk数、累計処理chunk数、1秒あたりの処理chunk数をprogress表示する。

exclude例:

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --exclude '^archive/' --exclude '\.tmp\.md$'
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

対象モデルとprefix policy:

- `intfloat/multilingual-e5-small`: query=`query: `, passage=`passage: `
- `cl-nagoya/ruri-v3-*`: query=`検索クエリ: `, passage=`検索文書: `
- `pfnet/plamo-embedding-1b`: `plamo-custom` backendでmodel card推奨の `AutoModel.encode_query` / `AutoModel.encode_document` を使う。prefixはPLaMo model側で扱うためtt-search側では付与しない。

推奨:

- 安定運用では `intfloat/multilingual-e5-small` を基本とする。
- 日本語精度を重視する場合は `cl-nagoya/ruri-v3-*` を優先候補とする。
- `pfnet/plamo-embedding-1b` は現状、CPU/MPSともに非有限値を返す確率が高く、retry前提になるため実験用扱いとする。通常利用では推奨しない。

現在のbackend:

- `sentence-transformers`
  - `intfloat/multilingual-e5-small`, `cl-nagoya/ruri-v3-*` など。
- `plamo-custom`
  - `pfnet/plamo-embedding-1b` 専用。
  - `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` と `AutoModel.from_pretrained(..., trust_remote_code=True)` を使う。
  - model configの `torch_dtype=bfloat16` に合わせ、modelは明示的に `bfloat16` でloadする。sqlite-vecへ保存するembeddingは既存通り `float32` へ変換する。
  - PLaMo custom codeのRotaryEmbedding cacheがload後に確率的に壊れることがあるため、device移動後、dimension probe前に全layerのrotary cacheを再生成する。
  - `--device auto` は他backendと同じく、PyTorch MPSが利用可能なら `mps`、不可なら `cpu` を使う。
  - PLaMoの `encode_document` / `encode_query` はCPUでも確率的に非有限値を返すことがあるため、非有限値の場合のみwarningを出して最大5回retryする。全試行失敗した場合はエラーにし、不正vectorは保存しない。
  - document chunkは `encode_document(texts, tokenizer)`、queryは `encode_query(text, tokenizer)` でembeddingする。
  - 公式要件として `sentencepiece` が必要。

利用例:

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run tt-search index --db notes.sqlite --root ~/notes --model cl-nagoya/ruri-v3-70m --device mps
uv run tt-search index --db notes.sqlite --root ~/notes --model pfnet/plamo-embedding-1b --device auto
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode vec --device auto
```

## Search Modes

`tt-search search --db A.sqlite [--db B.sqlite...] --mode` は以下に対応する。

- `fts`
  - FTS5 trigramだけで検索する。
- `vec`
  - sqlite-vec `vec0` のKNNだけで検索する。
- `fts-vec`
  - FTS/LIKEで候補を取り、query vectorとの距離でrerankする。
- `vec-fts`
  - vector検索で候補を取り、FTS rankまたはLIKE一致でrerankする。

3文字未満のqueryはFTS5 trigramの `MATCH` では扱いづらいため、`LIKE` fallbackで候補取得または文字一致スコア計算を行う。

## Output

検索結果には以下を含める。

- `score`
- 絶対path: `path`
- index時rootからの相対path: `relative_path`
- chunk番号: `chunk_index`
- 行番号範囲: `start_line`, `end_line`
- snippet
- `--explain` 指定時:
  - `fts_rank`
  - `vec_distance`

`--json` 指定時は `SearchResult` の内容をJSON配列として出力する。
他エージェントから利用する場合は、`path`, `relative_path`, `start_line`, `end_line`, `text` を参照すると、該当ファイル内の検索hit範囲を特定できる。

## Indexed File Listing

`tt-search files --db notes.sqlite` は、DBの `files` テーブルに保存されているindex済みファイル一覧を表示する。

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

`tt-search search` は `--db` を複数指定できる。

```bash
uv run tt-search search --db notes.sqlite --db work.sqlite --query "検索語" --mode fts-vec
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

`tt-search server` はローカルHTTP serverをforegroundで起動する。

```bash
uv run tt-search server --db notes.sqlite --db work.sqlite --device auto
```

`tt-search search` は同じDB集合のserverが起動中ならserverへ問い合わせる。
serverがない、応答しない、DB fingerprintが一致しない場合はdirect検索へfallbackする。

```bash
uv run tt-search search --db notes.sqlite --db work.sqlite --query "検索語" --mode fts-vec
uv run tt-search search --db notes.sqlite --query "検索語" --mode vec --no-server
```

server discovery:

- `~/.cache/tt-search/servers/<db-set-hash>.json` にhost/port/DB fingerprintを保存する。
- serverは `/health` と `/search` を提供する。
- serverは `127.0.0.1` bindを既定とし、初期実装ではdaemon化しない。

注意:

- server起動時にDB metadataを読み、vector系検索用のembedderを1回だけloadする。
- indexでDBを更新した後はserver再起動を推奨する。
- serverは検索ごとにDB fileのmtime/sizeを確認し、起動後にDBが変わっていればエラーにする。
- `search --device cpu` のようにserverと異なるdeviceを明示した場合、そのserverは使わずdirect検索する。

## MCP Server

`tt-search mcp` はCodexなどのコーディングエージェント向けのstdio MCP serverとして起動する。

```bash
uv run tt-search mcp --db notes.sqlite --device auto
```

Codex設定例:

```toml
[mcp_servers.tt-search]
command = "uv"
args = ["run", "tt-search", "mcp", "--db", "/absolute/path/to/notes.sqlite", "--device", "auto"]
cwd = "/absolute/path/to/local_search"
```

仕様:

- transportはstdio。
- JSON-RPC messageはJSON Linesと `Content-Length` framingの両方を受け付ける。
- `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `prompts/list` に応答する。
- toolは `search` と `roots`。
- `search` toolの引数:
  - `query`: 必須文字列。
  - `mode`: `fts`, `vec`, `fts-vec`, `vec-fts`。デフォルトは `fts-vec`。
  - `limit`: 結果数。デフォルトは10。
  - `candidates`: rerank前候補数。デフォルトは50。
  - `explain`: `fts_rank` / `vec_distance` を出力に含めるか。デフォルトはfalse。
- tool結果はJSON文字列として返す。
- 各resultには `path`, `relative_path`, `start_line`, `end_line`, `chunk_index`, `start_offset`, `end_offset`, `score`, `source`, `text` を含める。
- `explain=true` の場合は `fts_rank`, `vec_distance` も含める。
- `roots` toolは引数なし。`--db` で指定された各DBについて `db_path` と `roots` を返す。
- `roots` の各要素には `root_path`, `file_count`, `chunk_count` を含める。

## Test/Quality Gate

現時点の確認コマンド:

```bash
uv run pytest
uv run ruff check .
```

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
