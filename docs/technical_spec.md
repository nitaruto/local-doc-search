# tt-search 技術仕様

## 概要

`tt-search` は、日本語を含むローカルテキストファイルを検索するCLI。
SQLiteを永続ストアとして使い、FTS5 trigramによる文字列検索とsqlite-vecによるベクトル検索を組み合わせる。

主なコマンド:

- `tt-search index`: 指定root配下のファイルをSQLite DBへindexする。
- `tt-search search`: FTS/vec/hybrid検索を実行する。
- `tt-search info`: DB metadataと件数を表示する。

パッケージ管理と実行は `uv` を前提にする。

## DB構造

SQLite DBには以下を保存する。

- `metadata`
  - `schema_version`
  - `embedding_model`
  - `embedding_dim`
  - `created_at`
- `files`
  - 絶対path: `path`
  - index時のroot: `root_path`
  - rootからの相対path: `relative_path`
  - 差分判定用: `size`, `mtime_ns`, `content_hash`
- `chunks`
  - fileごとのchunk本文
  - `chunk_index`, `start_offset`, `end_offset`
- `chunks_fts`
  - `fts5(path UNINDEXED, text, tokenize='trigram')`
- `chunk_vec`
  - `vec0(embedding float[N])`
  - `N` はindex時のembedding dimension

既存DBに `root_path` / `relative_path` がない場合は、起動時に不足カラムを追加する。

## Indexing

`tt-search index --db path.sqlite --root DIR [--root DIR...] [--ext .md ...]` でindexする。

処理内容:

- rootは複数指定できる。
- rootがファイルの場合は、そのファイルの親ディレクトリをroot扱いにする。
- 拡張子指定がある場合は、その拡張子だけ対象にする。
- デフォルト対象拡張子は `.txt`, `.md`, `.markdown`, `.rst`。
- `.git`, `.venv`, `__pycache__`, dot directory は走査対象から除外する。
- UTF-8 / UTF-8 BOMとして読めないファイルはskipする。
- ファイル本文は段落単位でchunk化する。
- 長すぎる段落は文字数上限で分割し、少しoverlapさせる。
- document chunkは `passage: ...` prefixでembeddingする。

## 差分更新

indexは差分更新に対応している。

DBに保存済みの `root_path`, `relative_path`, `size`, `mtime_ns`, `content_hash` が現在のファイルと一致する場合、そのファイルは再indexしない。

ファイル状態ごとの挙動:

- 新規ファイル: chunk/FTS/vectorを新規作成する。
- 更新ファイル: そのファイルの既存chunk/FTS/vectorを削除し、ファイル単位で再作成する。
- 削除ファイル: DBから該当file/chunk/FTS/vectorを削除する。
- 変更なし: 再チャンク化、再embedding、FTS/vector更新をskipする。

現在の実装上の注意:

- 変更ファイル内の一部chunkだけを再embeddingする粒度ではなく、更新されたファイル全体を再indexする。
- index実行ごとにroot配下のファイル一覧走査は行う。
- 現状はhash計算のためにファイル本文を読む。大規模ディレクトリ向けには、`size + mtime_ns` が一致した時点でhash計算もskipする余地がある。

## Embedding

デフォルトモデルは `intfloat/multilingual-e5-small`。

index時:

- `--model` でsentence-transformers model名を指定できる。
- 指定モデル名、dimensionはDB metadataへ保存する。

search時:

- `--model` は指定しない。
- vector系検索ではDB metadataの `embedding_model` を読み、そのモデルでquery embeddingを生成する。
- queryは `query: ...` prefixでembeddingする。
- DBにembedding metadataがない場合は、reindexを促すエラーにする。

## Search Modes

`tt-search search --mode` は以下に対応する。

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
- snippet
- `--explain` 指定時:
  - `fts_rank`
  - `vec_distance`

`--json` 指定時は `SearchResult` の内容をJSON配列として出力する。

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
