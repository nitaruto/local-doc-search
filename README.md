# tt-search

`tt-search` is a local CLI for Japanese text search.

It builds a SQLite database from files under one or more directories and combines:

- SQLite FTS5 with `tokenize='trigram'`
- sqlite-vec `vec0`
- local sentence-transformers embeddings

Indexing shows progress for discovered files, skipped/unchanged files, embedding work, and
processed chunks per second.
Short paragraphs are packed into chunks up to the configured character limit; very long
paragraphs are split with overlap.
Chunking is selected through extension-based strategies so format-specific splitting can be
added without changing the indexing pipeline.

## Usage

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --ext .md --ext .txt
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode fts-vec
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode vec-fts --explain
uv run tt-search info --db notes.sqlite
uv run tt-search files --db notes.sqlite
uv run tt-search files --db notes.sqlite --json
```

Exclude files by root-relative POSIX path regex:

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --exclude '^archive/' --exclude '\.tmp\.md$'
```

The default embedding model is `intfloat/multilingual-e5-small`.
Search uses the embedding model stored in the SQLite DB at index time.
Search output includes both the absolute `path` and the indexed root-relative `relative_path`.
Search output also includes chunk line ranges so other agents can locate the hit text.
`tt-search files` lists the files currently stored in the SQLite index, including `path`,
`root_path`, `relative_path`, `size`, `mtime_ns`, and `content_hash`.
Without `--json`, it prints one indexed file per line as `key=value` fields.

Apple Silicon Metal acceleration can be selected with `--device`.

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode vec --device auto
```

`--device auto` uses MPS when PyTorch MPS is available and falls back to CPU otherwise.

Multiple compatible DBs can be searched together.

```bash
uv run tt-search search --db notes.sqlite --db work.sqlite --query "検索したい内容" --mode fts-vec
```

Run a local server to avoid loading the embedding model for every query.

```bash
uv run tt-search server --db notes.sqlite --device auto
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode vec
```
