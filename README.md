# tt-search

`tt-search` is a local CLI for Japanese text search.

It builds a SQLite database from files under one or more directories and combines:

- SQLite FTS5 with `tokenize='trigram'`
- sqlite-vec `vec0`
- local sentence-transformers embeddings

## Usage

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --ext .md --ext .txt
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode fts-vec
uv run tt-search search --db notes.sqlite --query "検索したい内容" --mode vec-fts --explain
uv run tt-search info --db notes.sqlite
```

The default embedding model is `intfloat/multilingual-e5-small`.
Search uses the embedding model stored in the SQLite DB at index time.
Search output includes both the absolute `path` and the indexed root-relative `relative_path`.
