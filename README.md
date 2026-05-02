# tt-search

`tt-search` is a local CLI for Japanese text search.

It builds a SQLite database from files under one or more directories and combines:

- SQLite FTS5 with `tokenize='trigram'`
- sqlite-vec `vec0`
- local embeddings via sentence-transformers or PLaMo custom code

Indexing shows progress for discovered files, skipped/unchanged files, embedding work, and
processed chunks per second.
Short paragraphs are packed into chunks up to 600 characters by default; very long
paragraphs are split with 120-character overlap.
Packed paragraph chunks overlap by one paragraph when possible.
Chunking is selected through extension-based strategies so format-specific splitting can be
added without changing the indexing pipeline.
Markdown chunks do not cross ATX heading section boundaries.

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
For Japanese-focused quality, `cl-nagoya/ruri-v3-*` is the preferred experimental upgrade path.
`sbintuitions/sarashina-embedding-v2-1b` is also supported through the SentenceTransformer
backend with its recommended retrieval prefixes. Check the model license before use because
it is distributed under the Sarashina non-commercial license.
`pfnet/plamo-embedding-1b` is supported for experiments, but it is not recommended for normal
use because its custom `encode_document` / `encode_query` path can return non-finite vectors
on both CPU and MPS and therefore requires retry handling. The PLaMo backend loads the model
as `bfloat16` and stores sqlite-vec embeddings as `float32`.
Search output includes both the absolute `path` and the indexed root-relative `relative_path`.
Search output also includes chunk line ranges so other agents can locate the hit text.
`tt-search files` lists the files currently stored in the SQLite index, including `path`,
`root_path`, `relative_path`, `size`, `mtime_ns`, and `content_hash`.
Without `--json`, it prints one indexed file per line as `key=value` fields.

Apple Silicon Metal acceleration can be selected with `--device`.

```bash
uv run tt-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run tt-search index --db notes.sqlite --root ~/notes --model sbintuitions/sarashina-embedding-v2-1b --device auto
uv run tt-search index --db notes.sqlite --root ~/notes --model pfnet/plamo-embedding-1b --device auto
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

Run a stdio MCP server for coding agents such as Codex.

```bash
uv run tt-search mcp --db notes.sqlite --device auto
```

Example Codex MCP configuration:

```toml
[mcp_servers.tt-search]
command = "uv"
args = ["run", "tt-search", "mcp", "--db", "/absolute/path/to/notes.sqlite", "--device", "auto"]
cwd = "/absolute/path/to/local_search"
```

The MCP server exposes:

- `search`: search indexed text with arguments matching `tt-search search`: `query`, `mode`,
  `limit`, `candidates`, and `explain`.
- `roots`: list the indexed root directories for the DBs configured with `--db`.
