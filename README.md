# local-doc-search

`local-doc-search` is a local CLI for Japanese text search.

It builds a SQLite database from files under one or more directories and combines:

- SQLite FTS5 with `tokenize='trigram'`
- sqlite-vec `vec0`
- local embeddings via sentence-transformers or PLaMo custom code

Indexing shows progress for discovered files, skipped/unchanged files, embedding batches,
and processed chunks per second.
Short paragraphs are packed into chunks up to 600 characters by default; very long
paragraphs are split with 120-character overlap.
Packed paragraph chunks overlap by one paragraph when possible.
Chunking is selected through extension-based strategies so format-specific splitting can be
added without changing the indexing pipeline.
Markdown chunks do not cross ATX heading section boundaries.
Markdown chunks include the active parent heading path as context.
Markdown fenced code blocks are skipped when building chunks.

## Usage

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --ext .md --ext .txt
uv run local-doc-search search --db notes.sqlite --query "検索したい内容"
uv run local-doc-search search --db notes.sqlite --pattern "検索 OR sqlite"
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --pattern "sqlite OR fts" --mode vec-fts --explain
uv run local-doc-search info --db notes.sqlite
uv run local-doc-search files --db notes.sqlite
uv run local-doc-search files --db notes.sqlite --json
```

Exclude files by root-relative POSIX path regex:

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --exclude '^archive/' --exclude '\.tmp\.md$'
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
`--query` is the semantic/vector query. `--pattern` is passed to SQLite FTS5 `MATCH` and can
use FTS5 operators such as `AND`, `OR`, `NOT`, and `NEAR`. If `--mode` is omitted, query-only
search defaults to `vec`, pattern-only search defaults to `fts`, and query+pattern defaults
to `vec-fts`.
`local-doc-search files` lists the files currently stored in the SQLite index, including `path`,
`root_path`, `relative_path`, `size`, `mtime_ns`, and `content_hash`.
Without `--json`, it prints one indexed file per line as `key=value` fields.

Apple Silicon Metal acceleration can be selected with `--device`.

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run local-doc-search index --db notes.sqlite --root ~/notes --model sbintuitions/sarashina-embedding-v2-1b --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model pfnet/plamo-embedding-1b --device auto
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --mode vec --device auto
```

`--device auto` uses MPS when PyTorch MPS is available and falls back to CPU otherwise.

Multiple compatible DBs can be searched together.

```bash
uv run local-doc-search search --db notes.sqlite --db work.sqlite --query "検索したい内容" --mode fts-vec
```

Index and search Codex session history. The database is fixed at
`~/.codex/local-doc-search/codex-history.sqlite`, so `--db` is not required.
`codex-index` reads `~/.codex/sessions` by default and uses
`cl-nagoya/ruri-v3-310m` unless `--model` is specified. Results include
the source session path and JSONL line number for the indexed turn. Very long
turns are split before embedding to keep sequence length bounded. If the input
root does not exist, `codex-index` fails before creating or rebuilding the DB.

```bash
uv run local-doc-search codex-index --rebuild
uv run local-doc-search codex-server --device auto
uv run local-doc-search codex-search --query "以前相談した内容" --mode fts-vec
uv run local-doc-search codex-search --pattern "実装 OR エラー"
uv run local-doc-search codex-search --query "以前相談した内容" --json
```

Run a local server to avoid loading the embedding model for every query.

```bash
uv run local-doc-search server --db notes.sqlite --device auto
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --mode vec
```

Run a stdio MCP server for coding agents such as Codex.

```bash
uv run local-doc-search mcp --db notes.sqlite --device auto
```

Example Codex MCP configuration:

```toml
[mcp_servers.local-doc-search]
command = "uv"
args = ["run", "local-doc-search", "mcp", "--db", "/absolute/path/to/notes.sqlite", "--device", "auto"]
cwd = "/absolute/path/to/local_search"
```

The MCP server exposes:

- `search`: search indexed text with arguments matching `local-doc-search search`: `query`, `mode`,
  `limit`, `candidates`, and `explain`.
- `codex_session_search`: search indexed Codex session history from the fixed
  `~/.codex/local-doc-search/codex-history.sqlite` DB with the same search arguments.
- `roots`: list the indexed root directories for the DBs configured with `--db`.
