# local-doc-search

`local-doc-search` is a local CLI for multilingual text search.

It builds a SQLite database from files under one or more directories and combines:

- SQLite FTS5 with `tokenize='trigram'`
- sqlite-vec `vec0`
- local embeddings via sentence-transformers or PLaMo custom code

Indexing shows progress for discovered files, skipped/unchanged files, embedding batches,
and processed chunks per second.
Indexing also prints a start summary with the command, DB, roots, model, device,
batch size, rebuild flags, extensions, and exclude patterns.
Normal incremental indexing commits after each changed file so interrupted runs keep completed
file updates. `--rebuild` keeps one transaction for consistency, while `--rebuild-offline`
clears the index and commits after each file for resumable offline rebuilds. Do not use
`--rebuild-offline` for a DB being served because interruption leaves a partial DB.
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
uv run local-doc-search index --db notes.sqlite --root ~/notes --rebuild-offline
uv run local-doc-search search --db notes.sqlite --query "検索したい内容"
uv run local-doc-search search --db notes.sqlite --pattern "検索 OR sqlite"
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --pattern "sqlite OR fts" --mode vec-fts --explain
uv run local-doc-search tui-search --db notes.sqlite --query "検索したい内容"
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
`Qwen/Qwen3-Embedding-0.6B` and `BAAI/bge-m3` are supported through the SentenceTransformer
backend as multilingual candidates. Qwen3 uses its retrieval instruction on queries and no
document prefix; bge-m3 uses no local prefix.
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
`local-doc-search tui-search` uses `fzf` to select a search result interactively and opens the
selected document at the hit line with `less` by default. Use `--pager` to choose another
command such as `vim`, `nvim`, or `bat`. The result list is shown as
`score source relative_path:start-end`, with the document text kept in the preview pane below
the result list by default. The hit chunk is placed near the upper part of the preview so long
files do not push it too far down. Use `--preview-window` to choose another fzf layout.

Apple Silicon Metal acceleration can be selected with `--device`.

```bash
uv run local-doc-search index --db notes.sqlite --root ~/notes --device auto --batch-size 32
uv run local-doc-search index --db notes.sqlite --root ~/notes --model Qwen/Qwen3-Embedding-0.6B --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model BAAI/bge-m3 --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model sbintuitions/sarashina-embedding-v2-1b --device auto
uv run local-doc-search index --db notes.sqlite --root ~/notes --model pfnet/plamo-embedding-1b --device auto
uv run local-doc-search search --db notes.sqlite --query "検索したい内容" --mode vec --device auto
```

`--device auto` uses MPS when PyTorch MPS is available and falls back to CPU otherwise.
Use `benchmark-embeddings` to compare embedding load time and warm encode speed without
building an index.

```bash
uv run local-doc-search benchmark-embeddings \
  --model Qwen/Qwen3-Embedding-0.6B \
  --model BAAI/bge-m3 \
  --model pfnet/plamo-embedding-1b \
  --model sbintuitions/sarashina-embedding-v2-1b \
  --device mps \
  --batch-size 16 \
  --documents 32 \
  --repeat 3 \
  --json
```

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
uv run local-doc-search search --query "検索したい内容" --mode vec
```

When `--db` is omitted, `search` uses the single live local server if exactly one
is registered. If the requested `--db` set is a subset of a live server's DB set,
that server is used and the search is limited to the requested DBs. Running servers
detect SQLite DB updates on search requests; if embedding metadata remains compatible,
they refresh fingerprints and continue without restarting.

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

Development checks:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

`pyright` checks both `src/local_doc_search` and `tests`.
