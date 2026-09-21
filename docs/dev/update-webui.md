# Updating the web UI past `b9611`

The embedded server ships a committed snapshot of llama.cpp's web UI in `src/inferna/llama/server/assets/webui/`, pinned by `LLAMACPP_WEBUI_VERSION` in `scripts/manage.py`. The pin is stuck at `b9611`. Upstream changed the bucket layout, so moving past it needs a new fetcher and new server routes. A version bump alone does not work.

Findings below were measured on 2026-09-21 against bucket builds `b10964` (`v0.4.1`), `b10809` and `latest`.

## Why the current fetcher fails

`fetch_webui_assets()` downloads four flat files (`LLAMACPP_WEBUI_ASSETS`) and `checksums.txt` from `https://huggingface.co/buckets/ggml-org/llama-ui/resolve/<build>/`. For every build above `b9620`, all but `index.html` return 404. The bucket now publishes:

- `dist.tar.gz` -- the whole SvelteKit build output, 3.1 MB.
- `dist.tar.gz.sha256` -- `<sha256>  dist.tar.gz`. It matched the download for `b10964`.

## Contents of `dist.tar.gz` (`b10964`)

75 entries, 9.7 MB unpacked.

| Part | Paths | Raw | gzip -9 |
|-|-|-|-|
| App JS | `_app/immutable/bundle.<hash>.js` | 8.86 MB | 2.60 MB |
| App CSS | `_app/immutable/assets/bundle.<hash>.css` | 543 KB | 294 KB |
| Shell | `index.html` | -- | 1.3 KB |
| Metadata | `_app/version.json`, `build.json` (`{"version":"b10964"}`) | small | small |
| PWA | `manifest.webmanifest`, `sw.js`, `workbox-<hash>.js`, `pwa-*.png`, `maskable-icon-512x512.png` | small | small |
| Icons | `favicon.ico`, `favicon.svg`, `favicon-dark.svg`, `apple-touch-icon-180x180.png`, `recommended-mcp/*.{ico,png}` | small | small |
| Splash | 48 `apple-splash-*.png` | 304 KB | incompressible |

For comparison, the committed `b9611` snapshot is `bundle.js.gz` 2.37 MB plus `bundle.css.gz` 291 KB.

Layout changes that affect inferna:

- JS and CSS names carry a content hash, so they change every build. Routes cannot be hard-coded.
- `loading.html` no longer exists.
- `index.html` loads the app with `import("./_app/immutable/bundle.<hash>.js")`, relative to the page. The UI works only when served at the site root.
- The bundle calls `navigator.serviceWorker.register` for `sw.js`.

## Server contract

Endpoint strings extracted from both bundles and diffed:

- **Added:** `/models/sse`, `/v1/stream`, `/v1/streams/lookup`. Upstream registers them in `tools/server/server.cpp:242,302-304`.
- **Removed:** `/mcp-resources`.
- **Unchanged:** `/props`, `/slots`, `/health`, `/tokenize`, `/completion`, `/tools`, `/cors-proxy`, `/v1/models`, `/v1/chat/completions`, `/v1/chat/completions/control`, `/models/load`, `/models/unload`, and the MCP endpoints.

How the UI handles the added endpoints being absent:

- `/v1/streams/lookup` and `/v1/stream` resume a response stream after a page reload. `lookupStreamSessions` failures are caught and logged with `console.warn`, and the UI continues. Losing reload-resume is the only effect.
- `/models/sse` is read by `watchModelEvents`, which the router-model status code calls. It swallows errors and reconnects on a timer, so a missing route produces a 404 every retry interval. Not yet checked: whether single-model mode reaches this path.

## Changes required

1. **Fetcher** (`LlamaCppBuilder.fetch_webui_assets`, `scripts/manage.py`):
   - Download `dist.tar.gz` and `dist.tar.gz.sha256`. Fail on a checksum mismatch; do not skip verification when the `.sha256` is missing.
   - Extract an allowlist of files, not the whole tarball. Reject any member path that is absolute or contains `..`.
   - Gzip each kept file reproducibly (`_gzip_to`, `mtime=0`).
   - Write a manifest (`assets.json`: URL path -> file name, content type) next to the files, and keep `VERSION`.
   - Delete the previous snapshot's files first. Hashed names otherwise pile up in `assets/webui/`.
   - Drop `LLAMACPP_WEBUI_ASSETS` and replace the `CEILING` comment above `LLAMACPP_WEBUI_VERSION`.
2. **Routes** (`src/inferna/llama/server/embedded.py:538-545`): replace the four fixed routes with one lookup in a dict built from `assets.json` at import. A dict lookup cannot traverse paths. Serve `index.html` at `/` as now. Keep `no-cache` on HTML. `_app/immutable/*` can use `Cache-Control: public, max-age=31536000, immutable`, since the names are content-hashed.
3. **API-key public paths** (`src/inferna/llama/server/python.py:52`, `PUBLIC_PATHS`): derive the set from the same manifest instead of listing four names. This is security-relevant. Too narrow and `--api-key` breaks the UI; too wide and it exposes non-asset routes.
4. **Package data** (`CMakeLists.txt:760`): the install step globs `assets/webui/*.gz`, which is not recursive. Files under `_app/immutable/` would be left out of the wheel. Switch to `GLOB_RECURSE` or `install(DIRECTORY ...)`, and install `assets.json`.
5. **Tests**: update the route assertions in `tests/test_mserver_embedded.py`, `tests/test_server.py` and `tests/test_server_live_http.py`. Add tests for:
   - serving from the manifest;
   - 404 for paths outside it, including `..` forms;
   - the public-path set under `--api-key`;
   - the fetcher's member filtering and checksum failure, using a local tarball.

## Decisions to make

- **Which files to ship.** Proposed: `index.html`, `_app/**`, `favicon.*`, `favicon-dark.svg`, `apple-touch-icon-180x180.png`, `manifest.webmanifest`, `recommended-mcp/*`. That adds about 0.23 MB to the wheel over today. Shipping everything, splash PNGs included, adds about 0.5 MB.
- **Service worker.** Not serving `sw.js` makes registration fail, and the app still runs. Serving it lets browsers cache the UI, which can leave users on a stale UI after upgrading inferna. Proposed: do not serve it. This is general service-worker behaviour and has not been tested with this bundle.
- **The three new endpoints.** Proposed: leave them unimplemented at first and confirm the UI degrades as described above. Implement `/v1/stream` and `/v1/streams/lookup` later if reload-resume is wanted.

## Verification

- `manage.py fetch_webui` against a pinned build and against `latest`.
- `inferna server --webui` in a browser, with and without `--api-key`: chat, streaming, model list, page reload mid-stream.
- The browser network tab should show no repeated 404s. If `/models/sse` shows up, gate it or implement it.
- `make test`, and `scripts/rwt.py` on one wheel to confirm the assets ship.
