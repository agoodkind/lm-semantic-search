# Run the external import check

This fixture is a separate Go module that imports a published `goodkind.io/lm-semantic-search` commit through the module proxy. It tests `library.PrepareText` with the real bge-small tokenizer, and `library.Open` over `library/embedded` with the bge-small ONNX embedder. The fixture has no `replace` for LMS. Its workspace contains only this fixture and the pinned gksyntax checkout, never an LMS checkout. The commands below were verified on macOS arm64.

Run every command from the LMS repository root.

1. Build LMS. The build generates the Swift grammar and stages the native libraries under `.make/cgo/darwin-arm64`:

   ```sh
   make build
   ```

2. Resolve the published commit. Save `Version`, and confirm that `Origin.Hash` equals the full commit:

   ```sh
   COMMIT=<full published commit>
   GOWORK=off go list -m -json goodkind.io/lm-semantic-search@$COMMIT
   ```

3. Pin the fixture to that commit:

   ```sh
   GOWORK=off go -C test/import get goodkind.io/lm-semantic-search@$COMMIT
   GOWORK=off go -C test/import mod tidy
   ```

4. Confirm that the gksyntax checkout is the revision that the published commit pins. The two commands print the same hash:

   ```sh
   git ls-tree $COMMIT third_party/gksyntax
   git -C third_party/gksyntax rev-parse HEAD
   ```

5. Create the fixture workspace. Without `GOWORK=off`, `go work init` finds the repository root `go.work` and refuses to run. Delete an earlier `test/import/go.work` and `test/import/go.work.sum` first:

   ```sh
   GOWORK=off go -C test/import work init "$PWD/test/import" "$PWD/third_party/gksyntax"
   ```

6. Confirm that the workspace resolves the saved `Version` and prints no `Replace` field:

   ```sh
   go -C test/import list -m -json goodkind.io/lm-semantic-search
   ```

7. Run the tests with the CGO environment:

   ```sh
   export CGO_ENABLED=1
   export CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path'
   export PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig"
   go -C test/import test -count=1 ./...
   ```

The go command rejects the `-Wl,-rpath,@loader_path` directive in the LMS ONNX package unless `CGO_LDFLAGS_ALLOW` permits it. The `onnxruntime.pc` file in `PKG_CONFIG_PATH` adds the library directory for `libonnxruntime` and `libtokenizers`. It also adds an absolute run-time search path, and the test binary loads ONNX Runtime without `DYLD_LIBRARY_PATH`. The go build cache can hide a missing `PKG_CONFIG_PATH`; add `-a` to prove a fresh link.

The tests read bge-small from `lm-semantic-search-offline-model-test-cache` under the system temporary directory. LMS `make test` fills that cache. When the cache is empty, `onnx.NewTokenizer` downloads and verifies the pinned artifacts there.
