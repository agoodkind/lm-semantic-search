# Sandbox daemon

`lm-semantic-search-daemon sandbox` runs a daemon that touches none of the operator's files and neither shared backend. Use it to try a change against a real daemon while the installed one keeps running.

```bash
lm-semantic-search-daemon sandbox
```

It prints where it put everything, then serves until you stop it:

```
root      /tmp/lms-sandbox-a1b2c3  (removed on exit)
socket    /tmp/lms-sandbox-a1b2c3/daemon.sock
state     /tmp/lms-sandbox-a1b2c3/state
context   /tmp/lms-sandbox-a1b2c3/context
models    /Users/you/.local/state/lm-semantic-search
profile   offline
store     local
embedder  onnx
serving. ctrl-c to stop.
```

Drive it from another terminal with the socket it printed:

```bash
lm-semantic-search --socket /tmp/lms-sandbox-a1b2c3/daemon.sock status
lm-semantic-search --socket /tmp/lms-sandbox-a1b2c3/daemon.sock codebase index /path/to/repo --wait
lm-semantic-search --socket /tmp/lms-sandbox-a1b2c3/daemon.sock codebase search /path/to/repo "your query"
```

## What it isolates

The sandbox roots every path the daemon reads or writes inside one directory, including `mcp-sync.lock`. That lock is the reason the context root moves: a daemon that relocates only its state still competes for the operator's lock, and a daemon holding it blocks every indexing job in the daemon that wants it.

It also defaults to the `offline` profile, which replaces the shared vector store with an on-disk index and the hosted embedder with an in-process one. A sandbox therefore dials neither Milvus nor the embedding server, and spends no GPU time that the installed daemon is competing for.

The debug listener takes a kernel-assigned port rather than the fixed one the installed daemon binds, so both can run at once, and so can several sandboxes.

The one path that points outside the root is the model cache. A downloaded embedding model is checksum-verified and identical for every daemon, so the sandbox reads the machine's copy instead of downloading its own. Without that, every run would need network and pay for the download again.

## What it does not do

It is one process. It starts no child, writes no pid file, and cannot be detached. Ending the command ends the daemon.

It stops on Ctrl-C, on `kill`, and when its terminal closes. It also stops if whatever launched it dies without signalling it, which is what otherwise leaves a daemon serving with nobody left to stop it.

## Changing settings

Every setting the installed daemon accepts works here, through the same environment variables and flags. The sandbox only supplies defaults, and leaves alone any variable already set. Changing one keeps the isolation of the others.

Run against the real vector store and embedder instead of the local ones:

```bash
CLAUDE_CONTEXT_PROFILE=standard lm-semantic-search-daemon sandbox
```

Keep the directory after the run, to inspect logs or reuse the index:

```bash
lm-semantic-search-daemon sandbox --root /tmp/my-sandbox
```

A root you name survives; the default temporary root is removed when the command ends.

## Relationship to the test suites

`make live` and `make offline-live` build their daemons through the same resolver this command uses, so a suite and a hand-run sandbox cannot drift apart.

The two suites differ from the default only where their purpose requires it. `make offline-live` takes the sandbox defaults as they are. `make live` names the real Milvus and a fake embedder first, then takes the rest, which is what keeps it validating against the real store while staying isolated everywhere else.
