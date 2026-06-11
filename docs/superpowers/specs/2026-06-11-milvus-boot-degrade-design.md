# Degraded boot when Milvus is unreachable

## Problem

The daemon hangs forever at boot when Milvus is unreachable. The Milvus SDK
dials with `grpc.WithBlock()` (`client_config.go:30` in
`milvus/client/v2@v2.6.5`) and the boot context has no deadline, so
`grpc.DialContext` parks in `WaitForStateChange` before the gRPC server ever
serves. Reproduced in isolation with a goroutine dump: `main.run ->
daemon.NewManager -> semantic.NewService -> milvusclient.New -> DialContext ->
WaitForStateChange`. A fast dial error would also kill boot, since
`NewManager` propagates the error. Both shapes violate the graceful-start
contract. Hidden in prod because Milvus is always up at boot.

## Behavior

- Boot dial gets a 5 second limit. On failure the daemon neither hangs nor
  exits: it boots with the semantic service degraded (configured but not
  connected) and the store-unavailable banner shows from the first second.
- One background goroutine owns reconnection. It starts only when the boot
  dial failed. Full-jitter exponential backoff: base 2s, doubling, cap 5
  minutes, retry forever. Each attempt is itself bounded to 5 seconds. Only
  this goroutine ever dials. Logging: first attempt, success, then every 10th
  attempt.
- The hot path stays cold. `Available()` is an atomic load. The reconnector
  publishes the client by writing the field and then flipping the atomic
  flag, the standard lazy-publication pattern. No request dials, locks, or
  waits.
- Once a dial has succeeded, grpc-go manages channel reconnection itself, so
  runtime blips are already covered. This fixes only the never-connected-at-
  boot gap.
- Banner: boot failure records store-unavailable health. The health snapshot
  read clears it once `Available()` reports true.
- `Close()` cancels the reconnector.
- Untouched: empty `MilvusAddress` keeps the not-configured path; the
  embedder constructor does not dial; no proto or render changes.

## Tests

Closed port boots without hanging and reports unavailable; a fake Milvus
(empty gRPC server, whose Unimplemented Connect the SDK accepts) started
mid-test flips `Available()` to true via the reconnector; the backoff
schedule is asserted with an injected sleeper (growth and cap, no real
sleeping); `Close()` during backoff exits. Live proof: the isolated tmux
repro must answer `daemon status` within seconds and show the banner.
