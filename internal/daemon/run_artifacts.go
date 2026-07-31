package daemon

import (
	"context"
	"errors"
	"log/slog"
	"os"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

// errLiveCheckpointLost names an absent live checkpoint on a codebase whose
// last completed run indexed files, which is the only shape in which a missing
// file means the index lost state on disk.
var errLiveCheckpointLost = errors.New("live merkle checkpoint is absent after a completed run that indexed files")

// runArtifacts is what a codebase's run record proves about the two artifacts a
// completed run leaves behind, the live merkle checkpoint and the live semantic
// collection. Both are written on the same condition: writeCompletedArtifacts
// writes the checkpoint only for a result carrying file hashes, promoteBootstrap
// promotes a collection only for a build that staged one, and promoteStagingMerkle
// tolerates an absent staging checkpoint only when the run committed nothing.
//
// The presence of a run record is not that evidence, and reading it as such is
// what reported healthy codebases as damaged: updateJobCompleted records
// LastSuccessfulRun for every completion, including a run that indexed zero
// files, which is what an empty repository, a fully ignored one, and one
// holding only oversized or unreadable files all produce.
//
// The closest existing type is collectionPresence, which carries the same
// three-valued shape over the same subject. It is deliberately not reused here,
// for the reason CollectionReadiness records in internal/status: keeping the
// types separate is what stops one being assigned into the other. The two
// answer different questions from different sources. collectionPresence is what
// a live store probe observed about one collection right now, so its unknown is
// a transport failure. runArtifacts is what the registry record proves about
// what a completed run left behind, so its unknown is a codebase with no
// completed run. Folding them would let a Milvus timeout read as "this run
// indexed nothing", which is the false-damage direction this whole rule exists
// to close.
type runArtifacts uint8

const (
	// runArtifactsUnknown means no completed run is recorded, so the record
	// proves nothing either way. A first build still in flight and a registry
	// entry from before the run summary was written both land here and they want
	// opposite handling, so each caller states which side of the unknown it takes.
	runArtifactsUnknown runArtifacts = iota
	// runArtifactsWritten means the last completed run indexed files, so it left
	// both a checkpoint and a promoted collection behind.
	runArtifactsWritten
	// runArtifactsNone means the last completed run indexed no file, so it left
	// neither. Their absence is this codebase's healthy shape.
	runArtifactsNone
)

// runArtifactsFor is the one place the run record is read for what it proves
// about a codebase's artifacts. Every gate that treats a missing checkpoint or
// a missing collection as damage reads it through here.
func runArtifactsFor(run *model.IndexRunSummary) runArtifacts {
	if run == nil {
		return runArtifactsUnknown
	}
	if run.IndexedFiles > 0 {
		return runArtifactsWritten
	}
	return runArtifactsNone
}

// expectsLiveCheckpoint reports whether a codebase should own a live merkle
// checkpoint on disk, which is the condition under which an absent file means
// the index lost state rather than that it never had one. Only positive
// evidence counts: an unknown record is read as a first build, whose absent
// checkpoint is expected.
func expectsLiveCheckpoint(run *model.IndexRunSummary) bool {
	return runArtifactsFor(run) == runArtifactsWritten
}

// expectsLiveCollection reports whether a completed run left a live semantic
// collection behind, which is the condition under which counting its rows can
// succeed. Only positive evidence counts here too, because the caller runs
// while a job is in flight and an unknown record there is a first build writing
// to staging, whose live collection does not exist yet.
//
// A codebase can own a collection without a run record, so callers holding the
// whole codebase use ownsLiveCollection instead.
func expectsLiveCollection(run *model.IndexRunSummary) bool {
	return runArtifactsFor(run) == runArtifactsWritten
}

// ownsLiveCollection reports whether a codebase should own a live semantic
// collection right now, reading the whole record rather than the run summary
// alone. Two shapes qualify and they reach the answer differently.
//
// A codebase whose last completed run indexed files promoted a collection, so
// its run record carries the evidence.
//
// An adopted codebase carries no run record at all and still owns a collection.
// adoptUnregisteredCodebase runs only for a path whose collection already
// exists, which is what makes it adoptable, and it records the codebase as
// indexed in the same critical section. Reading the run record alone left the
// adopted case counted as a first build, so an adopted codebase's rows went
// uncounted until its first sync completed. A first build is still excluded,
// because it writes to staging and is recorded as indexing rather than indexed
// until it promotes.
func ownsLiveCollection(codebase model.Codebase) bool {
	if expectsLiveCollection(codebase.LastSuccessfulRun) {
		return true
	}
	return codebase.LastSuccessfulRun == nil && codebase.Status == model.CodebaseStatusIndexed
}

// ranWithoutCreatingACollection reports positive evidence that a codebase's
// completed run promoted no collection, so there is no missing collection to
// repair. This is the opposite side of the unknown case from
// expectsLiveCollection: a registry entry with no run summary may well have had
// a collection that vanished, and the repair pass has to keep rebuilding it.
func ranWithoutCreatingACollection(run *model.IndexRunSummary) bool {
	return runArtifactsFor(run) == runArtifactsNone
}

// liveCheckpointState is the verdict on one live-checkpoint read.
type liveCheckpointState uint8

const (
	// liveCheckpointLoaded means a checkpoint file was read and matched the
	// config digest the read was made under.
	liveCheckpointLoaded liveCheckpointState = iota
	// liveCheckpointUnusable means a file was there but could not be used: it
	// failed to read or parse, or it belongs to a different index config. The
	// merkle package has already reported which.
	liveCheckpointUnusable
	// liveCheckpointNeverWritten means there is no file and none was ever
	// written, because the codebase's last completed run indexed no file. The
	// absence is this codebase's healthy shape and no surface may call it a
	// fault.
	liveCheckpointNeverWritten
	// liveCheckpointLost means there is no file but one should exist, so the
	// index lost state. loadLiveCheckpoint reports it to the operator.
	liveCheckpointLost
)

// liveCheckpoint is the result of one read of a codebase's live merkle
// checkpoint.
type liveCheckpoint struct {
	// snapshot is the checkpoint's content, or an empty snapshot stamped with
	// the config digest the read was made under when there was none to use.
	snapshot merkle.Snapshot
	// state says what the read found, so a caller never re-derives the
	// expectation from the codebase record itself.
	state liveCheckpointState
}

// usable reports whether the snapshot carries a real checkpoint's contents.
func (checkpoint liveCheckpoint) usable() bool {
	return checkpoint.state == liveCheckpointLoaded
}

// loadLiveCheckpoint is the one production read of a codebase's live merkle
// checkpoint. It resolves the snapshot path, applies expectsLiveCheckpoint, and
// reports an absent file as lost state exactly when the codebase should own it.
//
// Nothing below this function can produce that report, which is what keeps the
// rule from being reimplemented per call site: merkle treats every absent
// checkpoint as an empty one and logs it at debug, so a reader that skipped
// this function would go quiet on a real loss rather than shout on a healthy
// empty codebase. TestLiveCheckpointReadsGoThroughOneChokepoint holds the other
// half, failing when a daemon source file outside this one reads a snapshot
// directly.
//
// It logs its own diagnostic rather than handing one back, the same shape
// probeCollectionEvidence uses for a failed store inspect, so the report cannot
// drift from the read that produced it.
//
// configDigest is passed rather than read off the codebase because a running
// job reads its checkpoint under the config the build is writing under, which
// is not always the record's current one.
func (manager *Manager) loadLiveCheckpoint(ctx context.Context, codebase model.Codebase, configDigest string) liveCheckpoint {
	snapshotPath := manager.snapshotPathForCodebase(codebase)
	snapshot, err := merkle.LoadSnapshotForConfig(snapshotPath, configDigest, manager.legacyDigestForCodebase(codebase.ID))
	switch {
	case err == nil:
		return liveCheckpoint{snapshot: snapshot, state: liveCheckpointLoaded}
	case !errors.Is(err, os.ErrNotExist):
		return liveCheckpoint{snapshot: snapshot, state: liveCheckpointUnusable}
	case expectsLiveCheckpoint(codebase.LastSuccessfulRun):
		slog.ErrorContext(
			ctx,
			"read Merkle snapshot failed",
			"codebase_id", codebase.ID,
			"path", snapshotPath,
			"err", errLiveCheckpointLost,
		)
		return liveCheckpoint{snapshot: snapshot, state: liveCheckpointLost}
	default:
		return liveCheckpoint{snapshot: snapshot, state: liveCheckpointNeverWritten}
	}
}
