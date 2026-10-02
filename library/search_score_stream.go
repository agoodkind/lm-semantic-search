package library

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"math"
	"os"
	"sync"
)

const scoreFrameOverhead = 16

//go:embed search_stream_score.sql
var searchStreamScoreStatement string

type scoreStream struct {
	file            *os.File
	writer          *bufio.Writer
	digest          hash.Hash
	mutex           sync.Mutex
	reserved, bytes int64
	closed, removed bool
}

func newScoreStream(ctx context.Context, queryPath string) (*scoreStream, error) {
	file, err := os.OpenFile(queryPath+".scores", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if err != nil {
		slog.ErrorContext(ctx, "create verified score stream failed", "err", err)
		return nil, fmt.Errorf("create verified score stream: %w", err)
	}
	return &scoreStream{file: file, writer: bufio.NewWriterSize(file, 32*1024), digest: sha256.New()}, nil
}

func (stream *scoreStream) reserve(bytes int64) {
	stream.mutex.Lock()
	defer stream.mutex.Unlock()
	stream.reserved += bytes
}

func (stream *scoreStream) write(ctx context.Context, scores []VectorScore) (err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "write verified score stream failed", "err", err)
		}
	}()
	stream.mutex.Lock()
	defer stream.mutex.Unlock()
	for _, score := range scores {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("write verified score: %w", err)
		}
		size := int64(len(score.ID)) + scoreFrameOverhead
		if size > stream.reserved-stream.bytes {
			return fmt.Errorf("%w: verified score stream exceeds its byte reservation", ErrResourceLimit)
		}
		var length, bits [8]byte
		binary.LittleEndian.PutUint64(length[:], uint64(len(score.ID)))
		binary.LittleEndian.PutUint64(bits[:], math.Float64bits(score.Score))
		for _, part := range [][]byte{length[:], []byte(score.ID), bits[:]} {
			if _, err := stream.writer.Write(part); err != nil {
				return fmt.Errorf("write verified score stream: %w", err)
			}
			stream.digest.Write(part)
		}
		stream.bytes += size
	}
	return nil
}

func readStreamScore(ctx context.Context, reader *bufio.Reader, remaining int64) (_ VectorScore, _ int64, err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "decode verified score record failed", "err", err)
		}
	}()
	var length, bits [8]byte
	if _, err := io.ReadFull(reader, length[:]); err != nil {
		return VectorScore{}, 0, fmt.Errorf("read score ID length: %w", err)
	}
	count := binary.LittleEndian.Uint64(length[:])
	if remaining < scoreFrameOverhead || count > uint64(remaining-scoreFrameOverhead) || count > math.MaxInt64 || count > uint64(math.MaxInt) {
		return VectorScore{}, 0, errors.New("score ID exceeds the remaining stream bytes")
	}
	id := make([]byte, int(count))
	if _, err := io.ReadFull(reader, id); err != nil {
		return VectorScore{}, 0, fmt.Errorf("read score ID: %w", err)
	}
	if _, err := io.ReadFull(reader, bits[:]); err != nil {
		return VectorScore{}, 0, fmt.Errorf("read score bits: %w", err)
	}
	score := math.Float64frombits(binary.LittleEndian.Uint64(bits[:]))
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return VectorScore{}, 0, errors.New("stored exact score is not finite")
	}
	return VectorScore{ID: string(id), Score: score}, int64(count) + scoreFrameOverhead, nil
}

func (stream *scoreStream) importScores(ctx context.Context, query *queryDatabase) (err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "import verified score stream failed", "err", err)
		}
	}()
	if err := stream.writer.Flush(); err != nil {
		return fmt.Errorf("flush verified score stream: %w", err)
	}
	if _, err := stream.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind verified score stream: %w", err)
	}
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return queryDatabaseError(ctx, "begin streamed score import", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	update, err := writer.PrepareContext(ctx, searchStreamScoreStatement)
	if err != nil {
		return queryDatabaseError(ctx, "prepare streamed scores", err)
	}
	defer func() { err = errors.Join(err, closeStatement(ctx, update)) }()
	digest := sha256.New()
	reader := bufio.NewReaderSize(io.TeeReader(stream.file, digest), 32*1024)
	var consumed int64
	for consumed < stream.bytes {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("import verified score: %w", err)
		}
		score, size, err := readStreamScore(ctx, reader, stream.bytes-consumed)
		if err != nil {
			return fmt.Errorf("%w: decode verified score stream: %w", ErrVectorCorrupt, err)
		}
		consumed += size
		result, err := update.ExecContext(ctx, score.Score, score.ID)
		if err != nil {
			return queryDatabaseError(ctx, "save streamed vector score", err)
		}
		count, err := result.RowsAffected()
		if err != nil {
			return queryDatabaseError(ctx, "check streamed vector score", err)
		}
		if count != 1 {
			return fmt.Errorf("%w: streamed vector %s is missing or repeated", ErrVectorCorrupt, score.ID)
		}
	}
	if _, err := reader.ReadByte(); err == nil {
		return fmt.Errorf("%w: verified score stream has trailing data", ErrVectorCorrupt)
	} else if !errors.Is(err, io.EOF) {
		return fmt.Errorf("%w: read verified score stream end: %w", ErrVectorCorrupt, err)
	}
	if !bytes.Equal(digest.Sum(nil), stream.digest.Sum(nil)) {
		return fmt.Errorf("%w: verified score stream checksum differs", ErrVectorCorrupt)
	}
	if err := writer.Commit(); err != nil {
		return queryDatabaseError(ctx, "commit streamed scores", err)
	}
	return nil
}

func (stream *scoreStream) close(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			slog.ErrorContext(ctx, "close verified score stream failed", "err", err)
		}
	}()
	if !stream.closed {
		err = stream.file.Close()
		stream.closed = true
	}
	if !stream.removed {
		removeErr := os.Remove(stream.file.Name())
		if removeErr == nil || errors.Is(removeErr, os.ErrNotExist) {
			stream.removed = true
		} else {
			err = errors.Join(err, removeErr)
		}
	}
	return err
}
