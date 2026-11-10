package modeldownload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
)

const (
	rangeHeader         = "Range"
	contentRangeHeader  = "Content-Range"
	rangeUnit           = "bytes"
	unknownTotalLength  = "*"
	decimalBase         = 10
	integerBitSize      = 64
	clientErrorStatuses = 400
	serverErrorStatuses = 500
)

type responsePlan struct {
	offset     int64
	totalBytes int64
}

func runAttempt(
	ctx context.Context,
	request Request,
	partialPath string,
	reporter *progressReporter,
) (attemptOutcome, error) {
	existingSize, err := partialSize(ctx, partialPath)
	if err != nil {
		return outcomeFailed, err
	}
	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		request.URL,
		nil,
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"create download request failed",
			"url",
			request.URL,
			"err",
			err,
		)
		return outcomeFailed, fmt.Errorf(
			"create download request: %w",
			err,
		)
	}
	if existingSize > 0 {
		rangeValue := rangeUnit + "=" +
			strconv.FormatInt(existingSize, decimalBase) + "-"
		httpRequest.Header.Set(rangeHeader, rangeValue)
	}
	response, err := request.HTTPClient.Do(httpRequest)
	if err != nil {
		return transferOutcome(ctx), unavailable(ctx, request.URL, err)
	}
	plan, usable := planResponse(response, existingSize)
	if !usable {
		_ = response.Body.Close()
		return rejectResponse(ctx, request.URL, response, partialPath)
	}
	partialFile, hasher, err := openPartial(ctx, partialPath, plan.offset)
	if err != nil {
		_ = response.Body.Close()
		return outcomeFailed, err
	}
	reporter.begin(plan.offset, plan.totalBytes)
	_, copyErr := io.Copy(
		io.MultiWriter(partialFile, hasher, reporter),
		response.Body,
	)
	responseCloseErr := response.Body.Close()
	fileCloseErr := partialFile.Close()
	if copyErr != nil {
		return transferOutcome(ctx), unavailable(ctx, request.URL, copyErr)
	}
	if responseCloseErr != nil {
		return transferOutcome(ctx), unavailable(
			ctx,
			request.URL,
			responseCloseErr,
		)
	}
	if fileCloseErr != nil {
		return outcomeFailed, fileError(
			ctx,
			"close artifact",
			partialPath,
			fileCloseErr,
		)
	}
	reporter.finish()
	actualSHA256 := hex.EncodeToString(hasher.Sum(nil))
	return installPartial(ctx, request, partialPath, actualSHA256)
}

func transferOutcome(ctx context.Context) attemptOutcome {
	if ctx.Err() != nil {
		return outcomeFailed
	}
	return outcomeRetry
}

func partialSize(ctx context.Context, partialPath string) (int64, error) {
	info, err := os.Stat(partialPath)
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, fileError(
			ctx,
			"inspect artifact",
			partialPath,
			err,
		)
	}
	return info.Size(), nil
}

func planResponse(response *http.Response, existingSize int64) (responsePlan, bool) {
	switch response.StatusCode {
	case http.StatusOK:
		return responsePlan{
			offset:     0,
			totalBytes: max(response.ContentLength, 0),
		}, true
	case http.StatusPartialContent:
		start, totalBytes, parsed := parseContentRange(
			response.Header.Get(contentRangeHeader),
		)
		if !parsed || existingSize == 0 || start != existingSize {
			return responsePlan{offset: 0, totalBytes: 0}, false
		}
		return responsePlan{offset: existingSize, totalBytes: totalBytes}, true
	default:
		return responsePlan{offset: 0, totalBytes: 0}, false
	}
}

func parseContentRange(value string) (int64, int64, bool) {
	rangeAndTotal, hasUnit := strings.CutPrefix(value, rangeUnit+" ")
	if !hasUnit {
		return 0, 0, false
	}
	byteRange, total, hasTotal := strings.Cut(rangeAndTotal, "/")
	if !hasTotal {
		return 0, 0, false
	}
	first, last, hasLast := strings.Cut(byteRange, "-")
	if !hasLast {
		return 0, 0, false
	}
	start, err := strconv.ParseInt(first, decimalBase, integerBitSize)
	if err != nil {
		return 0, 0, false
	}
	if _, err := strconv.ParseInt(last, decimalBase, integerBitSize); err != nil {
		return 0, 0, false
	}
	if total == unknownTotalLength {
		return start, 0, true
	}
	totalBytes, err := strconv.ParseInt(total, decimalBase, integerBitSize)
	if err != nil {
		return 0, 0, false
	}
	return start, totalBytes, true
}

func rejectResponse(
	ctx context.Context,
	rawURL string,
	response *http.Response,
	partialPath string,
) (attemptOutcome, error) {
	statusErr := fmt.Errorf(
		"download artifact %s: %w: HTTP status %s",
		rawURL,
		ErrUnavailable,
		response.Status,
	)
	slog.ErrorContext(
		ctx,
		"download artifact failed",
		"url",
		rawURL,
		"status",
		response.Status,
		"err",
		statusErr,
	)
	status := response.StatusCode
	rangeRejected := status == http.StatusRequestedRangeNotSatisfiable ||
		status == http.StatusPartialContent
	if rangeRejected {
		if err := removePartial(ctx, partialPath); err != nil {
			return outcomeFailed, err
		}
		return outcomeRangeRejected, statusErr
	}
	if status >= clientErrorStatuses && status < serverErrorStatuses {
		return outcomeFailed, statusErr
	}
	return outcomeRetry, statusErr
}

func removePartial(ctx context.Context, partialPath string) error {
	err := os.Remove(partialPath)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return nil
	}
	slog.ErrorContext(ctx, "remove partial artifact failed", "path", partialPath, "err", err)
	return fmt.Errorf("remove partial artifact %s: %w", partialPath, err)
}

func openPartial(
	ctx context.Context,
	partialPath string,
	offset int64,
) (*os.File, hash.Hash, error) {
	hasher := sha256.New()
	flags := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	if offset > 0 {
		if err := hashFile(ctx, partialPath, hasher); err != nil {
			return nil, nil, err
		}
		flags = os.O_WRONLY | os.O_APPEND
	}
	partialFile, err := os.OpenFile(partialPath, flags, partialFileMode)
	if err != nil {
		return nil, nil, fileError(
			ctx,
			"create download file",
			partialPath,
			err,
		)
	}
	return partialFile, hasher, nil
}

func installPartial(
	ctx context.Context,
	request Request,
	partialPath string,
	actualSHA256 string,
) (attemptOutcome, error) {
	if actualSHA256 != request.SHA256 {
		checksumErr := fmt.Errorf(
			"artifact checksum mismatch for %s: got %s, want %s",
			request.URL,
			actualSHA256,
			request.SHA256,
		)
		slog.ErrorContext(
			ctx,
			"artifact checksum mismatch",
			"url",
			request.URL,
			"actual_sha256",
			actualSHA256,
			"expected_sha256",
			request.SHA256,
			"err",
			checksumErr,
		)
		if err := removePartial(ctx, partialPath); err != nil {
			return outcomeFailed, errors.Join(ErrUnavailable, checksumErr, err)
		}
		return outcomeFailed, fmt.Errorf("%w: %w", ErrUnavailable, checksumErr)
	}
	if err := os.Chmod(partialPath, artifactFileMode); err != nil {
		return outcomeFailed, fileError(
			ctx,
			"set artifact permissions",
			partialPath,
			err,
		)
	}
	if err := os.Rename(partialPath, request.DestinationPath); err != nil {
		return outcomeFailed, fileError(
			ctx,
			"install artifact",
			request.DestinationPath,
			err,
		)
	}
	return outcomeInstalled, nil
}
