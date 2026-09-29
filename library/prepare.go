package library

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/config"
)

// PrepareText splits one selected source text into ordered parts that each fit
// the embedding model's limits, and returns each part's exact transformed
// document input. It never truncates the text and never embeds or writes
// anything.
//
// Every part's EmbeddingInput is DocumentPrefix followed by the part's bytes of
// Text, and that whole input fits MaxTokens and MaxBytes. With a Tokenizer, the
// token limit applies to the counted input. Without one, the existing
// conservative conversion turns MaxTokens into a byte budget. Parts cut Text at
// the largest fitting UTF-8 boundary, in order, and their Suffix values are the
// decimal ordinals "0", "1", and so on. The same request always returns the
// same parts.
//
// A cut that isolates a span of whitespace alone produces no part, because no
// vector can describe it. Every non-whitespace byte of Text is in exactly one
// part. A request that fails validation, or a character that cannot fit next
// to the prefix, returns an error that wraps [ErrInvalidRequest].
func PrepareText(ctx context.Context, request PrepareRequest) ([]PreparedPart, error) {
	limits, err := prepareLimits(request)
	if err != nil {
		return nil, err
	}
	parts := make([]PreparedPart, 0, 1)
	start := 0
	for start < len(request.Text) {
		if err := ctx.Err(); err != nil {
			slog.WarnContext(ctx, "prepare text cancelled", "parts", len(parts), "err", err)
			return nil, fmt.Errorf("prepare text: %w", err)
		}
		end, err := largestFittingEnd(ctx, request, limits, start)
		if err != nil {
			return nil, err
		}
		if end == start {
			return nil, invalidRequest(fmt.Sprintf(
				"prepare text: the character at byte %d does not fit the model limits after the %d-byte document prefix",
				start,
				len(request.DocumentPrefix),
			))
		}
		if strings.TrimSpace(request.Text[start:end]) != "" {
			parts = append(parts, PreparedPart{
				Suffix:         strconv.Itoa(len(parts)),
				ByteStart:      start,
				ByteEnd:        end,
				EmbeddingInput: request.DocumentPrefix + request.Text[start:end],
			})
		}
		start = end
	}
	return parts, nil
}

// prepareLimit bounds one embedding input. A zero field applies no limit of
// that kind.
type prepareLimit struct {
	bytes  int
	tokens int
}

func prepareLimits(request PrepareRequest) (prepareLimit, error) {
	if !utf8.ValidString(request.Text) {
		return prepareLimit{}, invalidRequest("prepare text: Text is not valid UTF-8")
	}
	if strings.TrimSpace(request.Text) == "" {
		return prepareLimit{}, invalidRequest("prepare text: Text contains no non-whitespace character")
	}
	if !utf8.ValidString(request.DocumentPrefix) {
		return prepareLimit{}, invalidRequest("prepare text: DocumentPrefix is not valid UTF-8")
	}
	if request.MaxTokens < 0 || request.MaxBytes < 0 {
		return prepareLimit{}, invalidRequest(fmt.Sprintf(
			"prepare text: MaxTokens %d and MaxBytes %d must not be negative",
			request.MaxTokens,
			request.MaxBytes,
		))
	}
	if request.MaxTokens == 0 && request.MaxBytes == 0 {
		return prepareLimit{}, invalidRequest("prepare text: set MaxTokens, MaxBytes, or both")
	}
	if request.Tokenizer != nil && request.MaxTokens == 0 {
		return prepareLimit{}, invalidRequest("prepare text: a Tokenizer requires a positive MaxTokens")
	}

	limits := prepareLimit{bytes: request.MaxBytes, tokens: 0}
	if request.Tokenizer != nil {
		limits.tokens = request.MaxTokens
	} else if request.MaxTokens > 0 {
		derivedBytes := config.EmbedChunkByteBudgetForLimit(0, request.MaxTokens)
		if limits.bytes == 0 || derivedBytes < limits.bytes {
			limits.bytes = derivedBytes
		}
	}
	if limits.bytes > 0 && limits.bytes <= len(request.DocumentPrefix) {
		return prepareLimit{}, invalidRequest(fmt.Sprintf(
			"prepare text: the %d-byte document prefix leaves no room in the %d-byte input budget",
			len(request.DocumentPrefix),
			limits.bytes,
		))
	}
	return limits, nil
}

// largestFittingEnd returns the largest UTF-8 boundary end after start at which
// the prefixed input for Text[start:end] fits limits, or start when no
// character fits. The token search doubles a window from start until the
// window stops fitting and then bisects. Each count reads at most twice the
// bytes of the returned part.
func largestFittingEnd(
	ctx context.Context,
	request PrepareRequest,
	limits prepareLimit,
	start int,
) (int, error) {
	text := request.Text
	upper := len(text)
	if limits.bytes > 0 {
		upper = min(upper, start+limits.bytes-len(request.DocumentPrefix))
		upper = runeBoundaryAtOrBefore(text, upper)
	}
	if limits.tokens == 0 || upper == start {
		return upper, nil
	}

	fits := func(end int) (bool, error) {
		count, err := request.Tokenizer.CountTokens(ctx, request.DocumentPrefix+text[start:end])
		if err != nil {
			slog.WarnContext(ctx, "prepare text token count failed", "input_bytes", len(request.DocumentPrefix)+end-start, "err", err)
			return false, fmt.Errorf("prepare text: count tokens: %w", err)
		}
		return count <= limits.tokens, nil
	}

	fitting := start
	notFitting := -1
	step := limits.tokens
	for notFitting < 0 {
		probe := runeBoundaryAbove(text, fitting, min(start+step, upper))
		probeFits, err := fits(probe)
		if err != nil {
			return 0, err
		}
		if !probeFits {
			notFitting = probe
			continue
		}
		if probe == upper {
			return upper, nil
		}
		fitting = probe
		step *= 2
	}

	for {
		middle := runeBoundaryAbove(text, fitting, fitting+(notFitting-fitting)/2)
		if middle >= notFitting {
			return fitting, nil
		}
		middleFits, err := fits(middle)
		if err != nil {
			return 0, err
		}
		if middleFits {
			fitting = middle
		} else {
			notFitting = middle
		}
	}
}

// runeBoundaryAtOrBefore returns the largest UTF-8 boundary of text at or
// before position.
func runeBoundaryAtOrBefore(text string, position int) int {
	for position > 0 && position < len(text) && !utf8.RuneStart(text[position]) {
		position--
	}
	return position
}

// runeBoundaryAbove returns the largest UTF-8 boundary of text at or before
// position when that boundary is above floor. Otherwise it returns the
// boundary after the character that starts at floor. The caller keeps floor
// below len(text).
func runeBoundaryAbove(text string, floor int, position int) int {
	boundary := runeBoundaryAtOrBefore(text, position)
	if boundary > floor {
		return boundary
	}
	_, size := utf8.DecodeRuneInString(text[floor:])
	return floor + size
}
