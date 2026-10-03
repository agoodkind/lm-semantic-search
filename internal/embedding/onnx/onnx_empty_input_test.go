package embedding

import (
	"context"
	"errors"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

// asAdapterError narrows err to the typed adapter error the providers return for
// a refused input.
func asAdapterError(err error, target **adapterr.AdapterError) bool {
	return errors.As(err, target)
}

// TestONNXEmbedBatchRefusesEmptyContent proves the in-process provider honors the
// same refusal as the hosted one. Both implement one Provider contract, so a
// guarantee that held only for the hosted endpoint would leave the offline
// backend accumulating exactly the rows the guard exists to prevent.
//
// The provider is deliberately unloaded: no session and no tokenizer. Reaching
// either would panic, so passing also proves the refusal happens before any
// tokenization, which is what keeps an empty input off the shared runtime lock.
func TestONNXEmbedBatchRefusesEmptyContent(t *testing.T) {
	provider := newUnloadedONNXProvider(t, offlinemodel.BGESmall)

	result, err := provider.EmbedBatch(context.Background(), []string{"", "   \n\t "})
	if err != nil {
		t.Fatalf("EmbedBatch returned error: %v", err)
	}
	if len(result.Vectors) != 2 {
		t.Fatalf("vectors = %d, want 2 index-aligned with the inputs", len(result.Vectors))
	}
	for index, vector := range result.Vectors {
		if vector != nil {
			t.Fatalf("Vectors[%d] = %#v, want nil; content with nothing in it got a vector", index, vector)
		}
	}
	if len(result.Skipped) != 2 {
		t.Fatalf("skipped = %d, want 2; a refusal must be reported, never silent", len(result.Skipped))
	}
	for _, skip := range result.Skipped {
		if skip.Reason != adapterr.EmbedRejectionEmptyContent {
			t.Fatalf("skipped reason = %q, want %q", skip.Reason, adapterr.EmbedRejectionEmptyContent)
		}
		// Nothing was measured and no ceiling was reached, so reporting the model's
		// token window here would name a limit that had nothing to do with this.
		if skip.MaxTokens.Reported || skip.ReportedTokens.Reported {
			t.Fatalf("skipped carries token figures %+v/%+v, want none", skip.ReportedTokens, skip.MaxTokens)
		}
	}
}

// TestONNXEmbedRefusesEmptyContentAsAnError covers the single-input entry point,
// which returns an error rather than a skip. The message must name the actual
// condition instead of pointing at a size limit the caller cannot act on.
func TestONNXEmbedRefusesEmptyContentAsAnError(t *testing.T) {
	provider := newUnloadedONNXProvider(t, offlinemodel.BGESmall)

	vector, err := provider.Embed(context.Background(), "   ")
	if err == nil {
		t.Fatal("Embed returned a vector for whitespace-only input")
	}
	if vector != nil {
		t.Fatalf("Embed returned %d values alongside the rejection", len(vector))
	}
	var rejection *adapterr.AdapterError
	if !asAdapterError(err, &rejection) {
		t.Fatalf("error is not a typed adapter error: %v", err)
	}
	if rejection.Code != string(adapterr.EmbedRejectionEmptyContent) {
		t.Fatalf("code = %q, want %q", rejection.Code, adapterr.EmbedRejectionEmptyContent)
	}
	if rejection.Hint != "" {
		t.Fatalf("hint = %q, want none; no shorter input would help an already empty one", rejection.Hint)
	}
}
