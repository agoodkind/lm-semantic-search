package daemon

import (
	"context"
	"net/http"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost/platform"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/platformactivity"
)

// ModelDownloadOptions uses production defaults for nil or zero fields.
type ModelDownloadOptions struct {
	NetworkSource   platform.Source
	HTTPClient      *http.Client
	Presets         *offlinemodel.Registry
	RecheckInterval time.Duration
	RetryInterval   time.Duration
	// AttemptBackoff replaces delays between attempts within one download when
	// positive. Zero selects delays that start at one second and double.
	AttemptBackoff time.Duration
}

// ManagerOptions selects production defaults for nil or zero download fields.
type ManagerOptions struct {
	ModelDownload ModelDownloadOptions
}

// DefaultManagerOptions sets the platform network source, [http.DefaultClient],
// and the pinned registry. The deferred recheck interval is five minutes.
// The failed download retry interval is one minute.
func DefaultManagerOptions() ManagerOptions {
	return ManagerOptions{
		ModelDownload: ModelDownloadOptions{
			NetworkSource:   platform.New(),
			HTTPClient:      http.DefaultClient,
			Presets:         offlinemodel.PinnedRegistry(),
			RecheckInterval: modelDownloadRecheckInterval,
			RetryInterval:   modelDownloadRetryInterval,
			AttemptBackoff:  0,
		},
	}
}

func (options ModelDownloadOptions) withDefaults() ModelDownloadOptions {
	defaults := DefaultManagerOptions().ModelDownload
	if options.NetworkSource == nil {
		options.NetworkSource = defaults.NetworkSource
	}
	if options.HTTPClient == nil {
		options.HTTPClient = defaults.HTTPClient
	}
	if options.Presets == nil {
		options.Presets = defaults.Presets
	}
	if options.RecheckInterval <= 0 {
		options.RecheckInterval = defaults.RecheckInterval
	}
	if options.RetryInterval <= 0 {
		options.RetryInterval = defaults.RetryInterval
	}
	return options
}

// NewManager loads persisted daemon state from disk.
func NewManager(ctx context.Context, cfg config.Config) (*Manager, error) {
	return NewManagerWithOptions(ctx, cfg, DefaultManagerOptions())
}

// NewManagerWithOptions returns before the background model download
// completes when the local backend uses the ONNX provider.
func NewManagerWithOptions(
	ctx context.Context,
	cfg config.Config,
	options ManagerOptions,
) (*Manager, error) {
	return newManagerWithDependencies(ctx, cfg, managerDependencies{
		semanticFactory: nil,
		activitySource:  platformactivity.New(ctx),
		modelDownload:   options.ModelDownload,
	})
}
