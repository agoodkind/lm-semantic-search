package library

import (
	"context"

	"goodkind.io/lm-semantic-search/library/observation"
)

func (library *Library) acquireWriter(ctx context.Context) (func() error, error) {
	ctx, span := observation.Start(ctx, library.config.Observer, observation.WriterAdmission)
	release, err := library.lock.acquire(ctx)
	span.End(ctx, err, observation.Data{})
	return release, err
}
