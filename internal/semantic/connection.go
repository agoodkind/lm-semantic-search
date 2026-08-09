package semantic

import (
	"context"
	cryptorand "crypto/rand"
	"fmt"
	"log/slog"
	"math/big"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

const (
	reconnectBackoffBase = 2 * time.Second
	reconnectBackoffCap  = 5 * time.Minute
)

var bootDialTimeout = 5 * time.Second

var reconnectSleep = func(ctx context.Context, duration time.Duration) bool {
	timer := time.NewTimer(duration)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

var reconnectJitter = func(limit time.Duration) time.Duration {
	if limit <= 0 {
		return 0
	}

	randomLimit := big.NewInt(int64(limit) + 1)
	randomValue, err := cryptorand.Int(cryptorand.Reader, randomLimit)
	if err != nil {
		return limit
	}
	return time.Duration(randomValue.Int64())
}

// callTimeouts resolves the per-call Milvus deadline policy from configuration.
// Only the mutation bound is operator-tunable, because it is the one bound whose
// sufficient value depends on how many rows a call matches.
//
// The millisecond count is converted by config.MilvusMutationCallTimeout rather
// than multiplied here, so a count too large to hold as a duration falls back to
// the built-in bound instead of wrapping into one that is effectively absent.
func (service *Service) callTimeouts() milvusgrpc.CallTimeouts {
	configuredMutation := config.MilvusMutationCallTimeout(service.cfg.MilvusMutationCallTimeoutMS)
	return milvusgrpc.DefaultCallTimeouts().WithMutation(configuredMutation)
}

func (service *Service) dialMilvus(ctx context.Context) (*milvusclient.Client, error) {
	dialContext, cancel := context.WithTimeout(ctx, bootDialTimeout)
	defer cancel()

	clientConfig := &milvusclient.ClientConfig{
		Address:     service.cfg.MilvusAddress,
		APIKey:      service.cfg.MilvusToken,
		DialOptions: milvusgrpc.DialOptions(slog.Default(), service.callTimeouts()),
	}
	client, err := milvusclient.New(dialContext, clientConfig)
	if err != nil {
		return nil, milvusDialError{address: service.cfg.MilvusAddress, err: err}
	}
	return client, nil
}

type milvusDialError struct {
	address string
	err     error
}

func (err milvusDialError) Error() string {
	return fmt.Sprintf("connect to Milvus at %s: %v", err.address, err.err)
}

func (err milvusDialError) Unwrap() error {
	return err.err
}

func (service *Service) startReconnector(ctx context.Context) {
	reconnectContext, cancel := context.WithCancel(ctx)
	service.reconnectCancel = cancel
	service.reconnectDone = make(chan struct{})

	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(reconnectContext, "Milvus reconnect panic", "err", fmt.Errorf("panic: %v", recovered), "address", service.cfg.MilvusAddress)
			}
		}()
		service.reconnectLoop(reconnectContext)
	}()
}

func (service *Service) reconnectLoop(ctx context.Context) {
	defer close(service.reconnectDone)

	backoffLimit := reconnectBackoffBase
	for attempt := 1; ; attempt++ {
		client, err := service.dialMilvus(ctx)
		if err == nil {
			err = service.publishClient(ctx, client)
			if err == nil {
				service.noteReconnectSucceeded(ctx, attempt)
				return
			}
			if closeErr := client.Close(ctx); closeErr != nil {
				slog.WarnContext(ctx, "close Milvus client after recovery failure", "err", closeErr)
			}
			service.milvus = nil
		}
		if attempt == 1 || attempt%10 == 0 {
			slog.WarnContext(ctx, "Milvus reconnect failed", "address", service.cfg.MilvusAddress, "attempt", attempt, "err", err)
		}

		sleepDuration := reconnectJitter(backoffLimit)
		if !reconnectSleep(ctx, sleepDuration) {
			return
		}
		backoffLimit = nextReconnectBackoff(backoffLimit)
	}
}

func (service *Service) noteReconnectSucceeded(ctx context.Context, attempt int) {
	slog.InfoContext(ctx, "Milvus reconnect succeeded", "address", service.cfg.MilvusAddress, "attempt", attempt)
}

func (service *Service) publishClient(
	ctx context.Context,
	client *milvusclient.Client,
) error {
	service.available.Store(false)
	service.milvus = client
	service.residency.invalidateResidency()
	if err := service.recoverInterruptedPromotions(ctx); err != nil {
		wrappedErr := fmt.Errorf("recover interrupted collection promotion: %w", err)
		slog.WarnContext(ctx, "publish Milvus client failed", "err", wrappedErr)
		return wrappedErr
	}
	service.available.Store(true)
	service.startResidencyReconciliation(ctx, client)
	return nil
}

func (service *Service) startResidencyReconciliation(
	ctx context.Context,
	client *milvusclient.Client,
) {
	reconciliationContext, cancel := context.WithCancel(context.WithoutCancel(ctx))
	generation := service.residency.beginReconciliation()
	service.reconciliationMutex.Lock()
	if service.reconciliationCancel != nil {
		service.reconciliationCancel()
	}
	service.reconciliationCancel = cancel
	service.reconciliationWork.Add(1)
	service.reconciliationMutex.Unlock()
	go func() {
		defer service.reconciliationWork.Done()
		defer func() {
			if recovered := recover(); recovered != nil {
				service.residency.invalidateResidency()
				slog.ErrorContext(
					reconciliationContext,
					"Milvus residency reconciliation panic",
					"err", fmt.Errorf("panic: %v", recovered),
				)
			}
		}()
		service.reconcileResidency(reconciliationContext, client, generation)
	}()
}

func (service *Service) reconcileResidency(
	ctx context.Context,
	client *milvusclient.Client,
	generation uint64,
) {
	collections, err := client.ListCollections(
		ctx,
		milvusclient.NewListCollectionOption(),
	)
	if err != nil {
		service.handleReconciliationError(ctx, err, "list collections")
		return
	}
	for _, collectionName := range collections {
		if isRecoveryCollection(collectionName) {
			continue
		}
		loadState, loadStateErr := client.GetLoadState(
			ctx,
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if loadStateErr != nil {
			service.handleReconciliationError(
				ctx,
				loadStateErr,
				"get load state for "+collectionName,
			)
			return
		}
		state := collectionResidencyUnknown
		switch loadState.State {
		case entity.LoadStateLoaded:
			state = collectionResidencyReady
		case entity.LoadStateLoading:
			state = collectionResidencyLoading
		case entity.LoadStateUnloading:
			state = collectionResidencyUnknown
		case entity.LoadStateNotLoad:
			state = collectionResidencyCold
		}
		service.residency.applyReconciliation(
			ctx,
			generation,
			collectionName,
			state,
		)
	}
}

func (service *Service) handleReconciliationError(
	ctx context.Context,
	err error,
	operation string,
) {
	if storeUnavailable(err) {
		service.residency.invalidateResidency()
	}
	slog.WarnContext(ctx, "Milvus residency reconciliation failed", "operation", operation, "err", err)
}

func (service *Service) stopResidencyReconciliation(ctx context.Context) error {
	service.reconciliationMutex.Lock()
	if service.reconciliationCancel != nil {
		service.reconciliationCancel()
	}
	service.reconciliationMutex.Unlock()
	done := make(chan struct{})
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(
					ctx,
					"wait for Milvus residency reconciliation panicked",
					"error",
					fmt.Errorf("%v", recovered),
				)
			}
			close(done)
		}()
		service.reconciliationWork.Wait()
	}()
	select {
	case <-ctx.Done():
		err := fmt.Errorf(
			"wait for Milvus residency reconciliation shutdown: %w",
			ctx.Err(),
		)
		slog.WarnContext(ctx, "Milvus residency reconciliation shutdown canceled", "err", err)
		return err
	case <-done:
		return nil
	}
}

func nextReconnectBackoff(current time.Duration) time.Duration {
	next := current * 2
	if next > reconnectBackoffCap {
		return reconnectBackoffCap
	}
	return next
}
