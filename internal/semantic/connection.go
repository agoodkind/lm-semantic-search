package semantic

import (
	"context"
	cryptorand "crypto/rand"
	"fmt"
	"log/slog"
	"math/big"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
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

func (service *Service) dialMilvus(ctx context.Context) (*milvusclient.Client, error) {
	dialContext, cancel := context.WithTimeout(ctx, bootDialTimeout)
	defer cancel()

	clientConfig := &milvusclient.ClientConfig{
		Address: service.cfg.MilvusAddress,
		APIKey:  service.cfg.MilvusToken,
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
			service.publishClient(client)
			service.noteReconnectSucceeded(ctx, attempt)
			return
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

func (service *Service) publishClient(client *milvusclient.Client) {
	service.milvus = client
	service.available.Store(true)
}

func nextReconnectBackoff(current time.Duration) time.Duration {
	next := current * 2
	if next > reconnectBackoffCap {
		return reconnectBackoffCap
	}
	return next
}
