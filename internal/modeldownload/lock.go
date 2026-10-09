package modeldownload

import (
	"context"
	"errors"
	"os"
	"syscall"
	"time"
)

const (
	lockFileSuffix   = ".lock"
	lockFileMode     = 0o600
	lockPollInterval = 200 * time.Millisecond
)

func acquireLock(ctx context.Context, request Request) (*os.File, error) {
	lockPath := request.DestinationPath + lockFileSuffix
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, lockFileMode)
	if err != nil {
		return nil, fileError(ctx, "open artifact lock", lockPath, err)
	}
	for {
		lockErr := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if lockErr == nil {
			return lockFile, nil
		}
		if !errors.Is(lockErr, syscall.EWOULDBLOCK) {
			_ = lockFile.Close()
			return nil, fileError(ctx, "acquire artifact lock", lockPath, lockErr)
		}
		sleepContext(ctx, lockPollInterval)
		if cancelErr := ctx.Err(); cancelErr != nil {
			_ = lockFile.Close()
			return nil, unavailable(ctx, request.URL, cancelErr)
		}
	}
}
