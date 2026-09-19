package semantic

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
)

// While the maintenance gate is closed a cold collection is refused before any
// LoadCollection request reaches Milvus, and the refusal carries the
// maintenance class. Opening the gate lets the same acquire load as normal.
func TestMaintenanceGateRefusesColdCollectionLoads(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newLoadPathTestService(t, server)
	const collectionName = "hybrid_code_chunks_maintenance"
	server.setCollections(collectionName)
	server.setLoadStates(commonpb.LoadState_LoadStateLoaded, collectionName)

	service.SetMaintenance(true)
	lease, err := service.AcquireCollection(context.Background(), collectionName)
	if lease != nil {
		lease.Release()
	}
	if !errors.Is(err, ErrMaintenance) {
		t.Fatalf("AcquireCollection during maintenance error = %v, want ErrMaintenance", err)
	}
	if calls := server.loadCallCount(); calls != 0 {
		t.Fatalf("LoadCollection calls during maintenance = %d, want 0", calls)
	}

	service.SetMaintenance(false)
	lease, err = service.AcquireCollection(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("AcquireCollection after maintenance returned error: %v", err)
	}
	lease.Release()
	if calls := server.loadCallCount(); calls != 1 {
		t.Fatalf("LoadCollection calls after maintenance = %d, want 1", calls)
	}
}
