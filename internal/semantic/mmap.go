package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	mmapEnabledKey              = "mmap.enabled"
	mmapEnabledValue            = "true"
	mmapPolicyVersion           = 2
	mmapInvertedIndexType       = "INVERTED"
	mmapSparseInvertedIndexType = "SPARSE_INVERTED_INDEX"
	releasePollInterval         = 250 * time.Millisecond
	releasePollTimeout          = 90 * time.Second
	indexVisibilityPollInterval = 250 * time.Millisecond
)

type mmapOutcome int

const (
	mmapOutcomeUnknown mmapOutcome = iota
	mmapOutcomeMigrated
	mmapOutcomeAlready
	mmapOutcomeSkipped
)

type mmapMigrationMode int

const (
	mmapExistingCollection mmapMigrationMode = iota
	mmapCreatedCollection
)

type mmapTargetKind int

const (
	mmapFieldTarget mmapTargetKind = iota
	mmapIndexTarget
)

type mmapTarget struct {
	kind mmapTargetKind
	name string
}

type mmapInspection struct {
	loadState          entity.LoadStateCode
	missingTargets     []mmapTarget
	denseIndexPresent  bool
	contentHashPresent bool
	sparseIndexPresent bool
}

func (service *Service) releaseCollection(ctx context.Context, collectionName string) error {
	if err := service.milvus.ReleaseCollection(
		ctx,
		milvusclient.NewReleaseCollectionOption(collectionName),
	); err != nil {
		return wrapStoreError(ctx, err, "release Milvus collection "+collectionName)
	}
	return nil
}

func (service *Service) awaitCollectionReleased(
	ctx context.Context,
	collectionName string,
) error {
	pollCtx, cancel := context.WithTimeout(ctx, releasePollTimeout)
	defer cancel()
	for {
		state, err := service.milvus.GetLoadState(
			pollCtx,
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if err != nil {
			return wrapStoreError(ctx, err, "poll release state for "+collectionName)
		}
		if state.State == entity.LoadStateNotLoad {
			return nil
		}
		select {
		case <-pollCtx.Done():
			err := fmt.Errorf("await release of %s: %w", collectionName, pollCtx.Err())
			slog.ErrorContext(ctx, "await collection release failed", "err", err)
			return err
		case <-time.After(releasePollInterval):
		}
	}
}

func (service *Service) inspectMmapPolicy(
	ctx context.Context,
	collectionName string,
) (mmapInspection, error) {
	collection, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		return mmapInspection{}, wrapStoreError(
			ctx,
			err,
			"describe collection for mmap "+collectionName,
		)
	}

	missingTargets := make([]mmapTarget, 0, len(collection.Schema.Fields))
	for _, field := range collection.Schema.Fields {
		if !mmapFieldSupported(field) {
			continue
		}
		if field.TypeParams[mmapEnabledKey] != mmapEnabledValue {
			missingTargets = append(missingTargets, mmapTarget{
				kind: mmapFieldTarget,
				name: field.Name,
			})
		}
	}

	inspection := mmapInspection{
		loadState:          entity.LoadStateNotLoad,
		missingTargets:     missingTargets,
		denseIndexPresent:  false,
		contentHashPresent: false,
		sparseIndexPresent: false,
	}
	indexFields := []string{
		denseVectorFieldName,
		contentHashFieldName,
		sparseVectorFieldName,
	}
	for _, fieldName := range indexFields {
		indexNames, err := service.milvus.ListIndexes(
			ctx,
			milvusclient.NewListIndexOption(collectionName).WithFieldName(fieldName),
		)
		if err != nil {
			return mmapInspection{}, wrapStoreError(
				ctx,
				err,
				"list indexes on "+fieldName+" for "+collectionName,
			)
		}
		slices.Sort(indexNames)
		for _, indexName := range indexNames {
			description, describeErr := service.milvus.DescribeIndex(
				ctx,
				milvusclient.NewDescribeIndexOption(collectionName, indexName),
			)
			if describeErr != nil {
				return mmapInspection{}, wrapStoreError(
					ctx,
					describeErr,
					"describe index "+indexName+" on "+collectionName,
				)
			}
			params := description.Params()
			if !noteMmapIndexPresence(&inspection, fieldName, params["index_type"]) {
				continue
			}
			if params[mmapEnabledKey] != mmapEnabledValue {
				inspection.missingTargets = append(inspection.missingTargets, mmapTarget{
					kind: mmapIndexTarget,
					name: indexName,
				})
			}
		}
	}

	loadState, err := service.milvus.GetLoadState(
		ctx,
		milvusclient.NewGetLoadStateOption(collectionName),
	)
	if err != nil {
		return mmapInspection{}, wrapStoreError(ctx, err, "get load state for "+collectionName)
	}
	inspection.loadState = loadState.State
	return inspection, nil
}

func mmapFieldSupported(field *entity.Field) bool {
	if field.PrimaryKey || field.Name == idFieldName || field.Name == sparseVectorFieldName {
		return false
	}
	if field.Name == denseVectorFieldName {
		return true
	}
	return !field.DataType.IsVectorType()
}

func noteMmapIndexPresence(
	inspection *mmapInspection,
	fieldName string,
	indexType string,
) bool {
	switch {
	case fieldName == denseVectorFieldName:
		inspection.denseIndexPresent = true
		return true
	case fieldName == contentHashFieldName && indexType == mmapInvertedIndexType:
		inspection.contentHashPresent = true
		return true
	case fieldName == sparseVectorFieldName && indexType == mmapSparseInvertedIndexType:
		inspection.sparseIndexPresent = true
		return true
	default:
		return false
	}
}

func (service *Service) mmapInspectionComplete(
	inspection mmapInspection,
	mode mmapMigrationMode,
) bool {
	if len(inspection.missingTargets) != 0 || !inspection.denseIndexPresent {
		return false
	}
	if mode != mmapCreatedCollection {
		return true
	}
	if !inspection.contentHashPresent {
		return false
	}
	return !service.cfg.HybridMode || inspection.sparseIndexPresent
}

func mmapLoadStateStable(state entity.LoadStateCode) bool {
	return state == entity.LoadStateLoaded || state == entity.LoadStateNotLoad
}

func (service *Service) waitForCreatedMmapTargets(
	ctx context.Context,
	collectionName string,
) (mmapInspection, error) {
	pollCtx, cancel := context.WithTimeout(ctx, service.callTimeouts().Metadata)
	defer cancel()
	for {
		inspection, err := service.inspectMmapPolicy(pollCtx, collectionName)
		if err != nil {
			return mmapInspection{}, err
		}
		if inspection.denseIndexPresent && inspection.contentHashPresent &&
			(!service.cfg.HybridMode || inspection.sparseIndexPresent) {
			return inspection, nil
		}
		select {
		case <-pollCtx.Done():
			err := fmt.Errorf(
				"wait for required mmap indexes on %s: %w",
				collectionName,
				pollCtx.Err(),
			)
			slog.ErrorContext(ctx, "wait for required mmap indexes failed", "err", err)
			return mmapInspection{}, err
		case <-time.After(indexVisibilityPollInterval):
		}
	}
}

func (service *Service) alterMmapTargets(
	ctx context.Context,
	collectionName string,
	targets []mmapTarget,
) error {
	for _, target := range targets {
		switch target.kind {
		case mmapFieldTarget:
			if err := service.milvus.AlterCollectionFieldProperty(
				ctx,
				milvusclient.NewAlterCollectionFieldPropertiesOption(
					collectionName,
					target.name,
				).WithProperty(mmapEnabledKey, mmapEnabledValue),
			); err != nil {
				return wrapStoreError(ctx, err, "enable mmap on field "+target.name+" of "+collectionName)
			}
		case mmapIndexTarget:
			if err := service.milvus.AlterIndexProperties(
				ctx,
				milvusclient.NewAlterIndexPropertiesOption(
					collectionName,
					target.name,
				).WithProperty(mmapEnabledKey, mmapEnabledValue),
			); err != nil {
				return wrapStoreError(ctx, err, "enable mmap on index "+target.name+" of "+collectionName)
			}
		}
	}
	return nil
}

func (service *Service) restoreMmapReadyState(
	ctx context.Context,
	collectionName string,
	operationErr error,
) error {
	restoreCtx, cancel := context.WithTimeout(
		context.WithoutCancel(ctx),
		service.sharedCollectionLoadCeiling(),
	)
	defer cancel()
	restoreErr := service.loadCollection(restoreCtx, collectionName)
	if restoreErr == nil {
		return operationErr
	}
	joinedErr := errors.Join(
		operationErr,
		fmt.Errorf("restore ready collection %s after mmap failure: %w", collectionName, restoreErr),
	)
	slog.ErrorContext(ctx, "restore ready collection after mmap failure failed", "err", joinedErr)
	return joinedErr
}

func (service *Service) ensureMmapEnabledOnCollection(
	ctx context.Context,
	collectionName string,
	mode mmapMigrationMode,
) (mmapOutcome, error) {
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check collection "+collectionName+" for mmap",
	)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	if !hasCollection {
		return mmapOutcomeSkipped, nil
	}

	observedState, observation, err := service.residency.Observe(ctx, collectionName)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	inspection, err := service.inspectMmapPolicy(ctx, collectionName)
	observation.ReleaseContext(ctx)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	if observedState == collectionResidencyLoading ||
		!mmapLoadStateStable(inspection.loadState) {
		return mmapOutcomeSkipped, nil
	}
	if !inspection.denseIndexPresent && mode == mmapExistingCollection {
		return mmapOutcomeSkipped, nil
	}
	if service.mmapInspectionComplete(inspection, mode) {
		return mmapOutcomeAlready, nil
	}

	maintenance, err := service.residency.Maintain(ctx, collectionName)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	defer maintenance.ReleaseContext(ctx)
	return service.migrateMmapUnderMaintenance(ctx, collectionName, mode)
}

func (service *Service) migrateMmapUnderMaintenance(
	ctx context.Context,
	collectionName string,
	mode mmapMigrationMode,
) (mmapOutcome, error) {
	var inspection mmapInspection
	var err error

	if mode == mmapCreatedCollection {
		inspection, err = service.waitForCreatedMmapTargets(ctx, collectionName)
	} else {
		inspection, err = service.inspectMmapPolicy(ctx, collectionName)
	}
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	if !inspection.denseIndexPresent && mode == mmapExistingCollection {
		return mmapOutcomeSkipped, nil
	}
	if service.mmapInspectionComplete(inspection, mode) {
		return mmapOutcomeAlready, nil
	}
	if !mmapLoadStateStable(inspection.loadState) {
		return mmapOutcomeSkipped, nil
	}

	priorReady := inspection.loadState == entity.LoadStateLoaded
	releaseStarted := false
	if priorReady {
		releaseStarted = true
		if err := service.releaseCollection(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, service.restoreMmapReadyState(ctx, collectionName, err)
		}
		if err := service.awaitCollectionReleased(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, service.restoreMmapReadyState(ctx, collectionName, err)
		}
	}

	if err := service.alterMmapTargets(ctx, collectionName, inspection.missingTargets); err != nil {
		if releaseStarted {
			err = service.restoreMmapReadyState(ctx, collectionName, err)
		}
		return mmapOutcomeUnknown, err
	}
	rechecked, err := service.inspectMmapPolicy(ctx, collectionName)
	if err == nil && !service.mmapInspectionComplete(rechecked, mode) {
		err = fmt.Errorf("mmap policy %d incomplete after alteration on %s", mmapPolicyVersion, collectionName)
	}
	if err != nil {
		if releaseStarted {
			err = service.restoreMmapReadyState(ctx, collectionName, err)
		}
		return mmapOutcomeUnknown, err
	}
	if priorReady {
		if err := service.loadCollection(ctx, collectionName); err != nil {
			return mmapOutcomeUnknown, service.restoreMmapReadyState(ctx, collectionName, err)
		}
	}
	slog.InfoContext(
		ctx,
		"semantic.mmap_enabled",
		"collection", collectionName,
		"policy_version", mmapPolicyVersion,
		"was_loaded", priorReady,
	)
	return mmapOutcomeMigrated, nil
}

func (service *Service) ensureMmapEnabledOnce(
	ctx context.Context,
	collectionName string,
	mode mmapMigrationMode,
) (mmapOutcome, error) {
	if version, done := service.ensuredMmapEnabled.Load(collectionName); done &&
		version == mmapPolicyVersion {
		return mmapOutcomeAlready, nil
	}
	outcome, err := service.ensureMmapEnabledOnCollection(ctx, collectionName, mode)
	if err != nil {
		return mmapOutcomeUnknown, err
	}
	if outcome == mmapOutcomeMigrated || outcome == mmapOutcomeAlready {
		service.ensuredMmapEnabled.Store(collectionName, mmapPolicyVersion)
	}
	return outcome, nil
}

// EnsureMmapEnabledAllCollections applies mmap policy version 2 to durable
// collections and leaves staging collections to create-time migration.
func (service *Service) EnsureMmapEnabledAllCollections(ctx context.Context) {
	if !service.Available() {
		return
	}
	collections, err := service.ListCollections(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "semantic.mmap_sweep_list_failed", "err", err)
		return
	}

	migrated := 0
	already := 0
	skipped := 0
	failed := 0
	for _, collectionName := range collections {
		if isStagingCollection(collectionName) {
			skipped++
			continue
		}
		outcome, ensureErr := service.ensureMmapEnabledOnce(
			ctx,
			collectionName,
			mmapExistingCollection,
		)
		if ensureErr != nil {
			failed++
			continue
		}
		switch outcome {
		case mmapOutcomeMigrated:
			migrated++
		case mmapOutcomeAlready:
			already++
		case mmapOutcomeSkipped:
			skipped++
		case mmapOutcomeUnknown:
		}
	}

	attrs := []slog.Attr{
		slog.Int("total", len(collections)),
		slog.Int("migrated", migrated),
		slog.Int("already_mmapped", already),
		slog.Int("skipped_no_index", skipped),
		slog.Int("failed", failed),
	}
	level := slog.LevelDebug
	message := "semantic.mmap_sweep_complete"
	switch {
	case failed > 0:
		level = slog.LevelWarn
		message = "semantic.mmap_sweep_complete_with_failures"
	case migrated > 0:
		level = slog.LevelInfo
	}
	slog.LogAttrs(ctx, level, message, attrs...)
}
