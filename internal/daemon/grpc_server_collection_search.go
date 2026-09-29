package daemon

import (
	"context"
	"fmt"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/protobuf/types/known/structpb"
)

// SearchCollection searches a registered document collection with a typed
// filter tree. An invalid filter fails with InvalidArgument and, for a column
// violation, the rejected column in the ErrorInfo metadata. Maintenance mode
// fails the search before any collection load or query.
func (server *GRPCServer) SearchCollection(ctx context.Context, request *pb.SearchCollectionRequest) (resp *pb.SearchCollectionResponse, err error) {
	ctx, done := beginRPC(ctx, "SearchCollection")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetQuery(), "query", false); argErr != nil {
		return nil, argErr
	}
	filter, filterErr := pbCollectionFilterTree(request.GetFilter())
	if filterErr != nil {
		return nil, adapterr.RespondGRPC(ctx, filterErr)
	}
	searchResult, callErr := server.manager.SearchCollection(ctx, CollectionSearchRequest{
		CollectionID:  request.GetCollectionId(),
		Query:         request.GetQuery(),
		Limit:         request.GetLimit(),
		MinScore:      request.GetMinScore(),
		Filter:        filter,
		GroupBy:       request.GetGroupBy(),
		PerGroupLimit: request.GetPerGroupLimit(),
		Offset:        request.GetOffset(),
		RankingToken:  request.GetRankingToken(),
		CallerState:   "",
	})
	if callErr != nil {
		return nil, adapterr.RespondGRPC(ctx, callErr)
	}
	health := server.manager.DependencyHealth()
	results := make([]view.CollectionResultView, 0, len(searchResult.Hits))
	pbHits := make([]*pb.CollectionSearchHit, 0, len(searchResult.Hits))
	for _, hit := range searchResult.Hits {
		results = append(results, view.CollectionResultView{
			RowKey:  hit.Chunk.RelativePath,
			Score:   hit.Chunk.Score,
			Content: hit.Chunk.Content,
		})
		pbHits = append(pbHits, &pb.CollectionSearchHit{
			RowKey:  hit.Chunk.RelativePath,
			Content: hit.Chunk.Content,
			Score:   hit.Chunk.Score,
			Scalars: collectionHitScalarsToPB(hit.Scalars),
		})
	}
	searchView := view.CollectionSearchView{
		CollectionID: request.GetCollectionId(),
		Query:        request.GetQuery(),
		Results:      results,
	}
	return &pb.SearchCollectionResponse{
		Hits:             pbHits,
		DependencyHealth: toDependencyHealth(health),
		DisplayText:      server.envelopeText(ctx, health, render.CollectionSearch(searchView)),
		RankingTruncated: searchResult.RankingTruncated,
		RankingToken:     searchResult.RankingToken,
	}, nil
}

// GetCollectionItemState returns the content fingerprint the collection's
// Merkle checkpoint records for one item. An unknown item or an unregistered
// collection returns an empty fingerprint.
func (server *GRPCServer) GetCollectionItemState(ctx context.Context, request *pb.GetCollectionItemStateRequest) (resp *pb.GetCollectionItemStateResponse, err error) {
	ctx, done := beginRPC(ctx, "GetCollectionItemState")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetItemId(), "item_id", false); argErr != nil {
		return nil, argErr
	}
	fingerprint, callErr := server.manager.CollectionItemState(ctx, request.GetCollectionId(), request.GetItemId())
	if callErr != nil {
		return nil, adapterr.RespondGRPC(ctx, callErr)
	}
	health := server.manager.DependencyHealth()
	stateView := view.CollectionItemStateView{
		CollectionID: request.GetCollectionId(),
		ItemID:       request.GetItemId(),
		Fingerprint:  fingerprint,
	}
	return &pb.GetCollectionItemStateResponse{
		IndexedFingerprint: fingerprint,
		DisplayText:        server.envelopeText(ctx, health, render.CollectionItemState(stateView)),
	}, nil
}

// pbCollectionFilterTree converts the wire filter tree. A nil filter converts
// to nil, which matches every row. A node with no member set, a literal with no
// value set, and a negate node without a child fail with InvalidArgument.
// Column and type checks against the declaration run later in the manager.
func pbCollectionFilterTree(filter *pb.CollectionFilter) (*semantic.CollectionFilter, error) {
	if filter == nil {
		return nil, nil
	}
	converted, err := pbCollectionFilterNode(filter, 1)
	if err != nil {
		return nil, err
	}
	return &converted, nil
}

func pbCollectionFilterNode(filter *pb.CollectionFilter, depth int) (semantic.CollectionFilter, error) {
	var rejected semantic.CollectionFilter
	if depth > maxCollectionFilterDepth {
		return rejected, adapterr.NewInvalidArgument(fmt.Sprintf("filter tree is deeper than %d levels", maxCollectionFilterDepth))
	}
	switch node := filter.GetNode().(type) {
	case *pb.CollectionFilter_AllOf:
		children, err := pbCollectionFilterChildren(node.AllOf.GetFilters(), depth)
		if err != nil {
			return rejected, err
		}
		return semantic.AllOf(children...), nil
	case *pb.CollectionFilter_AnyOf:
		children, err := pbCollectionFilterChildren(node.AnyOf.GetFilters(), depth)
		if err != nil {
			return rejected, err
		}
		return semantic.AnyOf(children...), nil
	case *pb.CollectionFilter_Negate:
		if node.Negate == nil {
			return rejected, adapterr.NewInvalidArgument("filter negate node needs exactly one child")
		}
		child, err := pbCollectionFilterNode(node.Negate, depth+1)
		if err != nil {
			return rejected, err
		}
		return semantic.Negate(child), nil
	case *pb.CollectionFilter_Equals:
		value, err := pbCollectionFilterValue(node.Equals.GetColumn(), node.Equals.GetValue())
		if err != nil {
			return rejected, err
		}
		return semantic.ColumnEquals(node.Equals.GetColumn(), value), nil
	case *pb.CollectionFilter_InSet:
		values := make([]semantic.ScalarValue, 0, len(node.InSet.GetValues()))
		for _, wireValue := range node.InSet.GetValues() {
			value, err := pbCollectionFilterValue(node.InSet.GetColumn(), wireValue)
			if err != nil {
				return rejected, err
			}
			values = append(values, value)
		}
		return semantic.ColumnIn(node.InSet.GetColumn(), values), nil
	case *pb.CollectionFilter_Range:
		return semantic.ColumnRange(node.Range.GetColumn(), optionalInt64(node.Range.Lower), optionalInt64(node.Range.Upper)), nil
	case *pb.CollectionFilter_IsNull:
		return semantic.ColumnIsNull(node.IsNull.GetColumn()), nil
	case *pb.CollectionFilter_IsPresent:
		return semantic.ColumnIsPresent(node.IsPresent.GetColumn()), nil
	default:
		return rejected, adapterr.NewInvalidArgument("filter node sets no member")
	}
}

func pbCollectionFilterChildren(filters []*pb.CollectionFilter, depth int) ([]semantic.CollectionFilter, error) {
	children := make([]semantic.CollectionFilter, 0, len(filters))
	for _, child := range filters {
		converted, err := pbCollectionFilterNode(child, depth+1)
		if err != nil {
			return nil, err
		}
		children = append(children, converted)
	}
	return children, nil
}

func pbCollectionFilterValue(column string, value *pb.CollectionFilterValue) (semantic.ScalarValue, error) {
	switch typed := value.GetValue().(type) {
	case *pb.CollectionFilterValue_StringValue:
		return semantic.StringScalar(typed.StringValue), nil
	case *pb.CollectionFilterValue_BoolValue:
		return semantic.BoolScalar(typed.BoolValue), nil
	case *pb.CollectionFilterValue_Int64Value:
		return semantic.Int64Scalar(typed.Int64Value), nil
	default:
		var rejected semantic.ScalarValue
		return rejected, adapterr.NewInvalidFilterColumn(column, fmt.Sprintf("filter value for column %q sets no value", column))
	}
}

// optionalInt64 copies a proto3 optional int64. The filter then does not alias
// the wire message's memory. A nil input stays nil, which is an open bound.
func optionalInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

// collectionHitScalarsToPB converts hit cells to the wire shape. An absent
// cell sets no value, and a null cell sets null_value.
func collectionHitScalarsToPB(cells []semantic.ScalarCell) []*pb.CollectionHitScalar {
	scalars := make([]*pb.CollectionHitScalar, 0, len(cells))
	for _, cell := range cells {
		scalar := &pb.CollectionHitScalar{Column: cell.Column, Value: nil}
		switch cell.State {
		case semantic.ScalarCellNull:
			scalar.Value = &pb.CollectionHitScalar_NullValue{NullValue: structpb.NullValue_NULL_VALUE}
		case semantic.ScalarCellValue:
			setHitScalarValue(scalar, cell.Value)
		case semantic.ScalarCellAbsent:
		}
		scalars = append(scalars, scalar)
	}
	return scalars
}

func setHitScalarValue(scalar *pb.CollectionHitScalar, value semantic.ScalarValue) {
	switch value.Type {
	case model.ScalarTypeBool:
		scalar.Value = &pb.CollectionHitScalar_BoolValue{BoolValue: value.Bool}
	case model.ScalarTypeInt64:
		scalar.Value = &pb.CollectionHitScalar_Int64Value{Int64Value: value.Int64}
	case model.ScalarTypeString:
		scalar.Value = &pb.CollectionHitScalar_StringValue{StringValue: value.String}
	default:
		scalar.Value = &pb.CollectionHitScalar_StringValue{StringValue: value.String}
	}
}
