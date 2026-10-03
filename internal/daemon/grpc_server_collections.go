package daemon

import (
	"context"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
)

// RegisterCollection registers a document collection with its declared scalar
// columns. A conflict with the saved or stored schema fails with ErrorInfo
// reason collection_schema_mismatch and the conflicting column in the
// metadata. An invalid declaration fails with InvalidArgument and the rejected
// column in the metadata.
func (server *GRPCServer) RegisterCollection(ctx context.Context, request *pb.RegisterCollectionRequest) (resp *pb.RegisterCollectionResponse, err error) {
	ctx, done := beginRPC(ctx, "RegisterCollection")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	codebase, callErr := server.manager.RegisterCollection(ctx, CollectionRegistration{
		CollectionID: request.GetCollectionId(),
		Declaration: model.CollectionDeclaration{
			ItemIDColumn: request.GetItemIdColumn(),
			Scalars:      pbScalarColumns(request.GetScalars()),
		},
	})
	if callErr != nil {
		return nil, adapterr.RespondGRPC(ctx, callErr)
	}
	savedDeclaration := model.CollectionDeclaration{ItemIDColumn: "", Scalars: nil}
	if codebase.Declaration != nil {
		savedDeclaration = *codebase.Declaration
	}
	ack := view.MutationAckView{
		Kind:            view.AckRegisterCollection,
		Path:            "",
		JobID:           "",
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    request.GetCollectionId(),
		CollectionName:  codebase.CollectionName,
		CodebaseID:      codebase.ID,
		ConversationID:  "",
		DocumentCount:   0,
		NeededCount:     0,
		TotalCount:      0,
	}
	health := server.manager.DependencyHealth()
	return &pb.RegisterCollectionResponse{
		CodebaseId:     codebase.ID,
		CollectionName: codebase.CollectionName,
		ItemIdColumn:   savedDeclaration.ItemIDColumn,
		Scalars:        scalarColumnsToPB(savedDeclaration.Scalars),
		DisplayText: server.envelopeText(
			ctx,
			health,
			render.MutationAck(ack),
			"codebase_id",
			codebase.ID,
		),
	}, nil
}

// pbScalarColumns converts wire scalar declarations. An unspecified or unknown
// wire type converts to an empty type, which declaration validation rejects.
func pbScalarColumns(declarations []*pb.ScalarColumnDeclaration) []model.ScalarColumn {
	columns := make([]model.ScalarColumn, 0, len(declarations))
	for _, declaration := range declarations {
		columns = append(columns, model.ScalarColumn{
			Name:      declaration.GetColumn(),
			Type:      scalarTypeFromPB(declaration.GetType()),
			Nullable:  declaration.GetNullable(),
			MaxLength: declaration.GetMaxLength(),
		})
	}
	return columns
}

func scalarTypeFromPB(scalarType pb.ScalarColumnType) model.ScalarType {
	switch scalarType {
	case pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING:
		return model.ScalarTypeString
	case pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL:
		return model.ScalarTypeBool
	case pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64:
		return model.ScalarTypeInt64
	case pb.ScalarColumnType_SCALAR_COLUMN_TYPE_UNSPECIFIED:
		return ""
	default:
		return ""
	}
}

func scalarColumnsToPB(columns []model.ScalarColumn) []*pb.ScalarColumnDeclaration {
	declarations := make([]*pb.ScalarColumnDeclaration, 0, len(columns))
	for _, column := range columns {
		declarations = append(declarations, &pb.ScalarColumnDeclaration{
			Column:    column.Name,
			Type:      scalarTypeToPB(column.Type),
			Nullable:  column.Nullable,
			MaxLength: column.MaxLength,
		})
	}
	return declarations
}

func scalarTypeToPB(scalarType model.ScalarType) pb.ScalarColumnType {
	switch scalarType {
	case model.ScalarTypeString:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING
	case model.ScalarTypeBool:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL
	case model.ScalarTypeInt64:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64
	default:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_UNSPECIFIED
	}
}
