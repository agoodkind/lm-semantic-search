package daemon

import (
	"context"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"google.golang.org/grpc/status"
)

// GraphTool calls a cbm graph tool for the codebase that covers the requested path.
func (server *GRPCServer) GraphTool(ctx context.Context, request *pb.GraphToolRequest) (resp *pb.GraphToolResponse, err error) {
	ctx, done := beginRPC(ctx, "GraphTool")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetToolName(), "tool_name", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetArgsJson(), "args_json", false); argErr != nil {
		return nil, argErr
	}

	requestedPath, pathErr := resolveRequestPath(request.GetPath(), request.GetClient().GetCallerCwd())
	if pathErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, adapterr.NewInvalidPath(pathErr.Error(), pathErr)))
	}

	codebase, _, found, _, callErr := server.manager.GetIndex(ctx, requestedPath)
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(requestedPath, callErr)))
	}
	if !found {
		return nil, status.Error(adapterr.Respond(ctx, adapterr.NewNotIndexed(requestedPath, nil)))
	}

	resultJSON, callErr := server.manager.GraphTool(ctx, codebase.ID, request.GetToolName(), request.GetArgsJson())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(requestedPath, callErr)))
	}

	return &pb.GraphToolResponse{
		ResultJson:  resultJSON,
		DisplayText: resultJSON,
	}, nil
}
