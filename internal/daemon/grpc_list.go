package daemon

import (
	"context"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/pbconv"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
)

// ListIndexes returns all tracked codebases.
func (server *GRPCServer) ListIndexes(ctx context.Context, request *pb.ListIndexesRequest) (resp *pb.ListIndexesResponse, err error) {
	ctx, done := beginRPC(ctx, "ListIndexes")
	defer done(&err)
	_ = request
	views := server.manager.ListIndexesView()
	response := &pb.ListIndexesResponse{
		Indexes: make([]*pb.Codebase, 0, len(views)),
	}
	rows := make([]view.CodebaseRowView, 0, len(views))
	for _, codebaseView := range views {
		pbCodebase := pbconv.ToCodebase(codebaseView.Codebase)
		applyDisplayTokens(pbCodebase, codebaseView.Display)
		reuseSiblingCount := int32(0)
		if codebaseView.Display == displayDiscovered {
			reuseSiblingCount = server.manager.worktreeReuseForecast(codebaseView.Codebase)
		}
		applyReuseForecast(pbCodebase, reuseSiblingCount)

		// An actively-indexing codebase carries its live breakdown so the list
		// row, the TUI, and get_indexing_status all render the same tree from one
		// resolver. The breakdown rides on active_progress for the TUI and is
		// resolved here for the text rows.
		active := false
		breakdown := view.OutcomeBreakdown{ScopeLabel: "", Processed: 0, ScopeTotal: 0, FileRows: nil, ChunksTotal: 0, ChunkRows: nil}
		if jobID := codebaseView.Codebase.ActiveJobID; jobID != "" {
			if activeJob, ok := server.manager.GetJob(jobID); ok {
				pbCodebase.ActiveProgress = pbconv.ToProgress(activeJob.Progress)
				breakdown = resolveOutcomeBreakdown(activeJob.Progress)
				active = len(breakdown.FileRows) > 0 || len(breakdown.ChunkRows) > 0
			}
		}

		response.Indexes = append(response.Indexes, pbCodebase)
		rows = append(rows, view.CodebaseRowView{
			ID:                codebaseView.Codebase.ID,
			CanonicalPath:     codebaseView.Codebase.CanonicalPath,
			Display:           view.Display(codebaseView.Display),
			ReuseSiblingCount: reuseSiblingCount,
			Active:            active,
			Breakdown:         breakdown,
		})
	}
	health := server.manager.DependencyHealth()
	response.DependencyHealth = toDependencyHealth(health)
	response.DisplayText = server.envelopeText(ctx, health, render.ListIndexes(rows))
	return response, nil
}
