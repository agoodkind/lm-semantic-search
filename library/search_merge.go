package library

import (
	"context"
	"database/sql"
)

// rankStatement ranks every candidate occurrence in the query database and
// writes the final ordered result to ranked with ordinals from 0.
//
// The dense rank orders by exact score descending, then occurrence ID
// ascending, over every occurrence. Repeated content keeps one rank per
// occurrence. The lexical rank covers each occurrence with a lexical score for
// its content and orders by that score descending, then occurrence ID
// ascending. With :dense set, the final score is the raw exact COSINE score.
// Otherwise it is the reciprocal rank fusion 1/(:rrf_k + dense rank) plus,
// when present, 1/(:rrf_k + lexical rank); both ranks start at 1.
//
// A positive :min_score keeps final scores at or above it. A positive
// :per_group keeps that many occurrences of each group key over the whole
// ordered result. Final order is score descending, SortKey ascending, then
// occurrence ID ascending.
const rankStatement = `WITH dense AS (
	SELECT c.owner_id, c.row_key, c.sort_key, c.source_blob_id, c.vector_id, c.scalars, c.group_key,
		v.score AS dense_score, l.score AS lexical_score,
		ROW_NUMBER() OVER (ORDER BY v.score DESC, c.owner_id, c.row_key) AS dense_rank
	FROM candidates c
	JOIN query_vectors v ON v.vector_id = c.vector_id
	LEFT JOIN lexical_scores l ON l.search_hash = c.search_hash
), lexical AS (
	SELECT owner_id, row_key,
		ROW_NUMBER() OVER (ORDER BY lexical_score DESC, owner_id, row_key) AS lexical_rank
	FROM dense WHERE lexical_score IS NOT NULL
), fused AS (
	SELECT d.owner_id, d.row_key, d.sort_key, d.source_blob_id, d.vector_id, d.scalars, d.group_key,
		CASE WHEN :dense = 1 THEN d.dense_score
			ELSE 1.0 / (:rrf_k + d.dense_rank) + COALESCE(1.0 / (:rrf_k + x.lexical_rank), 0.0)
		END AS final_score
	FROM dense d LEFT JOIN lexical x ON x.owner_id = d.owner_id AND x.row_key = d.row_key
), floored AS (
	SELECT owner_id, row_key, sort_key, source_blob_id, vector_id, scalars, group_key, final_score
	FROM fused WHERE :min_score <= 0 OR final_score >= :min_score
), grouped AS (
	SELECT owner_id, row_key, sort_key, source_blob_id, vector_id, scalars, final_score,
		ROW_NUMBER() OVER (
			PARTITION BY group_key ORDER BY final_score DESC, sort_key, owner_id, row_key
		) AS group_rank
	FROM floored
)
INSERT INTO ranked (ordinal, owner_id, row_key, source_blob_id, vector_id, scalars, score)
SELECT ROW_NUMBER() OVER (ORDER BY final_score DESC, sort_key, owner_id, row_key) - 1,
	owner_id, row_key, source_blob_id, vector_id, scalars, final_score
FROM grouped WHERE :per_group = 0 OR group_rank <= :per_group`

// rankCandidates writes the final ordered result of the query database into
// its ranked table and returns the row count.
func rankCandidates(ctx context.Context, query *queryDatabase, plan searchPlan) (int64, error) {
	perGroup := 0
	if plan.request.GroupBy != "" {
		perGroup = plan.request.PerGroupLimit
	}
	result, err := query.conn.ExecContext(ctx, rankStatement,
		sql.Named("dense", plan.rank.Mode == Dense),
		sql.Named("rrf_k", plan.rank.RRFK),
		sql.Named("min_score", plan.request.MinScore),
		sql.Named("per_group", perGroup),
	)
	if err != nil {
		return 0, queryDatabaseError(ctx, "rank candidates", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return 0, queryDatabaseError(ctx, "count ranked rows", err)
	}
	return count, nil
}
