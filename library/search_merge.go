package library

import (
	"context"
	"database/sql"
	"errors"
)

// Rank statements. Each ordering table is keyed in rank order, and each rank
// table assigns rowids in the key order of its ordering table. No statement
// sorts outside the query database b-trees.
//
// The dense order is exact score descending, then occurrence ID ascending,
// over every candidate occurrence. Repeated content keeps one rank per
// occurrence. The lexical order covers each candidate with a lexical score for
// its content, by that score descending, then occurrence ID ascending. Both
// ranks start at 1.
//
// With :dense set, the final score is the raw exact COSINE score. Otherwise it
// is the reciprocal rank fusion 1/(:rrf_k + dense rank) plus, when present,
// 1/(:rrf_k + lexical rank). The final order is score descending, SortKey
// ascending, then occurrence ID ascending.
const (
	denseOrderStatement = `INSERT INTO dense_order (negated_score, owner_id, row_key)
		SELECT -v.score, c.owner_id, c.row_key FROM candidates c JOIN query_vectors v ON v.vector_id = c.vector_id`
	denseRankStatement = `INSERT INTO dense_ranks (owner_id, row_key)
		SELECT owner_id, row_key FROM dense_order ORDER BY negated_score, owner_id, row_key`
	lexicalOrderStatement = `INSERT INTO lexical_order (negated_score, owner_id, row_key)
		SELECT -l.score, c.owner_id, c.row_key FROM candidates c JOIN lexical_scores l ON l.search_hash = c.search_hash`
	lexicalRankStatement = `INSERT INTO lexical_ranks (owner_id, row_key)
		SELECT owner_id, row_key FROM lexical_order ORDER BY negated_score, owner_id, row_key`
	finalOrderStatement = `INSERT INTO final_order
		(negated_score, sort_key, owner_id, row_key, source_blob_id, vector_id, scalars, group_key)
		SELECT CASE WHEN :dense = 1 THEN -v.score
				ELSE -(1.0 / (:rrf_k + d.rank) + COALESCE(1.0 / (:rrf_k + x.rank), 0.0)) END,
			c.sort_key, c.owner_id, c.row_key, c.source_blob_id, c.vector_id, c.scalars, c.group_key
		FROM candidates c
		JOIN query_vectors v ON v.vector_id = c.vector_id
		JOIN dense_ranks d ON d.owner_id = c.owner_id AND d.row_key = c.row_key
		LEFT JOIN lexical_ranks x ON x.owner_id = c.owner_id AND x.row_key = c.row_key`
	finalRowsStatement = `SELECT negated_score, owner_id, row_key, source_blob_id, vector_id, scalars, group_key
		FROM final_order ORDER BY negated_score, sort_key, owner_id, row_key`
	groupCountStatement     = `SELECT count FROM group_counts WHERE group_key = :group_key`
	incrementGroupStatement = `INSERT INTO group_counts (group_key, count) VALUES (:group_key, 1)
		ON CONFLICT (group_key) DO UPDATE SET count = count + 1`
)

// rankCandidates writes the final ordered result of the query database into
// its ranked table and returns the row count. It reads the final order once,
// stops at the first score below a positive MinScore, keeps at most
// PerGroupLimit occurrences of each group key when GroupBy is set, and assigns
// ordinals from 0. The group counts are rows of the query database, which
// MaxTemporaryBytes limits.
func rankCandidates(ctx context.Context, query *queryDatabase, plan searchPlan) (count int64, err error) {
	writer, err := query.conn.BeginTx(ctx, nil)
	if err != nil {
		return 0, queryDatabaseError(ctx, "begin ranking", err)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, writer.Rollback())
		}
	}()
	orderStatements := []string{denseOrderStatement, denseRankStatement, lexicalOrderStatement, lexicalRankStatement}
	for _, statement := range orderStatements {
		if _, err := writer.ExecContext(ctx, statement); err != nil {
			return 0, queryDatabaseError(ctx, "order candidates", err)
		}
	}
	if _, err := writer.ExecContext(ctx, finalOrderStatement,
		sql.Named("dense", plan.rank.Mode == Dense),
		sql.Named("rrf_k", plan.rank.RRFK),
	); err != nil {
		return 0, queryDatabaseError(ctx, "order final scores", err)
	}
	count, err = writeRanked(ctx, writer, plan.request)
	if err != nil {
		return 0, err
	}
	if err := writer.Commit(); err != nil {
		return 0, queryDatabaseError(ctx, "commit ranking", err)
	}
	return count, nil
}

// writeRanked reads final_order in key order and inserts the rows that pass
// the MinScore floor and the group quota into ranked.
func writeRanked(ctx context.Context, writer *sql.Tx, request SearchRequest) (count int64, err error) {
	rows, err := writer.QueryContext(ctx, finalRowsStatement)
	if err != nil {
		return 0, queryDatabaseError(ctx, "read final order", err)
	}
	defer func() {
		err = errors.Join(err, closeRows(ctx, rows))
	}()
	insert := rankedInsert{}
	defer func() {
		err = errors.Join(err, insert.close(ctx))
	}()
	for rows.Next() {
		var negatedScore float64
		var row rankedRow
		var groupKey string
		if err := rows.Scan(&negatedScore, &row.ownerID, &row.rowKey, &row.sourceBlobID, &row.vectorID, &row.scalars, &groupKey); err != nil {
			return 0, queryDatabaseError(ctx, "scan final order", err)
		}
		row.score = -negatedScore
		if request.MinScore > 0 && row.score < request.MinScore {
			break
		}
		if request.GroupBy != "" {
			admitted, err := admitGroupMember(ctx, writer, groupKey, request.PerGroupLimit)
			if err != nil {
				return 0, err
			}
			if !admitted {
				continue
			}
		}
		row.ordinal = count
		if err := insert.append(ctx, writer, row); err != nil {
			return 0, queryDatabaseError(ctx, "save ranked row", err)
		}
		count++
	}
	if err := rows.Err(); err != nil {
		return 0, queryDatabaseError(ctx, "read final order", err)
	}
	if err := insert.flush(ctx, writer); err != nil {
		return 0, queryDatabaseError(ctx, "save ranked rows", err)
	}
	return count, nil
}

// admitGroupMember reports whether the group of groupKey has fewer than limit
// members, and counts one more member when it does.
func admitGroupMember(ctx context.Context, writer *sql.Tx, groupKey string, limit int) (bool, error) {
	var members int
	err := writer.QueryRowContext(ctx, groupCountStatement, sql.Named("group_key", groupKey)).Scan(&members)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return false, queryDatabaseError(ctx, "read group count", err)
	}
	if members >= limit {
		return false, nil
	}
	if _, err := writer.ExecContext(ctx, incrementGroupStatement, sql.Named("group_key", groupKey)); err != nil {
		return false, queryDatabaseError(ctx, "count group member", err)
	}
	return true, nil
}
