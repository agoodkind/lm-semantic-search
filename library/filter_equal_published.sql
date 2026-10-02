SELECT v.rowid AS row_id, v.owner_id, v.row_key FROM occurrence_scalars v
WHERE v.namespace = :namespace AND v.column_name = :column
AND NOT EXISTS (
    SELECT 1 FROM effective_scalars e WHERE e.namespace = v.namespace
    AND e.owner_id = v.owner_id AND e.row_key = v.row_key
    AND e.column_name = v.column_name
)
AND {{predicate}}
{{cursor}}
ORDER BY v.rowid LIMIT :batch_size
