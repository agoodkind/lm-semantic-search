SELECT v.rowid AS row_id, v.owner_id, v.row_key FROM effective_scalars v
WHERE v.namespace = :namespace AND v.column_name = :column
AND {{predicate}}
{{cursor}}
ORDER BY v.rowid LIMIT :batch_size
