SELECT owner_id, row_key FROM filter_sets WHERE node = :node
ORDER BY owner_id, row_key LIMIT :batch_size
