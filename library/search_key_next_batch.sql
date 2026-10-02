SELECT owner_id, row_key FROM filter_sets
WHERE node = :node AND (owner_id, row_key) > (:after_owner, :after_row)
ORDER BY owner_id, row_key LIMIT :batch_size
