DELETE FROM lexical_occurrences
WHERE namespace = ? AND (owner_id, row_key) IN ({{rows}})
RETURNING owner_id, row_key, search_hash
