INSERT INTO receipt_rows (namespace, owner_id, generation_order, generation_token, row_key, occurrence_hash)
SELECT namespace, owner_id, generation_order, generation_token, row_key, occurrence_hash
FROM staged_occurrences
WHERE namespace = ? AND owner_id = ? AND generation_order = ? AND generation_token = ?
