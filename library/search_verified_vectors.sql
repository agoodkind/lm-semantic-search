SELECT o.owner_id, o.row_key, o.sort_key, v.vector_id, v.identity_digest, v.vector_checksum
FROM occurrences o
JOIN vectors v ON v.vector_id = o.vector_id
WHERE o.namespace = 'verified'
ORDER BY v.vector_id;
