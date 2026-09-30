SELECT occurrences.owner_id, occurrences.row_key, occurrences.vector_id,
       source_blobs.content, occurrences.occurrence_hash, occurrences.generation_order
FROM occurrences
JOIN source_blobs ON source_blobs.blob_id = occurrences.source_blob_id
WHERE occurrences.namespace = ?
ORDER BY occurrences.owner_id, occurrences.sort_key;
