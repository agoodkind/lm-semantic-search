SELECT o.row_key, o.generation_order, o.sort_key,
    s.blob_id, o.search_hash, v.input_hash,
    json_patch(
        (SELECT json_group_object(b.column_name, json_array(b.type, b.is_null, b.string_value, b.int64_value, b.bool_value))
         FROM occurrence_scalars b
         WHERE b.namespace = o.namespace AND b.owner_id = o.owner_id AND b.row_key = o.row_key),
        (SELECT json_group_object(e.column_name, json_array(e.type, e.is_null, e.string_value, e.int64_value, e.bool_value))
         FROM effective_scalars e
         WHERE e.namespace = o.namespace AND e.owner_id = o.owner_id AND e.row_key = o.row_key))
FROM occurrences o
LEFT JOIN source_blobs s ON s.blob_id = o.source_blob_id
LEFT JOIN vectors v ON v.vector_id = o.vector_id
WHERE o.namespace = ? AND o.owner_id = ?
ORDER BY o.row_key
