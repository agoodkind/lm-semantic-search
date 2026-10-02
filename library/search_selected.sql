SELECT json_group_array(json_object(
    'owner_hex', hex(c.owner_id), 'row_key', c.row_key,
    'sort_key', c.sort_key, 'vector_id', c.vector_id,
    'identity_digest', c.identity_digest, 'vector_checksum', c.vector_checksum,
    'source_blob_id', c.source_blob_id, 'search_hash', c.search_hash,
    'group_key', c.group_key, 'scalars', c.scalars || ''
))
FROM (
    {{columns}}
    AND (o.owner_id, o.row_key) IN (VALUES {{keys}})
) c
