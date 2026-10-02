SELECT COALESCE(json_group_array(json_object(
    'row_id', row_id, 'owner_hex', hex(owner_id), 'row_key', row_key
)), '[]') FROM ({{rows}})
