SELECT json_group_array(json_array(hex(owner_id), row_key))
FROM ({{keys}})
