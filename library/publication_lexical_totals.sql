WITH changes AS (
    SELECT json_extract(value, '$.hash') AS search_hash,
           json_extract(value, '$.delta') AS delta FROM json_each(?)
)
SELECT COUNT(*), COALESCE(SUM(delta), 0),
       COALESCE(SUM(delta * document_length), 0)
FROM changes JOIN lexical_content USING (search_hash)
