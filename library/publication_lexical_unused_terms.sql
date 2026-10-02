DELETE FROM lexical_terms
WHERE search_hash IN (
    SELECT json_extract(value, '$.hash') FROM json_each(?)
    WHERE json_extract(value, '$.delta') < 0
)
AND NOT EXISTS (SELECT 1 FROM lexical_occurrences
                WHERE lexical_occurrences.search_hash = lexical_terms.search_hash)
