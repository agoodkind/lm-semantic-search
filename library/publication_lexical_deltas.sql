WITH changes AS (
    SELECT json_extract(value, '$.hash') AS search_hash,
           json_extract(value, '$.delta') AS delta
    FROM json_each(?)
), term_changes AS (
    SELECT lexical_terms.term_hash, SUM(changes.delta) AS delta
    FROM changes JOIN lexical_terms USING (search_hash)
    GROUP BY lexical_terms.term_hash
), frequencies AS (
    SELECT term_changes.term_hash, COALESCE(lexical_df.df, 0) + delta AS df
    FROM term_changes LEFT JOIN lexical_df
      ON lexical_df.namespace = ? AND lexical_df.term_hash = term_changes.term_hash
)
