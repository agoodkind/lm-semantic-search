DELETE FROM lexical_df WHERE namespace = ?
AND term_hash IN (SELECT term_hash FROM frequencies WHERE df = 0)
