INSERT INTO lexical_df (namespace, term_hash, df)
SELECT ?, term_hash, df FROM frequencies WHERE df > 0
ON CONFLICT (namespace, term_hash) DO UPDATE SET df = excluded.df
