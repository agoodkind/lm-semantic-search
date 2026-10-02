SELECT search_hash, analyzer_identity, document_length
FROM lexical_content WHERE search_hash IN ({{hashes}})
