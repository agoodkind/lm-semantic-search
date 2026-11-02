.mode json
SELECT type AS record_type, properties AS property_value
FROM edges
WHERE type IN ('ASYNC_CALLS', 'IMPORTS')
UNION ALL
SELECT label AS record_type, properties AS property_value
FROM nodes
WHERE label = 'Route';
