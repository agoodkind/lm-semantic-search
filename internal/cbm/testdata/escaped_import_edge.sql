UPDATE nodes SET file_path = '' WHERE label = 'File' AND name = 'main.js';
INSERT INTO edges (project, source_id, target_id, type, properties)
SELECT source.project, source.id, target.id, 'IMPORTS', '{"local_name":"Cf' || char(92) || 'u0067"}'
FROM nodes AS source, nodes AS target
WHERE source.label = 'File'
  AND source.name = 'main.js'
  AND target.label = 'Module'
  AND target.file_path = 'settings.js';
INSERT INTO edges (project, source_id, target_id, type, properties)
SELECT source.project, source.id, target.id, 'IMPORTS', '{"local_name":"Other"}'
FROM nodes AS source, nodes AS target
WHERE source.label = 'File'
  AND source.name = 'main.js'
  AND target.label = 'Module'
  AND target.file_path = 'unrelated.js';
SELECT count(*) FROM edges WHERE type = 'IMPORTS';
