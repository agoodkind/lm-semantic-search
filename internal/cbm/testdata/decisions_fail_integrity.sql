.output /dev/null
.dbconfig no_ckpt_on_close on
.output stdout
INSERT INTO projects (name, indexed_at, root_path) VALUES
  ('extra-1', '2026-01-01T00:00:00Z', '/extra-1'),
  ('extra-2', '2026-01-01T00:00:00Z', '/extra-2'),
  ('extra-3', '2026-01-01T00:00:00Z', '/extra-3'),
  ('extra-4', '2026-01-01T00:00:00Z', '/extra-4'),
  ('extra-5', '2026-01-01T00:00:00Z', '/extra-5'),
  ('extra-6', '2026-01-01T00:00:00Z', '/extra-6');
UPDATE project_summaries SET summary = 'Synthetic WAL decision record';
