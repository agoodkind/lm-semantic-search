DELETE FROM project_summaries;
CREATE TRIGGER block_decision_restore BEFORE INSERT ON project_summaries
BEGIN
  SELECT RAISE(ABORT, 'blocked');
END;
