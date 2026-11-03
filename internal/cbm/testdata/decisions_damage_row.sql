.output /dev/null
.dbconfig defensive off
.output stdout
PRAGMA writable_schema = ON;
UPDATE sqlite_master SET rootpage = 999999 WHERE name = 'project_summaries';
PRAGMA writable_schema = OFF;
PRAGMA schema_version = 999999;
