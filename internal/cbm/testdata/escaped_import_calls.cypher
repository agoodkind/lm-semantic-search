MATCH (caller:Function {name:'run'})-[:CALLS]->(target) RETURN target.qualified_name LIMIT 10
