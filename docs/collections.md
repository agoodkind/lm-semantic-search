# Generic collection ingestion

Registered document collections use per-item fingerprints to track ingestion.
The [service contract](../proto/lmsemanticsearch/v1/service.proto) defines their
RPC fields, stream framing, scalar declarations, and errors.

## Checkpoints

A manifest fingerprint represents one complete item. A changed item without
delivered rows remains pending, and its checkpoint does not advance.
A processed item advances its checkpoint even when every delivered row is
already stored and the job performs no vector writes.

A bounded manifest reply selects modified items before new items. A cursor
rotates overflow across later replies.

## Row families

A row family includes one client row and all its split parts. An ordinary
delivery inserts a family only when no stored part has usable, nonblank content.
The stored part's path must equal the row key or begin with the row key followed
by a slash. A stored family retains its content when the client changes its text
or omits that row from a later delivery.

The job reads stored rows for all delivered items in one batch, using the
collection's declared item ID column. Each item uses that shared read to select
missing families and reuse existing vectors.

## Bootstrap and vector reuse

An empty ingestion diff triggers bootstrap when the stored collection is missing,
or when the collection is known to contain zero rows despite a nonempty
checkpoint. An unknown row count does not trigger that rebuild.

A bootstrap reuses vectors from an existing live collection unless the job forces
a rebuild. When the backend is available, bootstrap resumes a staging checkpoint
only if its staging collection still exists. Bootstrap restarts from an empty
checkpoint when the staging collection is missing.
