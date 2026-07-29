# Tool calls carry what the user saw

Status: proposed
Date: 2026-07-29

Scope: what a conversation tool call sends and stores. Message text and reasoning are unchanged.

## The rule

A tool call sends the text the user saw. Nothing else.

A shell call sends its command. A file read sends the path it read. A tool whose input is a document sends that document. The harness's serialization of the call never leaves clyde.

## What sends it

Each provider's parser. Only the provider knows its own harness's tool shapes, so only the provider can say what the user saw.

`input_json` is removed from the wire. `deriveToolCommandAndLang` is deleted; it exists to re-guess a command from two key names after the parser has already read the input, and it returns nothing usable for every tool that is not a shell.

## What the wire carries

One tool call carries its name, the text the user saw, a language hint, its output, and whether it errored.

The language hint stays because the engine splits and decomposes by format. Shell is a language, not a harness, so the engine may keep parsing it into program names and file paths.

## What the engine stores

One row per tool call: the tool's name, the program names and file paths extracted from a shell command when there is one, then the text.

```
Read
/tmp/phone.png
```

```
Bash
df
cat
/Volumes/Chaos Storage
df -h "/Volumes/Chaos Storage" | cat
```

The row lives at `convtool/<conversation>/<message>/<toolIndex>`.

## What this replaces

A tool call stored three rows and two of them carried the same text.

The engine appended the arguments to its summary row, truncated at 2,000 bytes, and stored the same arguments again as their own row split at the 60,000-byte budget. Measured across 364 tool calls in one conversation, the arguments row's entire text is a literal substring of the summary row's text in 339 of them, 93%, and that repeat is 27.8% of the conversation's tool-row text. By cosine similarity, the store's own metric, the pair has a median of 0.961, so a search cannot separate them.

Across the collection, 185,708 arguments rows sit beside 163,324 summary rows, and roughly 173,000 of them repeat text already stored, about 6.6% of everything.

The serialization also fills the corpus with text no one wrote. The two most repeated strings in the store are `{"type":"input_text",\n"text":` and `}\n]`.

## Size

The 2,000-byte truncation goes away. The row splits at the ordinary conversation chunk budget.

96.5% of tool calls never needed splitting: of 163,324 argument payloads, 5,663 required a second piece. When a call does split, every piece begins with the tool's name so a search for the tool matches each piece. Derived rows are compared by path and content hash rather than reassembled, so repeating the name costs nothing in convergence.

## Existing rows

Nothing deletes them.

A conversation reaching the store again finds one expected path where three stored paths used to be. The comparison sees a message whose derived rows no longer match, replaces that message's rows, and three become one. The corpus reaches the new shape through ordinary re-ingest and shrinks as it does.

Until a conversation is re-ingested it keeps its three rows and stays searchable exactly as before.

## Tool results

Tool results are not stored. The content policy offers chat turns and tool calls, where a tool call means the invocation without what it returned. The 455,445 result rows in the store predate that policy and this change does not touch them.

## What proves it

No stored row holds harness serialization. The strings `{"type":"input_text"` and `"file_path":` appear in no newly written row.

A conversation ingested after the change stores one row per tool call, holding the tool name and the text the user saw.

The same conversation ingested twice does nothing the second time.

A tool call is still found by tool name, by program name, by file path, and by command text.

A tool call whose text exceeds the chunk budget splits, and every piece names the tool.
