// Compiles the generated Swift parser from the pinned upstream submodule as its
// own translation unit. The parser is produced by the `grammars` Makefile
// target and is not stored in this repository.
#include "upstream/src/parser.c"
