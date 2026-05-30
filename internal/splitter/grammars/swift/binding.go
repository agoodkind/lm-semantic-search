// Package swift exposes the tree-sitter Swift grammar to the splitter. The
// upstream repository commits the grammar definition (src/grammar.json) and the
// external scanner but not the generated parser, so the parser is produced from
// the pinned upstream/ git submodule by the tree-sitter CLI (the `grammars`
// Makefile target). The generated parser.c and the scanner.c are each compiled
// as their own translation unit through the grammar_parser.c and
// grammar_scanner.c shims in this directory, which keeps their macros from
// colliding. No generated file is stored in this repository.
package swift

// #cgo CFLAGS: -std=c11 -fPIC -I${SRCDIR}/upstream/src
// typedef struct TSLanguage TSLanguage;
// const TSLanguage *tree_sitter_swift(void);
import "C"

import "unsafe"

// Language returns the tree-sitter Language pointer for Swift.
func Language() unsafe.Pointer {
	return unsafe.Pointer(C.tree_sitter_swift())
}
