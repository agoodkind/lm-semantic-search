// Package dart exposes the tree-sitter Dart grammar to the splitter. The
// grammar's generated parser and external scanner live in the pinned upstream/
// git submodule and are each compiled as their own translation unit through the
// grammar_parser.c and grammar_scanner.c shims in this directory, which keeps
// their macros from colliding. Dart has no maintained Go-module binding against
// this runtime, so the grammar is pinned as a submodule rather than a module
// dependency.
package dart

// #cgo CFLAGS: -std=c11 -fPIC -I${SRCDIR}/upstream/src
// typedef struct TSLanguage TSLanguage;
// const TSLanguage *tree_sitter_dart(void);
import "C"

import "unsafe"

// Language returns the tree-sitter Language pointer for Dart.
func Language() unsafe.Pointer {
	return unsafe.Pointer(C.tree_sitter_dart())
}
