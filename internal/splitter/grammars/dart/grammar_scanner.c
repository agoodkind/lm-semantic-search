// Compiles the Dart external scanner from the pinned upstream submodule as its
// own translation unit, separate from the parser so their macros do not collide.
#include "upstream/src/scanner.c"
