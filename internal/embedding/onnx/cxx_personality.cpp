// libtokenizers.a references the hidden symbol DW.ref.__gxx_personality_v0.
// Only a C++ object with an exception handler defines that symbol, and a Linux
// PIE link fails when no object in the binary defines it. This function's
// try block makes the compiler emit the symbol.
extern "C" void lms_cxx_personality_anchor(void) {
    try {
        throw 0;
    } catch (int) {
    }
}
