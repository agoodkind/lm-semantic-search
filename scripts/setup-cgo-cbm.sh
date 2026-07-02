#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CBM_DIR="${ROOT_DIR}/third_party/cbm"
TARGET_GOOS="${GO_MK_TARGET_GOOS:-$(go env GOOS)}"
TARGET_GOARCH="${GO_MK_TARGET_GOARCH:-$(go env GOARCH)}"
PREFIX="${GO_MK_CGO_PREFIX:-${ROOT_DIR}/.make/cgo/${TARGET_GOOS}-${TARGET_GOARCH}}"
# go.mk resolves the workflow-provided cross toolchain (GO_MK_CC/GO_MK_CXX)
# into CC/CXX at the go-mk-cgo-deps hook, so plain CC/CXX is already the
# target's compiler in every invocation context; cc/c++ covers a direct
# script run outside make.
CC="${CC:-cc}"
CXX="${CXX:-c++}"

if [[ -z "${TARGET_GOOS}" ]]; then
    TARGET_GOOS="$(go env GOOS)"
fi
if [[ -z "${TARGET_GOARCH}" ]]; then
    TARGET_GOARCH="$(go env GOARCH)"
fi
if [[ "${PREFIX}" != /* ]]; then
    PREFIX="${ROOT_DIR}/${PREFIX}"
fi

ARCHIVE_PATH="${PREFIX}/lib/libcbm_engine.a"

if [[ ! -f "${CBM_DIR}/Makefile.cbm" ]]; then
    echo "setup-cgo-cbm: ${CBM_DIR}/Makefile.cbm is missing" >&2
    exit 1
fi

last_word() {
    local value="$1"
    local -a words

    read -r -a words <<<"${value}"
    printf '%s\n' "${words[${#words[@]} - 1]}"
}

tool_for_compiler() {
    local fallback_tool="$1"
    local compiler
    local compiler_basename
    local compiler_dir
    local suffix
    local candidate_basename
    local candidate
    local triple
    local arch
    local resolved_compiler

    compiler="$(last_word "${CC}")"
    compiler_basename="$(basename "${compiler}")"
    compiler_dir="$(dirname "${compiler}")"

    for suffix in clang gcc cc; do
        if [[ "${compiler_basename}" == *"${suffix}" ]]; then
            candidate_basename="${compiler_basename%"${suffix}"}${fallback_tool}"
            if [[ "${compiler_dir}" == "." ]]; then
                candidate="${candidate_basename}"
            else
                candidate="${compiler_dir}/${candidate_basename}"
            fi
            if command -v "${candidate}" >/dev/null 2>&1; then
                printf '%s\n' "${candidate}"
                return
            fi
        fi
    done

    # osxcross names its cctools by full target triple (arm64-apple-darwin26.1-ar)
    # and only aliases the clang wrappers with the o64/oa64 shorthand, so the
    # suffix swap above misses them. Derive the triple from the compiler and
    # search the compiler's own directory; the GNU ar fallback would build an
    # archive whose index ld64 rejects ("archive has no table of contents").
    triple="$("${compiler}" -print-target-triple 2>/dev/null || true)"
    resolved_compiler="$(command -v "${compiler}" 2>/dev/null || true)"
    if [[ -n "${triple}" && -n "${resolved_compiler}" ]]; then
        arch="${triple%%-*}"
        local -a triple_candidates
        triple_candidates=( "$(dirname "${resolved_compiler}")/${arch}"-*-"${fallback_tool}" )
        if [[ -e "${triple_candidates[0]}" ]]; then
            printf '%s\n' "${triple_candidates[0]}"
            return
        fi
    fi

    printf '%s\n' "${fallback_tool}"
}

if [[ -n "${AR:-}" ]]; then
    LMS_AR="${AR}"
else
    LMS_AR="$(tool_for_compiler ar)"
fi

if [[ -n "${OBJCOPY:-}" ]]; then
    LMS_OBJCOPY="${OBJCOPY}"
else
    LMS_OBJCOPY="$(tool_for_compiler objcopy)"
fi

mkdir -p "${PREFIX}/include/mcp" "${PREFIX}/include/tree_sitter" \
    "${PREFIX}/lib/pkgconfig"

make -C "${CBM_DIR}" \
    -f Makefile.cbm \
    -f "${ROOT_DIR}/scripts/cbm-lib.mk" \
    lms-cbm-lib \
    CC="${CC}" \
    CXX="${CXX}" \
    LMS_ARCHIVE="${ARCHIVE_PATH}" \
    LMS_TARGET_GOOS="${TARGET_GOOS}" \
    LMS_TARGET_GOARCH="${TARGET_GOARCH}" \
    LMS_AR="${LMS_AR}" \
    LMS_OBJCOPY="${LMS_OBJCOPY}"

cp "${CBM_DIR}/internal/cbm/cbm.h" "${PREFIX}/include/cbm.h"
cp "${CBM_DIR}/internal/cbm/arena.h" "${PREFIX}/include/arena.h"
cp "${CBM_DIR}/src/mcp/mcp.h" "${PREFIX}/include/mcp/mcp.h"
cp "${CBM_DIR}/internal/cbm/vendored/ts_runtime/include/tree_sitter/api.h" \
    "${PREFIX}/include/tree_sitter/api.h"

case "${TARGET_GOOS}" in
    darwin)
        cxx_runtime="-lc++"
        ;;
    linux)
        cxx_runtime="-lstdc++"
        ;;
    *)
        echo "setup-cgo-cbm: unsupported GOOS ${TARGET_GOOS}" >&2
        exit 1
        ;;
esac

cat >"${PREFIX}/lib/pkgconfig/cbm.pc" <<PC
prefix=${PREFIX}
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: cbm
Description: codebase-memory graph engine
Version: 0
Cflags: -I\${includedir}
Libs: -L\${libdir} -lcbm_engine ${cxx_runtime} -lm -lz
PC

echo "setup-cgo-cbm: installed ${ARCHIVE_PATH}"
