#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CBM_DIR="${ROOT_DIR}/third_party/cbm"
TARGET_GOOS="${GO_MK_TARGET_GOOS:-$(go env GOOS)}"
TARGET_GOARCH="${GO_MK_TARGET_GOARCH:-$(go env GOARCH)}"
PREFIX="${GO_MK_CGO_PREFIX:-${ROOT_DIR}/.make/cgo/${TARGET_GOOS}-${TARGET_GOARCH}}"
BUILD_DIR="${ROOT_DIR}/.make/cbm-engine/${TARGET_GOOS}-${TARGET_GOARCH}"
OBJECT_DIR="${BUILD_DIR}/obj"
KEEP_LIST="${BUILD_DIR}/cbm-exported-symbols.txt"
COMBINED_OBJECT="${BUILD_DIR}/libcbm_engine.o"
ARCHIVE_PATH="${PREFIX}/lib/libcbm_engine.a"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

if [[ -z "${TARGET_GOOS}" ]]; then
    TARGET_GOOS="$(go env GOOS)"
fi
if [[ -z "${TARGET_GOARCH}" ]]; then
    TARGET_GOARCH="$(go env GOARCH)"
fi

if [[ ! -f "${CBM_DIR}/Makefile.cbm" ]]; then
    echo "setup-cgo-cbm: ${CBM_DIR}/Makefile.cbm is missing" >&2
    exit 1
fi

mkdir -p "${OBJECT_DIR}" "${PREFIX}/lib/pkgconfig" "${PREFIX}/include/mcp" \
    "${PREFIX}/include/tree_sitter"
rm -rf "${OBJECT_DIR:?}"/*

makefile_var() {
    local variable_name="$1"
    local print_makefile="${BUILD_DIR}/print-${variable_name}.mk"

    printf '%s\n' 'print-%:' '	@printf '\''%s\n'\'' "$($*)"' >"${print_makefile}"
    MAKEFLAGS= make -C "${CBM_DIR}" -f Makefile.cbm -f "${print_makefile}" \
        --no-print-directory PKG_CONFIG=/usr/bin/false "print-${variable_name}"
}

split_words() {
    local value="$1"
    local -n output_ref="$2"

    # shellcheck disable=SC2206
    output_ref=(${value})
}

single_word_var() {
    local variable_name="$1"
    local value
    local -a words

    value="$(makefile_var "${variable_name}")"
    split_words "${value}" words
    printf '%s\n' "${words[0]}"
}

object_path_for_source() {
    local source_path="$1"
    local object_name

    object_name="${source_path//\//__}"
    object_name="${object_name%.*}.o"
    printf '%s/%s\n' "${OBJECT_DIR}" "${object_name}"
}

compile_c() {
    local source_path="$1"
    local object_path="$2"
    shift 2

    mkdir -p "$(dirname "${object_path}")"
    (cd "${CBM_DIR}" && "${CC}" "$@" -c -o "${object_path}" "${source_path}")
}

compile_cxx() {
    local source_path="$1"
    local object_path="$2"
    shift 2

    mkdir -p "$(dirname "${object_path}")"
    (cd "${CBM_DIR}" && "${CXX}" "$@" -c -o "${object_path}" "${source_path}")
}

tool_for_cc() {
    local tool_name="$1"
    local compiler_basename
    local compiler_dir
    local candidate
    local executable_name

    if [[ -n "${!tool_name:-}" ]]; then
        printf '%s\n' "${!tool_name}"
        return
    fi

    case "${tool_name}" in
        AR)
            executable_name="ar"
            ;;
        OBJCOPY)
            executable_name="objcopy"
            ;;
        *)
            executable_name="${tool_name}"
            ;;
    esac

    compiler_basename="$(basename "${CC}")"
    compiler_dir="$(dirname "${CC}")"
    for suffix in clang gcc cc; do
        if [[ "${compiler_basename}" == *"${suffix}" ]]; then
            candidate="${compiler_dir}/${compiler_basename%"${suffix}"}${executable_name}"
            if command -v "${candidate}" >/dev/null 2>&1; then
                printf '%s\n' "${candidate}"
                return
            fi
        fi
    done

    printf '%s\n' "${executable_name}"
}

archive_tool() {
    if [[ -n "${AR:-}" ]]; then
        printf '%s\n' "${AR}"
        return
    fi
    tool_for_cc "AR"
}

objcopy_tool() {
    if [[ -n "${OBJCOPY:-}" ]]; then
        printf '%s\n' "${OBJCOPY}"
        return
    fi
    tool_for_cc "OBJCOPY"
}

link_darwin_relocatable() {
    local arch_flag

    case "${TARGET_GOARCH}" in
        arm64)
            arch_flag="arm64"
            ;;
        amd64)
            arch_flag="x86_64"
            ;;
        *)
            echo "setup-cgo-cbm: unsupported darwin GOARCH ${TARGET_GOARCH}" >&2
            exit 1
            ;;
    esac

    "${CC}" -r -arch "${arch_flag}" -Wl,-exported_symbols_list,"${KEEP_LIST}" \
        -o "${COMBINED_OBJECT}" "${objects[@]}"
}

link_linux_relocatable() {
    local objcopy_bin

    "${CC}" -r -nostdlib -o "${COMBINED_OBJECT}" "${objects[@]}"
    objcopy_bin="$(objcopy_tool)"
    "${objcopy_bin}" --keep-global-symbols="${KEEP_LIST}" "${COMBINED_OBJECT}"
}

create_archive() {
    local archive_bin

    rm -f "${ARCHIVE_PATH}"
    archive_bin="$(archive_tool)"
    "${archive_bin}" crs "${ARCHIVE_PATH}" "${COMBINED_OBJECT}"
}

write_keep_list() {
    local symbol_prefix=""

    if [[ "${TARGET_GOOS}" == "darwin" ]]; then
        symbol_prefix="_"
    fi

    for symbol_name in \
        cbm_alloc_init \
        cbm_store_open_path \
        cbm_store_open_path_query \
        cbm_store_close \
        cbm_pipeline_new \
        cbm_pipeline_run \
        cbm_pipeline_free \
        cbm_mcp_server_new \
        cbm_mcp_server_set_project \
        cbm_mcp_server_set_config \
        cbm_mcp_handle_tool \
        cbm_mcp_server_free \
        cbm_cypher_execute \
        cbm_cypher_result_free; do
        printf '%s%s\n' "${symbol_prefix}" "${symbol_name}"
    done >"${KEEP_LIST}"
}

install_headers() {
    cp "${CBM_DIR}/internal/cbm/cbm.h" "${PREFIX}/include/cbm.h"
    cp "${CBM_DIR}/internal/cbm/arena.h" "${PREFIX}/include/arena.h"
    cp "${CBM_DIR}/src/mcp/mcp.h" "${PREFIX}/include/mcp/mcp.h"
    cp "${CBM_DIR}/internal/cbm/vendored/ts_runtime/include/tree_sitter/api.h" \
        "${PREFIX}/include/tree_sitter/api.h"
}

write_pkg_config() {
    local cxx_runtime

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
}

normal_cflags=(
    -std=c11
    -D_DEFAULT_SOURCE
    -D_GNU_SOURCE
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -Isrc
    -Ivendored
    -Ivendored/sqlite3
    -Ivendored/mimalloc/include
    -Ivendored/mimalloc/src
    -Iinternal/cbm
    -Iinternal/cbm/vendored
    -Iinternal/cbm/vendored/lz4
    -Iinternal/cbm/vendored/zstd
    -Iinternal/cbm/vendored/ts_runtime/include
    -Iinternal/cbm/vendored/ts_runtime/src
)

grammar_cflags=(
    -std=c11
    -D_DEFAULT_SOURCE
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -Iinternal/cbm
    -Iinternal/cbm/vendored/ts_runtime/include
    -Iinternal/cbm/vendored/ts_runtime/src
)

cxxflags=(
    -std=c++14
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -Iinternal/cbm
    -Iinternal/cbm/vendored
    -Iinternal/cbm/vendored/ts_runtime/include
)

mimalloc_cflags=(
    -std=c11
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -DMI_OVERRIDE=0
    -Ivendored/mimalloc/include
    -Ivendored/mimalloc/src
)

sqlite_cflags=(
    -std=c11
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -DSQLITE_DQS=0
    -DSQLITE_THREADSAFE=1
    -DSQLITE_ENABLE_FTS5
)

lz4_cflags=(
    -std=c11
    -D_DEFAULT_SOURCE
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -Iinternal/cbm
    -Iinternal/cbm/vendored/lz4
)

zstd_cflags=(
    -std=c11
    -D_DEFAULT_SOURCE
    -O2
    -w
    -fvisibility=hidden
    -fno-common
    -DTREE_SITTER_HIDE_SYMBOLS
    -DCBM_BIND_TS_ALLOCATOR=1
    -Iinternal/cbm/vendored/zstd
)

prod_sources_value="$(makefile_var PROD_SRCS)"
existing_sources_value="$(makefile_var EXISTING_C_SRCS)"
mimalloc_source="$(single_word_var MIMALLOC_SRC)"
sqlite_source="$(single_word_var SQLITE3_SRC)"
preprocessor_source="$(single_word_var PREPROCESSOR_SRC)"
unixcoder_blob_source="$(single_word_var UNIXCODER_BLOB_SRC)"

split_words "${prod_sources_value}" prod_sources
split_words "${existing_sources_value}" existing_sources

objects=()

api_source_needs_default_visibility() {
    local source_path="$1"

    case "${source_path}" in
        internal/cbm/cbm.c | src/store/store.c | src/pipeline/pipeline.c | src/mcp/mcp.c | src/cypher/cypher.c)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

compile_source_group() {
    local source_path
    local object_path
    local -a source_cflags

    for source_path in "$@"; do
        object_path="$(object_path_for_source "${source_path}")"
        if [[ "${source_path}" == internal/cbm/grammar_*.c ]] ||
            [[ "${source_path}" == "internal/cbm/ts_runtime.c" ]] ||
            [[ "${source_path}" == "internal/cbm/lsp_all.c" ]]; then
            compile_c "${source_path}" "${object_path}" "${grammar_cflags[@]}"
        else
            source_cflags=("${normal_cflags[@]}")
            if api_source_needs_default_visibility "${source_path}"; then
                source_cflags+=(-fvisibility=default)
            fi
            compile_c "${source_path}" "${object_path}" "${source_cflags[@]}"
        fi
        objects+=("${object_path}")
    done
}

compile_source_group \
    "${prod_sources[@]}" \
    "${existing_sources[@]}"

mimalloc_object="$(object_path_for_source "${mimalloc_source}")"
compile_c "${mimalloc_source}" "${mimalloc_object}" "${mimalloc_cflags[@]}"
objects+=("${mimalloc_object}")

sqlite_object="$(object_path_for_source "${sqlite_source}")"
compile_c "${sqlite_source}" "${sqlite_object}" "${sqlite_cflags[@]}"
objects+=("${sqlite_object}")

preprocessor_object="$(object_path_for_source "${preprocessor_source}")"
compile_cxx "${preprocessor_source}" "${preprocessor_object}" "${cxxflags[@]}"
objects+=("${preprocessor_object}")

for vendored_source in \
    internal/cbm/vendored/lz4/lz4.c \
    internal/cbm/vendored/lz4/lz4hc.c; do
    vendored_object="$(object_path_for_source "${vendored_source}")"
    compile_c "${vendored_source}" "${vendored_object}" "${lz4_cflags[@]}"
    objects+=("${vendored_object}")
done

vendored_zstd_object="$(object_path_for_source "internal/cbm/vendored/zstd/zstd.c")"
compile_c "internal/cbm/vendored/zstd/zstd.c" "${vendored_zstd_object}" "${zstd_cflags[@]}"
objects+=("${vendored_zstd_object}")

unixcoder_object="$(object_path_for_source "${unixcoder_blob_source}")"
(cd "${CBM_DIR}" && "${CC}" -c -o "${unixcoder_object}" "${unixcoder_blob_source}")
objects+=("${unixcoder_object}")

write_keep_list
case "${TARGET_GOOS}" in
    darwin)
        link_darwin_relocatable
        ;;
    linux)
        link_linux_relocatable
        ;;
    *)
        echo "setup-cgo-cbm: unsupported GOOS ${TARGET_GOOS}" >&2
        exit 1
        ;;
esac
create_archive
install_headers
write_pkg_config

echo "setup-cgo-cbm: installed ${ARCHIVE_PATH}"
