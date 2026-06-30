#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CBM_DIR="${ROOT_DIR}/third_party/cbm"
BUILD_DIR="${ROOT_DIR}/build/cbm-engine"
OBJECT_DIR="${BUILD_DIR}/obj"
KEEP_LIST="${BUILD_DIR}/cbm-exported-symbols.txt"
COMBINED_OBJECT="${BUILD_DIR}/libcbm_engine.o"
ARCHIVE_PATH="${ROOT_DIR}/build/libcbm_engine.a"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build-cbm-engine: darwin/arm64 is required" >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "build-cbm-engine: darwin/arm64 is required" >&2
    exit 1
fi

if [[ ! -f "${CBM_DIR}/Makefile.cbm" ]]; then
    echo "build-cbm-engine: ${CBM_DIR}/Makefile.cbm is missing" >&2
    exit 1
fi

mkdir -p "${OBJECT_DIR}"
rm -rf "${OBJECT_DIR:?}"/*

makefile_var() {
    local variable_name="$1"
    local print_makefile="${BUILD_DIR}/print-${variable_name}.mk"

    cat >"${print_makefile}" <<'MAKE'
print-%:
	@printf '%s\n' "$($*)"
MAKE
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

cat >"${KEEP_LIST}" <<'EOF'
_cbm_alloc_init
_cbm_store_open_path
_cbm_store_open_path_query
_cbm_store_close
_cbm_pipeline_new
_cbm_pipeline_run
_cbm_pipeline_free
_cbm_mcp_server_new
_cbm_mcp_server_set_project
_cbm_mcp_server_set_config
_cbm_mcp_handle_tool
_cbm_mcp_server_free
_cbm_cypher_execute
_cbm_cypher_result_free
EOF

ld -arch arm64 -r -exported_symbols_list "${KEEP_LIST}" -o "${COMBINED_OBJECT}" "${objects[@]}"
libtool -static -o "${ARCHIVE_PATH}" "${COMBINED_OBJECT}"

nm_output="$(nm -gU "${ARCHIVE_PATH}")"
printf '%s\n' "${nm_output}"

non_cbm_symbols="$(
    printf '%s\n' "${nm_output}" |
        awk '/ _/ { symbol = $NF; sub(/^_/, "", symbol); if (symbol !~ /^cbm_/) print symbol }'
)"

if [[ -n "${non_cbm_symbols}" ]]; then
    echo "build-cbm-engine: non-cbm global symbols remain:" >&2
    printf '%s\n' "${non_cbm_symbols}" >&2
    exit 1
fi
