#!/usr/bin/env bash
# Install a pinned tree-sitter CLI into a destination directory when no
# tree-sitter is already on PATH. The Swift grammar submodule ships only its
# grammar definition, so the parser is produced at build time by
# `tree-sitter generate`; pinning the CLI keeps that generated output
# reproducible across machines and CI. Downloads the official prebuilt release
# binary so a fresh macOS or Debian/Ubuntu host needs no npm, cargo, or brew.
set -euo pipefail

readonly TREE_SITTER_VERSION="0.25.10"
readonly DEST_DIR="${1:?usage: install-tree-sitter.sh <dest-dir>}"
readonly DEST_BIN="${DEST_DIR}/tree-sitter"
readonly RELEASE_BASE="https://github.com/tree-sitter/tree-sitter/releases/download"

# Script-global so the EXIT trap can clean it up regardless of where the run
# stopped; a function-local would be out of scope in the trap under set -u.
GZ_TMP=""

cleanup() {
    if [[ -n "${GZ_TMP}" ]]; then
        rm -f "${GZ_TMP}"
    fi
}
trap cleanup EXIT

detect_os() {
    local kernel
    kernel="$(uname -s)"
    case "${kernel}" in
        Linux) echo "linux" ;;
        Darwin) echo "macos" ;;
        *) echo "unsupported:${kernel}" ;;
    esac
}

detect_arch() {
    local machine
    machine="$(uname -m)"
    case "${machine}" in
        x86_64 | amd64) echo "x64" ;;
        aarch64 | arm64) echo "arm64" ;;
        *) echo "unsupported:${machine}" ;;
    esac
}

main() {
    if [[ -x "${DEST_BIN}" ]]; then
        echo "install-tree-sitter: ${DEST_BIN} already present"
        return 0
    fi

    local os arch
    os="$(detect_os)"
    arch="$(detect_arch)"
    if [[ "${os}" == unsupported:* || "${arch}" == unsupported:* ]]; then
        echo "install-tree-sitter: ${os} ${arch} not supported; install tree-sitter manually" >&2
        return 1
    fi

    local asset url
    asset="tree-sitter-${os}-${arch}.gz"
    url="${RELEASE_BASE}/v${TREE_SITTER_VERSION}/${asset}"
    mkdir -p "${DEST_DIR}"
    GZ_TMP="$(mktemp)"

    echo "install-tree-sitter: downloading tree-sitter v${TREE_SITTER_VERSION} (${os}/${arch})"
    curl -fsSL "${url}" -o "${GZ_TMP}" || {
        echo "install-tree-sitter: download failed: ${url}" >&2
        return 1
    }
    gunzip -c "${GZ_TMP}" >"${DEST_BIN}"
    chmod +x "${DEST_BIN}"
    echo "install-tree-sitter: installed ${DEST_BIN}"
}

main
