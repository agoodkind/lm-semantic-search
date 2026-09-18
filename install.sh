#!/usr/bin/env bash
set -euo pipefail

REPO="agoodkind/lm-semantic-search"
HOSTED_INSTALLER_URL="https://raw.githubusercontent.com/agoodkind/go-makefile/main/install.sh"
CLI_BINARY="lm-semantic-search"

BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
HOSTED_ARGS=()
INSTALL_ARGS=()

usage() {
    cat <<'USAGE'
install.sh installs lm-semantic-search release binaries from GitHub.

It installs the lm-semantic-search CLI through the go-makefile hosted
installer, then runs `lm-semantic-search install`, which installs the daemon
and MCP adapter from the same release, stages ONNX Runtime beside the daemon,
links lms to the CLI, and installs the daemon user service.

Usage:
  ./install.sh [flags]

Flags:
  --bin-dir PATH         install dir (default: $XDG_BIN_HOME or $HOME/.local/bin)
  --no-service          skip launchd/systemd user service setup
  --bin-only            compatibility alias for --no-service
  --version TAG         install this release tag instead of the latest release
  --require-attestation require an attestation for the hosted installer's own download
  -h, --help            show this help

Exit codes:
  0 success
  1 usage error
  2 install or service setup failure
USAGE
}

usage_error() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

install_error() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 2
}

need() {
    command -v "$1" >/dev/null 2>&1 || install_error "missing dependency: $1"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --bin-dir)
                shift
                if [[ $# -eq 0 ]]; then
                    usage_error "--bin-dir requires a path"
                fi
                BIN_DIR="$1"
                ;;
            --no-service | --bin-only)
                INSTALL_ARGS+=("--no-service")
                ;;
            --version)
                shift
                if [[ $# -eq 0 ]]; then
                    usage_error "--version requires a value"
                fi
                HOSTED_ARGS+=("--version" "$1")
                INSTALL_ARGS+=("--version" "$1")
                ;;
            --require-attestation)
                HOSTED_ARGS+=("--require-attestation")
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                usage_error "unknown flag: $1 (try --help)"
                ;;
        esac
        shift
    done
}

main() {
    parse_args "$@"
    need bash
    need curl

    printf 'install.sh: installing %s through the go-makefile hosted installer\n' "$CLI_BINARY" >&2
    # The ${name[@]+...} form expands an empty array to nothing; bash 3.2, the
    # macOS /bin/bash, treats a bare empty "${name[@]}" as unbound under set -u.
    if ! curl -fsSL "$HOSTED_INSTALLER_URL" | bash -s -- \
        --repo "$REPO" \
        --binary "$CLI_BINARY" \
        --bin-dir "$BIN_DIR" \
        ${HOSTED_ARGS[@]+"${HOSTED_ARGS[@]}"} \
        -- install --bin-dir "$BIN_DIR" ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}; then
        install_error "install failed"
    fi
}

main "$@"
