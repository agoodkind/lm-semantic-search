#!/usr/bin/env bash
set -euo pipefail

REPO="agoodkind/lm-semantic-search"
HOSTED_INSTALLER_URL="https://raw.githubusercontent.com/agoodkind/go-makefile/main/install.sh"
DAEMON_BINARY="lm-semantic-search-daemon"
CLI_BINARY="lm-semantic-search"
MCP_BINARY="lm-semantic-search-mcp"
ONNX_RUNTIME_VERSION="1.27.0"
ONNX_RUNTIME_AMD64_ARCHIVE="onnxruntime-linux-x64-1.27.0"
ONNX_RUNTIME_AMD64_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-1.27.0.tgz"
ONNX_RUNTIME_AMD64_SHA256="547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f"
ONNX_RUNTIME_ARM64_ARCHIVE="onnxruntime-linux-aarch64-1.27.0"
ONNX_RUNTIME_ARM64_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-aarch64-1.27.0.tgz"
ONNX_RUNTIME_ARM64_SHA256="3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48"

BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
INSTALL_SERVICE=1
HOSTED_ARGS=()

usage() {
    cat <<'USAGE'
install.sh installs lm-semantic-search release binaries from GitHub.

Usage:
  ./install.sh [flags]

Flags:
  --bin-dir PATH         install dir (default: $XDG_BIN_HOME or $HOME/.local/bin)
  --no-service          skip launchd/systemd user service setup
  --bin-only            compatibility alias for --no-service
  --version TAG         pass a release tag to the hosted installer
  --channel rolling|stable
                         pass a release channel to the hosted installer
  --require-attestation pass attestation requirement to the hosted installer
  -h, --help            show this help

Exit codes:
  0 success
  1 usage or unsupported platform
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
    local flag

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
                INSTALL_SERVICE=0
                ;;
            --version | --channel)
                flag="$1"
                shift
                if [[ $# -eq 0 ]]; then
                    usage_error "$flag requires a value"
                fi
                HOSTED_ARGS+=("$flag" "$1")
                ;;
            --require-attestation)
                HOSTED_ARGS+=("$1")
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

install_release_binary() {
    local binary="$1"

    need bash
    need curl
    printf 'install.sh: installing %s through go-makefile hosted installer\n' "$binary" >&2
    curl -fsSL "$HOSTED_INSTALLER_URL" | bash -s -- \
        --repo "$REPO" \
        --binary "$binary" \
        --bin-dir "$BIN_DIR" \
        "${HOSTED_ARGS[@]}"
}

install_linux_onnxruntime() (
    local archive_name
    local archive_path
    local archive_sha256
    local archive_url
    local checksum_output
    local extracted_directory
    local temporary_directory
    local versioned_library="libonnxruntime.so.$ONNX_RUNTIME_VERSION"

    case "$(uname -m)" in
        x86_64 | amd64)
            archive_name="$ONNX_RUNTIME_AMD64_ARCHIVE"
            archive_url="$ONNX_RUNTIME_AMD64_URL"
            archive_sha256="$ONNX_RUNTIME_AMD64_SHA256"
            ;;
        aarch64 | arm64)
            archive_name="$ONNX_RUNTIME_ARM64_ARCHIVE"
            archive_url="$ONNX_RUNTIME_ARM64_URL"
            archive_sha256="$ONNX_RUNTIME_ARM64_SHA256"
            ;;
        *)
            usage_error "unsupported Linux architecture: $(uname -m)"
            ;;
    esac

    need curl
    need install
    need ln
    need sha256sum
    need tar

    temporary_directory="$(mktemp -d -t lm-semantic-search-onnx.XXXXXX)" ||
        install_error "could not create ONNX Runtime temp directory"
    trap 'rm -rf -- "$temporary_directory"' EXIT
    archive_path="$temporary_directory/onnxruntime.tgz"
    extracted_directory="$temporary_directory/extracted"
    mkdir -p "$extracted_directory"

    if ! curl -fsSL "$archive_url" -o "$archive_path"; then
        install_error "could not download ONNX Runtime $ONNX_RUNTIME_VERSION"
    fi
    if ! checksum_output="$(sha256sum "$archive_path")"; then
        install_error "could not verify ONNX Runtime archive checksum"
    fi
    if [[ "${checksum_output%% *}" != "$archive_sha256" ]]; then
        install_error "ONNX Runtime archive checksum mismatch"
    fi
    if ! tar -xzf "$archive_path" -C "$extracted_directory"; then
        install_error "could not extract ONNX Runtime archive"
    fi

    install -m 0755 \
        "$extracted_directory/$archive_name/lib/$versioned_library" \
        "$BIN_DIR/$versioned_library"
    ln -sfn "$versioned_library" "$BIN_DIR/libonnxruntime.so.1"
    ln -sfn "$versioned_library" "$BIN_DIR/libonnxruntime.so"
    printf 'install.sh: installed %s beside %s\n' "$versioned_library" "$DAEMON_BINARY" >&2
)

install_launchd_service() {
    local installed_path="$1"
    local label="io.goodkind.lm-semantic-search-daemon"
    local plist_path="$HOME/Library/LaunchAgents/$label.plist"
    local log_path="$HOME/Library/Logs/lm-semantic-search-daemon.log"
    local domain
    local tmp_plist

    domain="gui/$(id -u)"
    mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
    touch "$log_path"
    tmp_plist="$(mktemp -t "$label.plist.XXXXXX")" || install_error "could not create launchd temp plist"
    cat >"$tmp_plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$installed_path</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$log_path</string>
    <key>StandardErrorPath</key>
    <string>$log_path</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>$HOME</string>
    </dict>
</dict>
</plist>
PLIST

    if [[ -f "$plist_path" ]] && cmp -s "$tmp_plist" "$plist_path" && launchctl print "$domain/$label" >/dev/null 2>&1; then
        rm -f "$tmp_plist"
        printf 'install.sh: service %s unchanged and loaded\n' "$label" >&2
        return 0
    fi

    mv "$tmp_plist" "$plist_path"
    launchctl bootout "$domain" "$plist_path" 2>/dev/null || true
    launchctl bootstrap "$domain" "$plist_path"
    printf 'install.sh: installed service %s\n' "$plist_path" >&2
}

install_systemd_service() {
    local installed_path="$1"
    local unit_name="lm-semantic-search-daemon.service"
    local user_dir="$HOME/.config/systemd/user"
    local unit_path="$user_dir/$unit_name"
    local tmp_unit

    need systemctl
    mkdir -p "$user_dir"
    tmp_unit="$(mktemp -t "$unit_name.XXXXXX")" || install_error "could not create systemd temp unit"
    cat >"$tmp_unit" <<UNIT
[Unit]
Description=lm-semantic-search daemon
Documentation=https://github.com/agoodkind/lm-semantic-search
After=network.target

[Service]
ExecStart=$installed_path
Restart=always
RestartSec=2
Environment=HOME=$HOME

[Install]
WantedBy=default.target
UNIT

    if [[ -f "$unit_path" ]] && cmp -s "$tmp_unit" "$unit_path" && systemctl --user is-active "$unit_name" >/dev/null 2>&1; then
        rm -f "$tmp_unit"
        printf 'install.sh: service %s unchanged and active\n' "$unit_name" >&2
        return 0
    fi

    mv "$tmp_unit" "$unit_path"
    systemctl --user daemon-reload
    systemctl --user enable "$unit_name"
    systemctl --user restart "$unit_name"
    printf 'install.sh: installed service %s\n' "$unit_path" >&2
}

install_daemon_service() {
    local installed_path="$1"

    case "$(uname -s)" in
        Darwin)
            install_launchd_service "$installed_path"
            ;;
        Linux)
            install_systemd_service "$installed_path"
            ;;
        *)
            usage_error "unsupported OS: $(uname -s)"
            ;;
    esac
}

main() {
    local daemon_path

    parse_args "$@"
    install_release_binary "$DAEMON_BINARY"
    install_release_binary "$CLI_BINARY"
    install_release_binary "$MCP_BINARY"
    if [[ "$(uname -s)" == "Linux" ]]; then
        install_linux_onnxruntime
    fi

    if [[ "$INSTALL_SERVICE" -eq 0 ]]; then
        printf 'install.sh: service setup skipped\n' >&2
        return 0
    fi

    daemon_path="$BIN_DIR/$DAEMON_BINARY"
    install_daemon_service "$daemon_path"
    printf 'install.sh: done\n' >&2
}

main "$@"
