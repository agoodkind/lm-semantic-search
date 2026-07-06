#!/usr/bin/env bash
set -euo pipefail

REPO="agoodkind/lm-semantic-search"
HOSTED_INSTALLER_URL="https://raw.githubusercontent.com/agoodkind/go-makefile/main/install.sh"
DAEMON_BINARY="lm-semantic-search-daemon"
CLI_BINARY="lm-semantic-search"
MCP_BINARY="lm-semantic-search-mcp"

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

    if [[ "$INSTALL_SERVICE" -eq 0 ]]; then
        printf 'install.sh: service setup skipped\n' >&2
        return 0
    fi

    daemon_path="$BIN_DIR/$DAEMON_BINARY"
    install_daemon_service "$daemon_path"
    printf 'install.sh: done\n' >&2
}

main "$@"
