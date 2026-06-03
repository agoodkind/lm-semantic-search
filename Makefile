# `make help` is the canonical source of truth for every target this repo
# supports. Run it before adding anything new. Lint, build, test, deadcode,
# release, baseline, and service-install all live in the central go-makefile
# pipeline fetched at parse time. Do NOT add project-local lint, deadcode,
# audit, fmt, vet, or staticcheck targets here. They duplicate the central
# pipeline and let agents bypass strict rules.

# Identity. This repo has no own version package; it cross-stamps gklog/version.
BINARY := claude-contextd
CMD    := ./cmd/claude-contextd
GKLOG_VPKG := goodkind.io/gklog/version
CLI_BINARY := claude-context
CLI_CMD := ./cmd/$(CLI_BINARY)
MCP_BINARY := claude-context-mcp
MCP_CMD := ./cmd/$(MCP_BINARY)

# Pipeline modules. Add go-service.mk if this binary ships as a daemon and
# set LAUNCHD_LABEL, SYSTEMD_UNIT, LOG_PATH before -include $(GO_MK).
GO_MK_MODULES := go-build.mk go-release.mk go-service.mk
BUILD_CHECKS := true
STATICCHECK_EXTRA_FLAGS = $(STATICCHECK_EXTRA_CORE_FLAGS) $(STATICCHECK_EXTRA_STRICT_FLAGS)

LAUNCHD_LABEL := io.goodkind.claude-contextd
SYSTEMD_UNIT := claude-contextd.service
# macOS launchd captures the daemon's stderr to this file; Linux logs to journald
# (the systemd unit sets no file path), so LOG_PATH there is only a harmless default.
ifeq ($(shell uname),Darwin)
LOG_PATH := $(HOME)/Library/Logs/claude-contextd.log
else
LOG_PATH := $(HOME)/.contextd/logs/claude-contextd.log
endif

# bootstrap.mk fetches go.mk + golangci.yml + every module in GO_MK_MODULES
# at parse time and -includes them. Update path: edit go-makefile/bootstrap.mk,
# then refresh consumer copies (one-off cp; not enshrined as infrastructure).
include bootstrap.mk

.DEFAULT_GOAL := check

# ---------------------------------------------------------------------------
# Project-local
# ---------------------------------------------------------------------------

.PHONY: build-clients install-clients deploy deploy-service daemon-wait daemon-status kill-orphans grammars

CLI_DIST_BIN := $(DIST_DIR)/$(CLI_BINARY)
MCP_DIST_BIN := $(DIST_DIR)/$(MCP_BINARY)
CLI_INSTALL_BIN := $(INSTALL_DIR)/$(CLI_BINARY)
MCP_INSTALL_BIN := $(INSTALL_DIR)/$(MCP_BINARY)

# ---------------------------------------------------------------------------
# Grammar generation
# ---------------------------------------------------------------------------
# The Swift grammar submodule commits only its grammar definition, not the
# generated parser, so the parser is produced from the pinned submodule by the
# tree-sitter CLI. The other grammars commit their parser and need no step. The
# generated files stay inside the submodule working tree (gitignored there) and
# are never committed to this repository.
SWIFT_GRAMMAR_DIR := internal/splitter/grammars/swift/upstream
SWIFT_GRAMMAR_DEF := $(SWIFT_GRAMMAR_DIR)/src/grammar.json
SWIFT_GRAMMAR_PARSER := $(SWIFT_GRAMMAR_DIR)/src/parser.c
TREE_SITTER_ABI ?= 14
# tree-sitter CLI lands here when the host has none on PATH, so a bare runner
# with only Go can still generate the Swift parser. Gitignored.
TREE_SITTER_LOCAL_DIR := $(CURDIR)/.bin

grammars:
	@if [ ! -f "$(SWIFT_GRAMMAR_DEF)" ]; then \
		echo "grammars: $(SWIFT_GRAMMAR_DIR) is empty; run 'git submodule update --init --recursive'"; \
		exit 1; \
	fi
	@ts_bin="$$(command -v tree-sitter 2>/dev/null || true)"; \
	if [ -z "$$ts_bin" ]; then \
		./scripts/install-tree-sitter.sh "$(TREE_SITTER_LOCAL_DIR)"; \
		ts_bin="$(TREE_SITTER_LOCAL_DIR)/tree-sitter"; \
	fi; \
	if [ ! -f "$(SWIFT_GRAMMAR_PARSER)" ] || [ "$(SWIFT_GRAMMAR_DEF)" -nt "$(SWIFT_GRAMMAR_PARSER)" ]; then \
		echo "grammars: generating Swift parser (abi $(TREE_SITTER_ABI))"; \
		( cd "$(SWIFT_GRAMMAR_DIR)" && "$$ts_bin" generate src/grammar.json --abi $(TREE_SITTER_ABI) ); \
		git -C "$(SWIFT_GRAMMAR_DIR)" checkout -- . >/dev/null 2>&1 || true; \
	else \
		echo "grammars: Swift parser already generated"; \
	fi

# tree-sitter generate also rewrites the upstream Go binding; the checkout above
# reverts those tracked edits so the submodule stays at its pinned commit. The
# generated parser.c and tree_sitter/ headers are gitignored in the submodule,
# so the checkout leaves them in place.

# Building, testing, vetting, linting, and govulncheck all compile the Swift
# grammar package, so they need the generated parser. The order-only
# prerequisite generates it first on a fresh checkout without forcing rebuilds.
build build-check check test lint vet govulncheck: | grammars

build-clients:
	@mkdir -p "$(DIST_DIR)"
	go build $(GO_BUILD_FLAGS) -o "$(CLI_DIST_BIN)" $(CLI_CMD)
	@echo "built: $(CLI_DIST_BIN)"
	$(call codesign_binary,$(CLI_DIST_BIN))
	go build $(GO_BUILD_FLAGS) -o "$(MCP_DIST_BIN)" $(MCP_CMD)
	@echo "built: $(MCP_DIST_BIN)"
	$(call codesign_binary,$(MCP_DIST_BIN))

install-clients: build-clients
	@printf 'install: installing %s to %s\n' '$(CLI_BINARY)' '$(CLI_INSTALL_BIN)'
	@mkdir -p "$(INSTALL_DIR)"
	@out="$$(mktemp "$(CLI_INSTALL_BIN)".new.XXXXXX)"; \
	trap 'rm -f "$$out"' EXIT; \
	cp -f "$(CLI_DIST_BIN)" "$$out"; \
	chmod 0755 "$$out"; \
	test -s "$$out"; \
	mv -f "$$out" "$(CLI_INSTALL_BIN)"
	@printf 'install: installing %s to %s\n' '$(MCP_BINARY)' '$(MCP_INSTALL_BIN)'
	@out="$$(mktemp "$(MCP_INSTALL_BIN)".new.XXXXXX)"; \
	trap 'rm -f "$$out"' EXIT; \
	cp -f "$(MCP_DIST_BIN)" "$$out"; \
	chmod 0755 "$$out"; \
	test -s "$$out"; \
	mv -f "$$out" "$(MCP_INSTALL_BIN)"

deploy:
	$(MAKE) install
	$(MAKE) install-clients
	$(MAKE) deploy-service
	$(MAKE) daemon-wait
	$(MAKE) daemon-status

deploy-service:
	@$(MAKE) service-restart || { \
		echo "service restart failed; installing user service"; \
		$(MAKE) service-install; \
		if [ "$$(uname)" = "Darwin" ]; then \
			launchctl enable "$(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL)" || true; \
		fi; \
		$(MAKE) service-restart; \
	}

daemon-status:
	"$(CLI_INSTALL_BIN)" daemon status

daemon-wait:
	@for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		if status_output="$$( "$(CLI_INSTALL_BIN)" daemon status )"; then \
			exit 0; \
		fi; \
		sleep 0.25; \
	done; \
	"$(CLI_INSTALL_BIN)" daemon status

# kill-orphans walks every running claude-context-mcp process and sends
# SIGKILL when its parent PID is 1 (init). Active sessions with a live parent
# stay untouched. This is the mitigation for the orphan-pile failure mode
# (199 zombies on the host pushing system load to 28) that bit the upstream
# TS adapter; the Go adapter avoids this by exiting on stdin EOF, PPID poll,
# and panic recovery, but this target stays as a paranoia cleanup.
kill-orphans:
	@killed=0; preserved=0; \
	for pid in $$(pgrep -x $(MCP_BINARY) || true); do \
		parent=$$(ps -o ppid= -p "$$pid" | tr -d ' '); \
		if [ "$$parent" = "1" ]; then \
			echo "kill-orphans: SIGKILL pid=$$pid (orphan, ppid=1)"; \
			kill -9 "$$pid" && killed=$$((killed + 1)); \
		else \
			parent_command=$$(ps -o comm= -p "$$parent" 2>&1 | head -n1); \
			echo "kill-orphans: preserve pid=$$pid (ppid=$$parent $$parent_command)"; \
			preserved=$$((preserved + 1)); \
		fi; \
	done; \
	echo "kill-orphans: killed $$killed orphan(s), preserved $$preserved live adapter(s)"
