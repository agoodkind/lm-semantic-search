# `make help` is the canonical source of truth for every target this repo
# supports. Run it before adding anything new. Lint, build, test, deadcode,
# release, baseline, and service-install all live in the central go-makefile
# pipeline fetched at parse time. Do NOT add project-local lint, deadcode,
# audit, fmt, vet, or staticcheck targets here. They duplicate the central
# pipeline and let agents bypass strict rules.

# Identity. This repo has no own version package; it cross-stamps gklog/version.
BINARY := lm-semantic-search-daemon
CMD    := ./cmd/lm-semantic-search-daemon
GKLOG_VPKG := goodkind.io/gklog/version
CLI_BINARY := lm-semantic-search
CLI_CMD := ./cmd/$(CLI_BINARY)
MCP_BINARY := lm-semantic-search-mcp
MCP_CMD := ./cmd/$(MCP_BINARY)

# make install builds and installs the daemon plus both client binaries.
INSTALL_BINS := $(BINARY):$(CMD) $(CLI_BINARY):$(CLI_CMD) $(MCP_BINARY):$(MCP_CMD)

# Pipeline modules. Add go-service.mk if this binary ships as a daemon and
# set LAUNCHD_LABEL, SYSTEMD_UNIT, LOG_PATH before -include $(GO_MK).
GO_MK_MODULES := go-build.mk go-release.mk go-service.mk
BUILD_CHECKS := true
STATICCHECK_EXTRA_FLAGS = $(STATICCHECK_EXTRA_CORE_FLAGS) $(STATICCHECK_EXTRA_STRICT_FLAGS)

LAUNCHD_LABEL := io.goodkind.lm-semantic-search-daemon
SYSTEMD_UNIT := lm-semantic-search-daemon.service
# macOS launchd captures the daemon's stderr to this file; Linux logs to journald
# (the systemd unit sets no file path), so LOG_PATH there is only a harmless default.
ifeq ($(shell uname),Darwin)
LOG_PATH := $(HOME)/Library/Logs/lm-semantic-search-daemon.log
else
LOG_PATH := $(HOME)/.lm-semantic-search/logs/lm-semantic-search-daemon.log
endif

# go.mk runs these as order-only prerequisites of every build, lint, vet, test,
# and govulncheck target (and install/release via the modules). GO_MK_GENERATE
# generates the tree-sitter parser; GO_MK_WORKSPACE_USE materializes a gitignored
# go.work that routes the gksyntax submodule into the build, so both exist before
# any target compiles the grammar packages.
GO_MK_GENERATE := gksyntax-grammars
GO_MK_GENERATE_INPUTS := third_party/gksyntax
GO_MK_GENERATE_OUTPUTS := \
	third_party/gksyntax/treesitter/grammars/swift/upstream/src/parser.c \
	third_party/gksyntax/treesitter/grammars/swift/upstream/src/tree_sitter/parser.h \
	third_party/gksyntax/treesitter/grammars/swift/upstream/src/tree_sitter/array.h \
	third_party/gksyntax/treesitter/grammars/swift/upstream/src/tree_sitter/alloc.h
GO_MK_WORKSPACE_USE := . third_party/gksyntax

# bootstrap.mk fetches go.mk + golangci.yml + every module in GO_MK_MODULES
# at parse time and -includes them. Update path: edit go-makefile/bootstrap.mk,
# then refresh consumer copies (one-off cp; not enshrined as infrastructure).
include bootstrap.mk

.DEFAULT_GOAL := check

# ---------------------------------------------------------------------------
# Project-local
# ---------------------------------------------------------------------------

.PHONY: deploy deploy-service daemon-wait daemon-status kill-orphans

# daemon-status and daemon-wait call the installed CLI; kill-orphans matches the
# installed MCP binary by name.
CLI_INSTALL_BIN := $(INSTALL_DIR)/$(CLI_BINARY)

# ---------------------------------------------------------------------------
# gksyntax submodule grammars
# ---------------------------------------------------------------------------
# The AST splitter and tree-sitter grammars live in goodkind.io/gksyntax, a git
# submodule under third_party/ routed through a generated, gitignored go.work
# (GO_MK_WORKSPACE_USE above). A plain module require is not possible because
# gksyntax vendors the dart and swift grammars as its own submodules, whose C
# sources are absent from a Go module zip, and a go.mod replace is rejected by
# gomoddirectives. gksyntax commits only the swift grammar definition, not the
# generated parser, so the parser is produced from the pinned submodule by the
# tree-sitter CLI. The generated parser stays inside the submodule working tree
# (gitignored there) and is never committed.
GKS_DIR := third_party/gksyntax
SWIFT_GRAMMAR_DIR := $(GKS_DIR)/treesitter/grammars/swift/upstream
SWIFT_GRAMMAR_DEF := $(SWIFT_GRAMMAR_DIR)/src/grammar.json
SWIFT_GRAMMAR_PARSER := $(SWIFT_GRAMMAR_DIR)/src/parser.c
TREE_SITTER_ABI ?= 14
# tree-sitter CLI lands here when the host has none on PATH. Gitignored.
TREE_SITTER_LOCAL_DIR := $(CURDIR)/.bin

.PHONY: gksyntax-grammars
gksyntax-grammars:
	@git submodule update --init --recursive $(GKS_DIR)
	@if [ ! -f "$(SWIFT_GRAMMAR_DEF)" ]; then \
		echo "gksyntax-grammars: $(SWIFT_GRAMMAR_DIR) is empty; run 'git submodule update --init --recursive'"; \
		exit 1; \
	fi
	@ts_bin="$$(command -v tree-sitter 2>/dev/null || true)"; \
	if [ -z "$$ts_bin" ]; then \
		"$(GKS_DIR)/scripts/install-tree-sitter.sh" "$(TREE_SITTER_LOCAL_DIR)"; \
		ts_bin="$(TREE_SITTER_LOCAL_DIR)/tree-sitter"; \
	fi; \
	if [ ! -f "$(SWIFT_GRAMMAR_PARSER)" ] || [ "$(SWIFT_GRAMMAR_DEF)" -nt "$(SWIFT_GRAMMAR_PARSER)" ]; then \
		echo "gksyntax-grammars: generating Swift parser (abi $(TREE_SITTER_ABI))"; \
		( cd "$(SWIFT_GRAMMAR_DIR)" && "$$ts_bin" generate src/grammar.json --abi $(TREE_SITTER_ABI) ); \
		git -C "$(SWIFT_GRAMMAR_DIR)" checkout -- . >/dev/null 2>&1 || true; \
	else \
		echo "gksyntax-grammars: Swift parser already generated"; \
	fi

# The order-only prerequisite that runs gksyntax-grammars before every compile,
# vet, lint, test, install, and release target is wired centrally in go.mk via
# GO_MK_GENERATE (set above), so no per-target list is maintained here.

# install runs as the inherited `deploy: install` prerequisite from
# go-build.mk; repeating it here as a recipe line runs the whole
# build-check + codesign pipeline a second time in a fresh sub-make.
deploy:
	$(MAKE) deploy-service
	$(MAKE) daemon-wait
	$(MAKE) daemon-status

# Probe loadedness silently instead of letting service-restart fail with
# a make error on a host where the service is not installed yet.
deploy-service:
	@if [ "$$(uname)" = "Darwin" ]; then \
		if launchctl print "$(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL)" >/dev/null 2>&1; then \
			$(MAKE) service-restart; \
		else \
			echo "deploy-service: $(LAUNCHD_LABEL) not loaded; installing user service"; \
			launchctl enable "$(LAUNCHD_DOMAIN)/$(LAUNCHD_LABEL)" >/dev/null 2>&1 || true; \
			$(MAKE) service-install; \
		fi; \
	else \
		if systemctl --user cat "$(SYSTEMD_UNIT)" >/dev/null 2>&1; then \
			$(MAKE) service-restart; \
		else \
			echo "deploy-service: $(SYSTEMD_UNIT) not installed; installing user service"; \
			$(MAKE) service-install; \
		fi; \
	fi

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

# kill-orphans walks every running lm-semantic-search-mcp process and sends
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
