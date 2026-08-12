# Release auto-update CI implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove published LMS releases automatically update on macOS 26 and Debian Trixie while local builds remain protected.

**Architecture:** Port the existing local-build classifier into update options. Add a Go release probe that installs the prior archive, starts its isolated offline daemon, and verifies the scheduler installs every current attested binary. Run that probe after the reusable release job on both supported host families.

**Tech Stack:** Go 1.26, GitHub Actions, system update scheduler, selfupdate attestations.

- [ ] Add failing local-build and option-wiring tests.
- [ ] Port the local-build classifier and update selfupdate.
- [ ] Add failing release-selection, archive, proxy, state, and version tests.
- [ ] Implement the Go post-release probe.
- [ ] Add macOS 26 and Debian Trixie release jobs with offline live coverage.
- [ ] Run focused tests, `make offline-live`, and `make check`.
- [ ] Complete adversarial review, open the pull request, and merge after checks pass.
