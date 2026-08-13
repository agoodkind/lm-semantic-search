# Release auto-update CI design

Status: approved
Date: 2026-08-12

Official release installs update automatically. Local builds remain protected from replacement.

The release workflow runs after publishing on macOS 26 arm64 and Debian Trixie amd64. Each job runs the existing offline live suite, installs the preceding release in an isolated directory, starts its real daemon with the offline profile, and waits for the scheduler to replace all three release binaries with the new attested release.

The probe uses Go. It selects releases through the GitHub API, runs the official installer for the preceding attested release, proxies update API requests with the workflow token, isolates all runtime paths, and reports daemon and updater diagnostics on failure.

Dirty, unstamped, `dev`, `unknown`, and git-describe-ahead builds count as local. Release and prerelease tags remain updateable. Update check, status, and apply commands remain present.
