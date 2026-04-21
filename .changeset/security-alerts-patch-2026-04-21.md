---
"@langchain/community": patch
---

chore(deps): bump transitive dependencies to patch 6 security alerts

Updates pnpm overrides to resolve critical and high severity Dependabot
alerts in transitive dependencies reached via optional peerDependencies
and dev tooling:

- `protobufjs` `^7.2.5` -> `^7.5.5` (CVE-2026-41242)
- `basic-ftp` `>=5.2.0` -> `^5.3.0` (CVE-2026-39983, GHSA-rp42-5vxx-qpwr, GHSA-6v7q-wjvx-w8wg)
- `vite` new override `^7.3.2` (CVE-2026-39363, CVE-2026-39364, plus CVE-2026-39365 as a side effect)

No workspace package has these as direct dependencies; overrides affect
the monorepo lockfile only and do not change published package contents.
