# LangChain.js Community Monorepo

This repository contains the community workspace for LangChain.js. Its main package is [`@langchain/community`](https://www.npmjs.com/package/@langchain/community), which provides third-party integrations built on top of `@langchain/core`.

The repo also includes internal tooling used to build and validate the package in a consistent way across the workspace.

## What's In This Repo

- `libs/community`: the `@langchain/community` package source, tests, and package-level documentation
- `internal/build`: shared build utilities built around `tsdown`
- `internal/tsconfig`: shared TypeScript configuration for workspace packages
- `.github/workflows`: CI for formatting, linting, unit tests, and scheduled standard integration tests

For end-user installation and package-specific usage examples, see `libs/community/README.md`.

## Requirements

- Node.js `>=20`
- `pnpm` `10.14.0`

The repository includes an `.nvmrc` file if you use `nvm`.

## Getting Started

```bash
nvm use
pnpm install
```

## Common Commands

Run these from the repository root:

```bash
pnpm build
pnpm lint
pnpm format:check
pnpm test:unit
```

Useful package-scoped commands:

```bash
pnpm --filter @langchain/community run build:compile
pnpm --filter @langchain/community run test
pnpm --filter @langchain/community run test:watch
pnpm --filter @langchain/community run test:standard:unit
pnpm --filter @langchain/community run format:check
```

## Workspace Tooling

This monorepo uses:

- `pnpm` for workspace and dependency management
- `turbo` for task orchestration
- `tsdown` for package builds
- `vitest` for tests
- `oxlint` and `dpdm` for linting
- `oxfmt` for formatting

## Testing And CI

GitHub Actions in this repo cover:

- formatting checks for `@langchain/community`
- lint checks for `@langchain/community`
- unit tests and standard unit tests
- scheduled or manually triggered standard integration tests for selected providers

If you are working on the main package, the most relevant local verification loop is usually:

```bash
pnpm --filter @langchain/community run build:compile
pnpm --filter @langchain/community run test
pnpm --filter @langchain/community run lint
```

## Contributing

Changes are typically made in `libs/community`, with shared tooling changes living under `internal/`.

Before opening a pull request:

```bash
pnpm format:check
pnpm lint
pnpm test:unit
```

If your change affects provider-specific standard tests, use the package-level scripts and CI workflows to validate the relevant suite.

## Related Docs

- Package docs: `libs/community/README.md`
- Build tooling docs: `internal/build/README.md`
