# LangChain.js Community Monorepo

This repository contains the community workspace for LangChain.js. It includes the main [`@langchain/community`](https://www.npmjs.com/package/@langchain/community) package plus several standalone provider packages built on top of `@langchain/core`.

The repo also includes internal tooling used to build, test, and validate packages consistently across the workspace.

> [!WARNING]
> **The community repo is being sunset.** See [#61](https://github.com/langchain-ai/langchain-community/issues/61) for details and guidance. Thank you to everyone who has contributed integrations, fixes, reviews, and maintenance over the years.

For end-user installation and package-specific usage examples, see the `README.md` file inside the relevant package under `libs/`.

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
pnpm --filter @langchain/nomic run build:compile
pnpm --filter @langchain/cerebras run test
```

You can replace the package name in `--filter` with any workspace package, such as `@langchain/yandex` or `@langchain/azure-cosmosdb`.

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

- formatting and lint checks across the workspace
- unit tests and standard unit tests for relevant packages
- scheduled or manually triggered standard integration tests for selected providers
- Changesets-based release PRs and npm publishing from `main`

If you are working on `@langchain/community`, the most relevant local verification loop is usually:

```bash
pnpm --filter @langchain/community run build:compile
pnpm --filter @langchain/community run test
pnpm --filter @langchain/community run lint
```

## Contributing

Changes are typically made in one of the package folders under `libs/`, with shared tooling and test infrastructure living under `internal/`.

Before opening a pull request:

```bash
pnpm format:check
pnpm lint
pnpm test:unit
```

If your change affects a specific provider package or standard test suite, use that package's scripts and CI workflows to validate the relevant build, unit tests, or integration tests.

## Related Docs

- Package docs: `libs/*/README.md`
- Build tooling docs: `internal/build/README.md`
- Standard test harness docs: `internal/standard-tests/README.md`
