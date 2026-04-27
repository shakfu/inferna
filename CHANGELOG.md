# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [Unreleased]

## [0.1.1]

### Fixed

- Fixed `AttributeError: 'LlamaContext' object has no attribute 'params'` in `inferna chat` by storing the originally-constructed `LlamaContextParams` on the chat object and reusing it when creating a fresh context per generation.
- Fixed sqlite-vector extension lookup in editable installs: `SqliteVectorStore.EXTENSION_PATH` now searches every entry in the `inferna.rag` package `__path__`, so the `vector.{dylib,so,dll}` artifact is found whether it lives next to the source tree or in the scikit-build-core editable mirror under site-packages.

## [0.1.0]

### Added

- Created inferna, a nanobind rewrite of [cyllama](https://github.com/shakfu/cyllama) v0.2.14.
