# Changelog

All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## 2025-10-17 0.0.1

### Implementation

-   Phase 1 complete: Project structure validated and dependencies installed
-   Package version tracking: `ml_event_tagger.__version__ = "0.0.1"`
-   Modern Python tooling: `pyproject.toml` with `uv` for fast dependency management
-   Configuration: 21 tags defined in `ml_event_tagger/config.py`

## 2025-10-17

### Changed

-   API field renamed `formatted_address` → `location`
-   Tag taxonomy: removed city tags (sf, oakland, berkeley), added venue/access tags (outdoor, indoor, public, private, free)
-   ROADMAP restructured: v0.2 (Data) and v0.3 (Performance & Security) now separate

### Added

-   Complete planning documentation: TAGS.md, IMPLEMENTATION_PLAN.md, MVP_DECISIONS.md
-   VERSION_MANAGEMENT.md for version tracking guidelines
-   This CHANGELOG

---

**Future releases:** See [docs/ROADMAP.md](docs/ROADMAP.md)
**Version management:** See [docs/VERSION_MANAGEMENT.md](docs/VERSION_MANAGEMENT.md)
