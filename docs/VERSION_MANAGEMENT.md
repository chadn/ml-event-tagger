# 📌 Version Management Guide

Quick reference for managing versions in ml-event-tagger.

---

## 🎯 Current Version

**0.0.1** - Phase 1 Complete (Project Setup)

---

## 📍 Where Version is Tracked

### Primary Source of Truth

**`pyproject.toml`** (line 3)

```toml
[project]
name = "ml-event-tagger"
version = "0.0.1"  # ← ONLY place you update version
```

This is the **single source of truth** for the version number (DRY principle).

### How It Works

-   **`pyproject.toml`** - Authoritative source
-   **`ml_event_tagger/__init__.py`** - Reads version dynamically via `importlib.metadata`
-   **`CHANGELOG.md`** - Documents changes for each version (with dates)
-   **FastAPI `/health` endpoint** - Returns `__version__` at runtime
-   **FastAPI app initialization** - Shows version in auto-generated docs

**Code:**

```python
# ml_event_tagger/__init__.py
from importlib.metadata import version
__version__ = version("ml-event-tagger")  # reads from pyproject.toml
```

---

## 🔄 How to Bump Version

### Step 1: Decide Version Number

Following [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

0.0.4 → 0.0.5  (documentation update)
0.0.4 → 0.1.0  (MVP release with code)
0.1.0 → 0.2.0  (new features: data expansion)
0.2.0 → 0.2.1  (bug fix)
0.9.0 → 1.0.0  (production ready)
```

**Our versioning:**

-   **0.0.x** - Development phase (bump patch for each non-trivial commit/feature)
-   **0.1.0** - MVP complete (all 8 phases done, working model and API)
-   **0.x.0** - Feature additions (v0.2, v0.3, etc. per ROADMAP)
-   **1.0.0** - Production ready

**Commit strategy:**

-   Bump patch (0.0.1 → 0.0.2) for each completed phase or significant feature
-   Bump minor (0.0.x → 0.1.0) when MVP is complete
-   Document each version in CHANGELOG with date

### Step 2: Update `pyproject.toml`

```toml
# pyproject.toml
[project]
version = "0.1.0"  # Update this ONE place only
```

### Step 3: Update CHANGELOG.md

Add new section at the top:

```markdown
## [0.1.0] - 2025-10-18

### Added

-   TensorFlow/Keras model training pipeline
-   FastAPI service with /predict and /health endpoints
-   Training notebook with evaluation plots

### Changed

-   (list changes)

### Fixed

-   (list fixes)
```

### Step 4: Commit and Tag

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.0"
git tag v0.1.0
git push origin main --tags
```

---

## 📊 Version in API

The version is automatically included in:

### 1. FastAPI App Initialization

```python
from ml_event_tagger import __version__

app = FastAPI(title="ML Event Tagger", version=__version__)
```

This shows up in:

-   Auto-generated `/docs` (Swagger UI)
-   Auto-generated `/redoc` (ReDoc)
-   OpenAPI schema at `/openapi.json`

### 2. Health Check Endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": __version__  # Returns current version
    }
```

**Example response:**

```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "0.1.0"
}
```

---

## 🗓️ Version History

**See:** [CHANGELOG.md](../CHANGELOG.md) for complete version history and changes.

**Current progress:** [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) overview table.

---

## 🆚 Comparison with Other Projects

### Next.js (what you're familiar with)

```json
// package.json
{
    "name": "my-app",
    "version": "0.1.0" // ← Source of truth
}
```

### Python (this project)

```toml
# pyproject.toml
[project]
version = "0.0.1"  # ← Source of truth (like package.json)
```

```python
# ml_event_tagger/__init__.py (reads dynamically)
from importlib.metadata import version
__version__ = version("ml-event-tagger")
```

**Both approaches:**

-   Single source of truth for version (package.json / pyproject.toml)
-   Updated manually when releasing
-   Accessible programmatically
-   Displayed in API/app
-   Follow same DRY principle

---

## 🔍 Checking Version

### In Code

```python
from ml_event_tagger import __version__
print(f"Running version {__version__}")
```

### Via API

```bash
curl http://localhost:8000/health
```

### In Python Shell

```python
import ml_event_tagger
print(ml_event_tagger.__version__)
```

---

## 📝 Best Practices

1. **Update ONE place only:** `pyproject.toml` version (DRY principle)
2. **Always update CHANGELOG.md** when bumping version
3. **Use git tags** for releases: `git tag v0.1.0`
4. **Follow semantic versioning** consistently
5. **Document breaking changes** clearly in CHANGELOG
6. **Test the version** appears correctly in `/health` endpoint

---

## 🚀 Release Checklist

When releasing a new version:

-   [ ] Update `pyproject.toml` with new version (ONLY place to edit)
-   [ ] Update `CHANGELOG.md` with changes
-   [ ] Test locally (version appears in `/health`)
-   [ ] Commit: `git commit -m "chore: bump version to X.Y.Z"`
-   [ ] Tag: `git tag vX.Y.Z` (optional)
-   [ ] Push: `git push origin main --tags`
-   [ ] Deploy to production (if applicable)
-   [ ] Verify version in deployed `/health` endpoint

---

## 📚 Related Docs

-   [CHANGELOG.md](../CHANGELOG.md) - Full version history
-   [ROADMAP.md](./ROADMAP.md) - Future versions planned
-   [Semantic Versioning](https://semver.org/) - Official SemVer spec
-   [Keep a Changelog](https://keepachangelog.com/) - CHANGELOG format
