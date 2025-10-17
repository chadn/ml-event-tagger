# 📌 Version Management Guide

Quick reference for managing versions in ml-event-tagger.

---

## 🎯 Current Version

**0.0.4** - Planning Phase

---

## 📍 Where Version is Tracked

### Primary Source of Truth

**`ml_event_tagger/__init__.py`**

```python
"""ML Event Tagger - Multi-label event classification service."""

__version__ = "0.0.4"
```

This is the **single source of truth** for the version number.

### Secondary Locations

-   `CHANGELOG.md` - Documents changes for each version
-   FastAPI `/health` endpoint - Returns version at runtime
-   FastAPI app initialization - Shows version in auto-generated docs

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

-   **0.0.x** - Planning and documentation (no code)
-   **0.1.0** - MVP with working model and API
-   **0.x.0** - Feature additions
-   **1.0.0** - Production ready

### Step 2: Update `__init__.py`

```python
# ml_event_tagger/__init__.py
__version__ = "0.1.0"  # Update this
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
git add ml_event_tagger/__init__.py CHANGELOG.md
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

| Version | Date       | Description                              | Status      |
| ------- | ---------- | ---------------------------------------- | ----------- |
| 0.0.1   | 2025-10-17 | Initial documentation                    | ✅ Released |
| 0.0.2   | 2025-10-17 | ROADMAP and tag taxonomy                 | ✅ Released |
| 0.0.3   | 2025-10-17 | Planning docs (MVP_DECISIONS, IMPL_PLAN) | ✅ Released |
| 0.0.4   | 2025-10-17 | Updated tags, restructured roadmap       | ✅ Released |
| 0.1.0   | TBD        | MVP with working code                    | ⏳ Planned  |
| 0.2.0   | TBD        | Data expansion                           | 🔜 Future   |
| 0.3.0   | TBD        | Performance & security                   | 🔜 Future   |
| 1.0.0   | TBD        | Production ready                         | 🧭 Future   |

---

## 🆚 Comparison with Other Projects

### Next.js (what you're familiar with)

```json
// package.json
{
    "name": "my-app",
    "version": "0.1.0"
}
```

### Python (this project)

```python
# ml_event_tagger/__init__.py
__version__ = "0.1.0"
```

**Both approaches:**

-   Single source of truth for version
-   Updated manually when releasing
-   Accessible programmatically
-   Displayed in API/app

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

1. **Always update both** `__init__.py` and `CHANGELOG.md` together
2. **Use git tags** for releases: `git tag v0.1.0`
3. **Follow semantic versioning** consistently
4. **Document breaking changes** clearly in CHANGELOG
5. **Test the version** appears correctly in `/health` endpoint

---

## 🚀 Release Checklist

When releasing a new version:

-   [ ] Update `ml_event_tagger/__init__.py` with new version
-   [ ] Update `CHANGELOG.md` with changes
-   [ ] Test locally (version appears in `/health`)
-   [ ] Commit: `git commit -m "chore: bump version to X.Y.Z"`
-   [ ] Tag: `git tag vX.Y.Z`
-   [ ] Push: `git push origin main --tags`
-   [ ] Deploy to production (if applicable)
-   [ ] Verify version in deployed `/health` endpoint

---

## 📚 Related Docs

-   [CHANGELOG.md](../CHANGELOG.md) - Full version history
-   [ROADMAP.md](./ROADMAP.md) - Future versions planned
-   [Semantic Versioning](https://semver.org/) - Official SemVer spec
-   [Keep a Changelog](https://keepachangelog.com/) - CHANGELOG format
