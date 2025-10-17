<!-- fde2fc75-50f0-4be2-ad6e-d8b44ab07e3b 3c7287b6-f219-40d7-92e7-c55aa2053f7a -->
# Documentation Improvements & MVP Planning

## Overview

Streamline project documentation to support a focused v0.1 MVP that demonstrates TensorFlow/Keras and full-stack integration skills. Emphasize clarity and reproducibility over complexity.

## Changes to Existing Docs

### 1. Fix ROADMAP.md

- Update cross-references to point correctly (./ARCHITECTURE.md not ../docs/)
- Remove completed checkboxes from v0.1 items
- Move authentication to v0.2
- Move monitoring/observability items to v0.2+
- Add clarification that v0.1 starts with 20 events for pipeline validation, then scales to 100

### 2. Update ARCHITECTURE.md

- Remove API contract JSON comment syntax errors
- Simplify API response format (less nesting)
- Add simple tag taxonomy section (15-20 tags)
- Add simplified training strategy section (70/15/15 split, basic metrics)
- Clarify which event fields to use: name + description + formatted_address
- Update folder structure to match organized package approach
- Remove overengineered sections (move caching, monitoring to "future improvements")

### 3. Update README.md

- Add clear "Skills Demonstrated" section
- Update repository structure to match organized package
- Add "Data & Labeling" section with tag list and simple guidelines
- Ensure quick start is accurate for the planned structure
- Add time estimate: "Clone to first prediction: ~10 minutes"

## New Planning Documents

### 4. Create docs/MVP_DECISIONS.md

Document all architectural and scope decisions made:

- Target: 100 labeled events, start with 20 for validation
- Tag taxonomy: 15-20 tags, list them with brief definitions
- Model: Simple Keras Sequential (embedding → pooling → dense)
- Data split: 70/15/15 (train/val/test)
- Event fields: name + description + formatted_address
- Success criteria: ≥60% precision, <300ms inference, working end-to-end
- What we're skipping for MVP (auth, monitoring, caching, etc.)
- Rationale: demonstrate understanding, not production readiness

### 5. Create docs/IMPLEMENTATION_PLAN.md

Step-by-step implementation checklist:

- Phase 1: Project setup (structure, requirements.txt, gitignore)
- Phase 2: Data preparation (define tags, label 20 events, validate)
- Phase 3: Preprocessing pipeline (text cleaning, tokenization)
- Phase 4: Model training (notebook + script, evaluation plots)
- Phase 5: API service (FastAPI, health check, predict endpoint)
- Phase 6: Testing & validation (manual tests, basic pytest)
- Phase 7: Docker & deployment prep
- Phase 8: Final documentation polish

Each phase with estimated time and success criteria.

### 6. Create docs/TAGS.md

Simple tag taxonomy documentation:

- List of 15-20 tags with one-line definitions
- Examples from actual events showing tag application
- Simple guidelines: "Use 2-5 tags per event", "Focus on primary characteristics"
- Note: This is for MVP demonstration, not production taxonomy

## File Updates Summary

**Modified:**

- docs/ROADMAP.md
- docs/ARCHITECTURE.md  
- README.md

**Created:**

- docs/MVP_DECISIONS.md
- docs/IMPLEMENTATION_PLAN.md
- docs/TAGS.md

## Success Criteria

- All cross-references work correctly
- Documentation hierarchy is clear and non-redundant
- MVP scope and decisions are explicitly documented
- Implementation path is clear and actionable
- Someone can understand the full plan in 15 minutes of reading

### To-dos

- [ ] Update ROADMAP.md: fix references, move items to appropriate versions, clarify v0.1 scope
- [ ] Revise ARCHITECTURE.md: simplify API contract, add tag taxonomy, clarify training strategy, update structure
- [ ] Enhance README.md: add skills section, update structure, add data/labeling overview
- [ ] Create docs/MVP_DECISIONS.md documenting all architectural choices and rationale
- [ ] Create docs/IMPLEMENTATION_PLAN.md with phased implementation checklist
- [ ] Create docs/TAGS.md with 15-20 tag taxonomy and simple guidelines