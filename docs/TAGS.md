# 🏷️ Tag Taxonomy & Labeling Guide

This document defines the tag vocabulary for the ml-event-tagger MVP and provides guidelines for consistent labeling.

**Version:** 1.0
**Last updated:** October 17, 2025

---

## 🎯 Purpose

A controlled vocabulary of 15-20 tags covering common event characteristics in the Oakland/Bay Area event scene (primarily from CMF data).

**Key principles:**

-   **Simple:** Easy to apply consistently
-   **Practical:** Based on actual events
-   **Multi-label:** Events can have multiple tags (2-5 typical)
-   **Demonstrative:** For MVP, not production-perfect taxonomy

---

## 📋 Complete Tag List

### Music Genres (6 tags)

| Tag         | Definition                      | Example Events                       |
| ----------- | ------------------------------- | ------------------------------------ |
| `music`     | General music event (any genre) | Concerts, DJ sets, jam sessions      |
| `house`     | House music specifically        | "Days Like This", "ELEMENTS"         |
| `techno`    | Techno music                    | Underground raves, techno parties    |
| `jazz`      | Jazz music                      | Jazz clubs, jazz festivals           |
| `classical` | Classical/orchestral music      | Symphony performances, chamber music |
| `rock`      | Rock music                      | Rock concerts, indie shows           |

**Labeling tip:** Use both `music` + specific genre. E.g., "Days Like This - House Music" → [`music`, `house`]

---

### Activities (5 tags)

| Tag      | Definition                  | Example Events                                 |
| -------- | --------------------------- | ---------------------------------------------- |
| `dance`  | Dancing is primary activity | Dance parties, nightclubs, dance classes       |
| `yoga`   | Yoga or meditation event    | Yoga classes, outdoor yoga, meditation circles |
| `art`    | Art-focused event           | Gallery openings, art markets, exhibitions     |
| `food`   | Food is primary focus       | Food festivals, farmers markets, tastings      |
| `market` | Market or bazaar            | Farmers markets, craft fairs, flea markets     |

**Labeling tip:** Can combine with other tags. E.g., "Art Market" → [`art`, `market`]

---

### Locations (3 tags)

| Tag        | Definition             | Example Events           |
| ---------- | ---------------------- | ------------------------ |
| `oakland`  | Event in Oakland       | Any Oakland-based event  |
| `sf`       | Event in San Francisco | Any SF-based event       |
| `berkeley` | Event in Berkeley      | Any Berkeley-based event |

**Labeling tip:** Use formatted_address field to determine. If address contains "Oakland, CA" → [`oakland`]

---

### Characteristics (4 tags)

| Tag         | Definition                  | Example Events                              |
| ----------- | --------------------------- | ------------------------------------------- |
| `outdoor`   | Outdoor/open-air event      | Park events, outdoor concerts, street fairs |
| `weekly`    | Recurring weekly event      | "Days Like This" (every Friday)             |
| `community` | Community-focused gathering | Neighborhood events, community meetings     |
| `family`    | Family-friendly event       | Kids' events, all-ages shows                |

**Labeling tip:** These describe event properties. E.g., an outdoor weekly house music event → [`music`, `house`, `outdoor`, `weekly`, `oakland`]

---

## 📝 Labeling Guidelines

### How Many Tags?

-   **Minimum:** 1 tag (rare)
-   **Typical:** 2-5 tags
-   **Maximum:** 7 tags (avoid over-tagging)

**Examples:**

-   "Days Like This - House Music" → [`music`, `house`, `dance`, `oakland`, `weekly`] (5 tags ✅)
-   "SF Jazz Festival" → [`music`, `jazz`, `sf`] (3 tags ✅)
-   "Lake Merritt Farmers Market" → [`food`, `market`, `oakland`, `outdoor`] (4 tags ✅)

### Tag Selection Process

1. **Read name, description, and location** fully
2. **Identify primary category** (music/activity type)
3. **Add genre/specific activity** if applicable
4. **Add location** (oakland/sf/berkeley)
5. **Add characteristics** (outdoor/weekly/etc.) if clear
6. **Review:** Do 2-5 tags capture the essence? ✅

### Edge Cases

**Music event with unclear genre:**

-   Use `music` only, skip genre-specific tag
-   E.g., "Live music night" → [`music`, `oakland`]

**Event in multiple categories:**

-   Use both! E.g., "Yoga + Live Music" → [`yoga`, `music`]

**Recurring events:**

-   Only add `weekly` if explicitly stated or clearly recurring
-   Not needed for "Days Like This" mentions (already known to be weekly)

**Location unclear:**

-   If address is missing or unclear, skip location tag
-   Don't guess

**Family-friendly:**

-   Only add `family` if explicitly mentioned or all-ages
-   Don't assume

---

## 🎨 Example Annotations

### Example 1: "Days Like This - House Music"

**Event data:**

```json
{
    "name": "Days Like This - House Music",
    "description": "Weekly house music gathering with local DJs",
    "formatted_address": "599 El Embarcadero, Oakland, CA 94610"
}
```

**Tags:** [`music`, `house`, `dance`, `oakland`, `weekly`]

**Rationale:**

-   `music` - Music event
-   `house` - House music genre
-   `dance` - Dancing is primary activity
-   `oakland` - Location in Oakland
-   `weekly` - "Weekly" in description

---

### Example 2: "ELEMENTS House Music"

**Event data:**

```json
{
    "name": "ELEMENTS House Music?",
    "description": "ELEMENTS is a monthly dance ritual centered around deep & soulful house music in Oakland",
    "formatted_address": ""
}
```

**Tags:** [`music`, `house`, `dance`, `oakland`, `community`]

**Rationale:**

-   `music` - Music event
-   `house` - House music explicitly stated
-   `dance` - "dance ritual"
-   `oakland` - Stated in description (address missing)
-   `community` - "ritual" suggests community gathering
-   NOT `weekly` - It's monthly, not weekly

---

### Example 3: "House+ Skates Second Sunday"

**Event data:**

```json
{
    "name": "House+ Skates Second Sunday",
    "description": "DJ's Play House and techno",
    "formatted_address": "288 9th Ave, Oakland, CA 94606"
}
```

**Tags:** [`music`, `house`, `techno`, `oakland`, `outdoor`]

**Rationale:**

-   `music` - DJs playing music
-   `house` - House mentioned
-   `techno` - Techno mentioned
-   `oakland` - Oakland address
-   `outdoor` - Brooklyn Basin is outdoor area
-   NOT `weekly` - Says "Second Sunday" (monthly)

---

### Example 4: "SF Jazz Festival"

**Event data:**

```json
{
    "name": "San Francisco Jazz Festival",
    "description": "Annual jazz festival featuring local and international artists",
    "formatted_address": "San Francisco, CA"
}
```

**Tags:** [`music`, `jazz`, `sf`, `outdoor`]

**Rationale:**

-   `music` - Music event
-   `jazz` - Jazz festival
-   `sf` - San Francisco
-   `outdoor` - Typical for festivals (could verify from description)
-   NOT `weekly` - It's annual

---

### Example 5: "Lake Merritt Farmers Market"

**Event data:**

```json
{
    "name": "Lake Merritt Farmers Market",
    "description": "Fresh produce, artisan goods, and local food vendors",
    "formatted_address": "Lake Merritt, Oakland, CA"
}
```

**Tags:** [`food`, `market`, `oakland`, `outdoor`, `weekly`]

**Rationale:**

-   `food` - Food vendors
-   `market` - Farmers market
-   `oakland` - Oakland location
-   `outdoor` - Outdoor market
-   `weekly` - Most farmers markets are weekly (verify if stated)

---

## ⚖️ Quality Control

### Before Finalizing Labels

Ask yourself:

1. ✅ **Are the tags accurate?** Double-check against event details
2. ✅ **Did I use 2-5 tags?** Not too few, not too many
3. ✅ **Is the primary category covered?** (music/activity)
4. ✅ **Is location included if known?** (oakland/sf/berkeley)
5. ✅ **Are characteristics appropriate?** (outdoor/weekly/community/family)
6. ✅ **Would someone else agree?** Consistency matters

### Common Mistakes to Avoid

❌ **Over-tagging:** Don't add every possible tag

-   Bad: [`music`, `house`, `techno`, `dance`, `oakland`, `outdoor`, `weekly`, `community`, `family`] (9 tags)
-   Good: [`music`, `house`, `oakland`, `weekly`] (4 tags)

❌ **Under-tagging:** Don't be too minimal

-   Bad: [`music`] (only 1 tag)
-   Good: [`music`, `house`, `oakland`] (3 tags)

❌ **Guessing location:** Don't add location tags if unclear

-   If address is empty or vague, skip location tag

❌ **Inconsistent genre use:** Always use `music` + genre

-   Bad: [`house`] (missing music tag)
-   Good: [`music`, `house`] (both tags)

---

## 🔄 Tag Evolution

### MVP (v0.1)

-   18 tags total
-   Focused on Oakland/Bay Area events
-   Music-heavy (reflects CMF data)

### Future (v0.2+)

-   May add more genres (electronic, hip-hop, folk, etc.)
-   May add more activities (sports, networking, education)
-   May add more locations (other Bay Area cities)
-   May split some tags (e.g., `electronic` parent of `house`/`techno`)

### Requesting New Tags

If you encounter events that don't fit well:

1. Note the event details
2. Propose new tag with definition
3. Provide 3+ examples
4. Review with maintainer

---

## 📊 Expected Tag Distribution

Based on CMF event data, expected tag frequency:

**High frequency (30%+ of events):**

-   `music`, `oakland`

**Medium frequency (10-30%):**

-   `house`, `dance`, `outdoor`, `weekly`

**Low frequency (5-10%):**

-   `techno`, `jazz`, `sf`, `community`

**Rare (<5%):**

-   `classical`, `rock`, `yoga`, `art`, `food`, `market`, `berkeley`, `family`

**Note:** This distribution will affect model performance. Rare tags will be harder to predict accurately.

---

## 🧩 Tools & Workflow

### Recommended Labeling Workflow

1. Open `data/events-raw-fb.json` (reference data)
2. Create/edit `data/labeled_events.json`
3. For each event:
    - Read all fields carefully
    - Apply tags following guidelines
    - Add event to labeled dataset
4. Periodically check tag distribution
5. Ensure minimum 5 examples per tag

### Validation Checklist

After labeling batch of events:

```python
import json
from collections import Counter

# Load labeled events
with open('data/labeled_events.json') as f:
    events = json.load(f)

# Check counts
print(f"Total events: {len(events)}")

# Tag distribution
all_tags = [tag for event in events for tag in event['tags']]
print("\nTag distribution:")
for tag, count in Counter(all_tags).most_common():
    print(f"  {tag}: {count}")

# Average tags per event
avg_tags = len(all_tags) / len(events)
print(f"\nAverage tags per event: {avg_tags:.1f}")
```

**Healthy ranges:**

-   Average tags per event: 2.5-4.5
-   No tag with <5 occurrences
-   No tag with >70% occurrence

---

## 📚 Related Docs

-   [MVP_DECISIONS.md](./MVP_DECISIONS.md) - Why these tags were chosen
-   [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Phase 2: Data Preparation
-   [ARCHITECTURE.md](./ARCHITECTURE.md) - How tags are used in the model

---

## 🤝 Contributing

This is an MVP tag taxonomy. Feedback welcome!

If you have suggestions:

-   Open an issue with proposed changes
-   Include examples of events that don't fit
-   Consider backward compatibility with existing labels
