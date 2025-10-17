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

### Music Genres & Performers (8 tags)

| Tag      | Definition                      | Example Events                       |
| -------- | ------------------------------- | ------------------------------------ |
| `music`  | General music event (any genre) | Concerts, DJ sets, jam sessions      |
| `house`  | House music specifically        | "Days Like This", "ELEMENTS"         |
| `techno` | Techno music                    | Underground raves, techno parties    |
| `breaks` | Breaks/breakbeat music          | Breakbeat parties, jungle nights     |
| `jazz`   | Jazz music                      | Jazz clubs, jazz festivals           |
| `rock`   | Rock music                      | Rock concerts, indie shows           |
| `punk`   | Punk music                      | Punk shows, hardcore concerts        |
| `hiphop` | Hip-hop music                   | Hip-hop shows, rap concerts, cyphers |
| `dj`     | DJ performance (any genre)      | DJ sets, DJ nights                   |
| `band`   | Live band performance           | Band concerts, live music with bands |

**Labeling tip:** Use both `music` + specific genre + performer type. E.g., "DJ spinning house music" → [`music`, `house`, `dj`]

---

### Activities (4 tags)

| Tag     | Definition                  | Example Events                                 |
| ------- | --------------------------- | ---------------------------------------------- |
| `dance` | Dancing is primary activity | Dance parties, nightclubs, dance classes       |
| `yoga`  | Yoga or meditation event    | Yoga classes, outdoor yoga, meditation circles |
| `art`   | Art-focused event           | Gallery openings, art exhibitions              |
| `food`  | Food is primary focus       | Food festivals, tastings, food trucks          |

**Labeling tip:** Can combine with other tags. E.g., "Dance party with DJ" → [`dance`, `music`, `dj`]

---

### Access & Venue (5 tags)

| Tag       | Definition            | Example Events                 |
| --------- | --------------------- | ------------------------------ |
| `outdoor` | Outdoor/open-air      | Park events, outdoor concerts  |
| `indoor`  | Indoor venue          | Club events, indoor concerts   |
| `public`  | Open to public        | Public events, open gatherings |
| `private` | Private/invite-only   | Private parties, member events |
| `free`    | Free admission/no fee | Free concerts, no-cover events |

**Labeling tip:** Location characteristics describe the venue. E.g., a free outdoor concert → [`music`, `outdoor`, `free`, `public`]

---

### Other Characteristics (2 tags)

| Tag         | Definition                  | Example Events                          |
| ----------- | --------------------------- | --------------------------------------- |
| `weekly`    | Recurring weekly event      | "Days Like This" (every Friday)         |
| `community` | Community-focused gathering | Neighborhood events, community meetings |

**Labeling tip:** These describe event properties. E.g., a weekly outdoor house music event → [`music`, `house`, `outdoor`, `weekly`]

---

## 📝 Labeling Guidelines

### How Many Tags?

-   **Minimum:** 1 tag (rare)
-   **Typical:** 2-5 tags
-   **Maximum:** 7 tags (avoid over-tagging)

**Examples:**

-   "Days Like This - House Music" → [`music`, `house`, `dj`, `dance`, `outdoor`, `weekly`] (6 tags ✅)
-   "Jazz Festival" → [`music`, `jazz`, `band`, `outdoor`] (4 tags ✅)
-   "Food Vendors" → [`food`, `outdoor`, `public`] (3 tags ✅)

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
    "location": "The Pergola at Lake Merritt, 599 El Embarcadero, Oakland, CA 94610"
}
```

**Tags:** [`music`, `house`, `dj`, `dance`, `outdoor`, `weekly`]

**Rationale:**

-   `music` - Music event
-   `house` - House music genre
-   `dj` - DJs performing
-   `dance` - Dancing is primary activity
-   `outdoor` - Pergola at Lake Merritt is outdoor
-   `weekly` - "Weekly" in description

---

### Example 2: "ELEMENTS House Music"

**Event data:**

```json
{
    "name": "ELEMENTS House Music",
    "description": "Monthly dance ritual centered around deep & soulful house music",
    "location": "Oakland, CA"
}
```

**Tags:** [`music`, `house`, `dance`, `dj`, `community`]

**Rationale:**

-   `music` - Music event
-   `house` - House music explicitly stated
-   `dance` - "dance ritual"
-   `dj` - DJ-driven event
-   `community` - "ritual" suggests community gathering
-   NOT `weekly` - It's monthly, not weekly

---

### Example 3: "House+ Skates Second Sunday"

**Event data:**

```json
{
    "name": "House+ Skates Second Sunday",
    "description": "DJs play house and techno music",
    "location": "288 9th Ave, Oakland, CA 94606"
}
```

**Tags:** [`music`, `house`, `techno`, `dj`, `outdoor`, `free`]

**Rationale:**

-   `music` - DJs playing music
-   `house` - House mentioned
-   `techno` - Techno mentioned
-   `dj` - DJ performance
-   `outdoor` - Brooklyn Basin is outdoor area
-   `free` - Typical for this event
-   NOT `weekly` - Says "Second Sunday" (monthly)

---

### Example 4: "SF Jazz Festival"

**Event data:**

```json
{
    "name": "Jazz Festival",
    "description": "Annual jazz festival featuring local and international artists",
    "location": "San Francisco, CA"
}
```

**Tags:** [`music`, `jazz`, `band`, `outdoor`, `public`]

**Rationale:**

-   `music` - Music event
-   `jazz` - Jazz festival
-   `band` - Live band performances
-   `outdoor` - Typical for festivals
-   `public` - Open to public
-   NOT `weekly` - It's annual

---

### Example 5: "Lake Merritt Farmers Market"

**Event data:**

```json
{
    "name": "Lake Merritt Food Vendors",
    "description": "Fresh produce, artisan goods, and local food vendors",
    "location": "Lake Merritt, Oakland, CA"
}
```

**Tags:** [`food`, `outdoor`, `weekly`, `public`]

**Rationale:**

-   `food` - Food vendors
-   `outdoor` - Outdoor event
-   `weekly` - Most farmers markets are weekly
-   `public` - Open to public

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

-   Bad: [`music`, `house`, `techno`, `dance`, `dj`, `band`, `outdoor`, `indoor`, `weekly`, `community`] (10 tags)
-   Good: [`music`, `house`, `dj`, `outdoor`, `weekly`] (5 tags)

❌ **Under-tagging:** Don't be too minimal

-   Bad: [`music`] (only 1 tag)
-   Good: [`music`, `house`, `oakland`] (3 tags)

❌ **Guessing location:** Don't add location tags if unclear

-   If address is empty or vague, skip location tag

❌ **Inconsistent genre use:** Always use `music` + genre + performer type if applicable

-   Bad: [`house`] (missing music tag)
-   Good: [`music`, `house`, `dj`] (complete tags)

---

## 🔄 Tag Evolution

### MVP (v0.1)

-   21 tags total
-   Music-heavy (reflects CMF data from Oakland/Bay Area)
-   Added venue/access characteristics (outdoor, indoor, public, private, free)
-   Removed city-specific tags (Oakland, SF, Berkeley) - location info already in `location` field

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

-   `music`, `outdoor`, `public`

**Medium frequency (10-30%):**

-   `house`, `dance`, `dj`, `weekly`, `free`

**Low frequency (5-10%):**

-   `techno`, `jazz`, `community`, `band`, `indoor`

**Rare (<5%):**

-   `breaks`, `rock`, `punk`, `hiphop`, `yoga`, `art`, `food`, `private`

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
