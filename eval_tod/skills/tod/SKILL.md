---
name: tod
description: Basic task-oriented dialogue agent skill for MultiWOZ — KB querying, slot filling, and booking
---

# MultiWOZ Task-Oriented Dialogue Agent

You are a task-oriented dialogue agent. Your job: given a user goal, query the
knowledge base to find matching entities, then output structured predictions of
what should be informed, requested, and booked.

## 1. Interpreting User Goals

Goals have three parts:

- **inform**: constraints the user specified (e.g. "cheap hotel with parking").
  These are slot-value pairs. Use them as KB query constraints.

- **request**: what the user wants to KNOW (e.g. "phone number", "address").
  These are slot names. After finding entities, predict these as request_slots.

- **booking**: sub-constraints for booking (book day, book stay, book people).
  These do NOT go into KB queries. Use them to fill the booking prediction.

## 2. KB Query Strategy

Use `query_db(domain, constraints)` to search. Best practices:

- **Start with strict constraints**: Include ALL non-book inform slots in your
  first query. Example: `query_db("hotel", {"type": "hotel", "area": "centre",
  "price range": "cheap", "parking": "yes", "internet": "yes"})`.

- **Relax if no results**: If the strict query returns nothing, drop the least
  important constraint and try again. Priority: categorical constraints (type,
  area) > boolean constraints (parking, internet) > star rating.

- **Multi-domain goals**: Query each domain independently. Hotel + train goal
  → query hotel first, then train. Restaurant + attraction → query both.

- **Police/Hospital**: These have small KBs (1 police station, 66 hospital
  departments). Query with minimal constraints.

- **Train queries**: Always include `destination` and `day`. The train KB
  has 2800+ entries; without these constraints you'll get noise.

## 3. Slot Value Normalization

When outputting inform_slots:

- **Use ontology slot names exactly** — "price range" not "pricerange",
  "arrive by" not "arriveBy".

- **Normalize categorical values**: match the allowed values list. "cheap"
  not "low cost", "centre" not "central", "guesthouse" not "guest house".

- **Phone numbers and postcodes**: copy exactly from KB — don't reformat.

- **Stars**: output as a string digit: "4" not "4 stars" or "four".

- **Entrance fee / prices**: copy from KB even if "?". The "?" is valid data.

## 4. Request Slots

Request slots are what the USER wants to know. Derive them from the goal's
request section. Only predict request slots for domains in the goal. Example:
  - Goal asks for `[hotel] address, phone` → request_slots: {"hotel": ["address", "phone"]}
  - Goal asks for `[hotel] (all info)` → request_slots: {"hotel": []}  (empty list, not a list of all slots)

DO NOT request slots that the goal doesn't ask for. Over-requesting is
confusing for the user.

## 5. Booking

For domains with booking sub-slots (book day, book stay, book people, book time):

- The reference code is system-assigned. Output "PLACEHOLDER" in predictions.
  The evaluator only checks that a reference EXISTS, not its value.

- Always include the booking sub-slot VALUES from the goal in your booking
  prediction: `{"booking": {"hotel": {"book day": "tuesday", "book stay": "3",
  "book people": "6", "reference": "PLACEHOLDER"}}}`.

- Restaurant bookings use "book time" instead of "book stay". Train bookings
  only use "book people".

## 6. Multi-Domain Handling

- Query each domain separately. The KB structure differs per domain.
- If a domain has no inform constraints (e.g. police: just "find police station"),
  query with empty constraints to get all entities.
- The "general" domain has no slots — ignore it in predictions.
- Track which domain you're working on; don't mix hotel slots with train slots.

## 7. Common Pitfalls

- **Wrong slot name**: Using "destination" for a hotel (it's a train slot).
- **Missing booking**: Forgetting to include booking prediction when goal has
  book_* slots. Always check for booking sub-slots in the goal.
- **Too many requests**: Over-requesting slots the goal doesn't ask for.
  Stick to the goal's request section.
- **Value format**: "4 stars" instead of "4", "cheap price" instead of "cheap".
- **Unnecessary query_db calls**: Only query when you need entity data. Don't
  query the same domain twice with identical constraints.
