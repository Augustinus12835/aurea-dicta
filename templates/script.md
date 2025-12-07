# Script Template

## Metadata
**Video:** [N] - [Title]
**Duration:** [X] minutes ([Y] seconds)
**Target Word Count:** [Z] words max (2.5 words/second)
**Subject:** [Subject area]

---

## Script Format

Each frame should follow this exact format:

```markdown
## Frame N (M:SS-M:SS) • NN words

[Narration text here. Can be multiple sentences.
Markdown formatting allowed but will be cleaned for TTS.]

---
```

### Format Rules

1. **Frame Headers:**
   - Format: `## Frame N (M:SS-M:SS) • NN words`
   - Frame numbers start at 0
   - Use single digit minutes without padding: `0:15` not `00:15`
   - Word count uses bullet character `•`

2. **One Frame Per Section:**
   - Each frame must have exactly ONE narration section
   - Never combine frames: `## Frame 1-2` is WRONG
   - Always separate: `## Frame 1`, `## Frame 2`

3. **No Separate Closing:**
   - Integrate closing into the final frame
   - Don't add: `## Closing (3:45-4:00)`

4. **Word Count Formula:**
   - Target: 2.5 words per second
   - 15 seconds → ~38 words
   - 30 seconds → ~75 words
   - 60 seconds → ~150 words

5. **Separators:**
   - Use `---` between all frames
   - Leave blank line after header, before separator

---

## Example Script Structure

```markdown
## Frame 0 (0:00-0:15) • 38 words

Here we'll look at [topic]. This is a fundamental concept that
[brief importance]. The visual shows [what's on screen].

---

## Frame 1 (0:15-0:30) • 38 words

[Next concept or quote introduction]

---

## Frame 2 (0:30-0:45) • 38 words

[Explanation or example]

---

... continue for all frames ...

## Frame 11 (3:00-4:00) • 150 words

[Final frame with content + closing summary. Don't create
a separate "Closing" section - integrate it here.]

---

## Summary Statistics

**Total Word Count:** ~XXX words
**Target:** ZZZ words
**Average Speaking Rate:** 2.5 words/second
```

---

## Style Compliance Checklist

Before finalizing, verify:

**Word Economy:**
- [ ] NO filler words ("basically", "essentially", "you know")
- [ ] NO rhetorical questions ("right?", "make sense?")
- [ ] NO verbal cushioning ("let me just...", "so what I mean is...")
- [ ] NO patronizing language ("make sure...", "you need to understand...")
- [ ] Each concept stated ONCE, no unnecessary repetition

**Precision:**
- [ ] Direct statements, active voice throughout
- [ ] Technical terms defined briefly
- [ ] No redundant explanations

**Structure:**
- [ ] One narration per frame
- [ ] No separate closing section
- [ ] Frame format matches specification exactly
