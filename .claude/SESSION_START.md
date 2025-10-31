# Session Start Guide for User

At the beginning of each new Claude Code session, paste this message to Claude:

```
Read `.claude/README.md` to understand the project context.
```

That's it! Claude will:
1. Read the README to understand what context files exist
2. Know where to look for specific information when needed
3. Not waste tokens reading everything upfront

## Why This Works

- **Efficient**: Claude only reads the index (~2k tokens) instead of all context files (~20k+ tokens)
- **Flexible**: Claude reads detailed files only when relevant to the current task
- **Maintainable**: Add new context files by updating the README index

## Example Session Start

```
User: Read `.claude/README.md` to understand the project context.

Claude: [reads README, responds with summary of what it found]

User: Let's implement the magnetics IDS mapper

Claude: [reads DEVELOPMENT_PRINCIPLES.md and .claudecontext for OMAS patterns,
         then proceeds with implementation]
```
