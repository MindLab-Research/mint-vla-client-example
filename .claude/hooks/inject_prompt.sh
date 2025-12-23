#!/bin/bash
# SessionStart hook: injects context reminder + optional PROMPT.md

BASE_CONTENT="# Session Start Reminder

**Before doing ANY work, read:**

1. **CLAUDE.md** - Project overview, architecture, Hall of Shame (past mistakes to avoid)
2. **Relevant skill** - Use exact commands, don't guess:
   - Server operations on dev → read \`.claude/skills/mint-dev/SKILL.md\`
   - Server operations on prod → read \`.claude/skills/mint-prod/SKILL.md\`
   - Ray cluster management → read \`.claude/skills/volcano-cluster/SKILL.md\`
   - Pre-merge testing → read \`.claude/skills/merge-gate/SKILL.md\`

**Do NOT:**
- Guess SSH hosts, log locations, or commands - they are documented
- Waste context on trial-and-error infrastructure debugging
- Run test scripts on server (run locally, server has no internet)"

# Append PROMPT.md if it exists (ephemeral, gitignored)
PROMPT_FILE="$CLAUDE_PROJECT_DIR/PROMPT.md"
if [ -f "$PROMPT_FILE" ]; then
  CONTENT="$BASE_CONTENT

---

$(cat "$PROMPT_FILE")"
else
  CONTENT="$BASE_CONTENT"
fi

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": $(echo "$CONTENT" | jq -R -s '.')
  }
}
EOF
