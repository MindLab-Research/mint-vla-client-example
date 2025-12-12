#!/bin/bash
# SessionStart hook: injects PROMPT.md into context

PROMPT_FILE="$CLAUDE_PROJECT_DIR/PROMPT.md"

if [ -f "$PROMPT_FILE" ]; then
  CONTENT=$(cat "$PROMPT_FILE")
  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": $(echo "$CONTENT" | jq -R -s '.')
  }
}
EOF
fi
