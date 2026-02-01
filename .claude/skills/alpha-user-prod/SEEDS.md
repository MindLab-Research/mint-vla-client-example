# alpha-user-prod seeds (self-contained)

These are seed references for bootstrapping the alpha-user workspace. They are not the boundary of what to attempt.

## Tinker Cookbook (git submodule)

- `.claude/skills/alpha-user-prod/tinker-cookbook/`
  - Official cookbook as a git submodule.
  - Use it to learn SDK call shapes, then move beyond it.
  - If the submodule is not present in a fresh checkout, initialize it with:
    - `git submodule update --init --recursive`

## Countdown environment (extracted)

- `.claude/skills/alpha-user-prod/seeds/countdown_env.py`
  - Countdown prompt + validation semantics (use each number exactly once; safe eval).
  - Uses a small built-in task list to avoid external dataset dependencies.

## Long RL seed (included)

- `.claude/skills/alpha-user-prod/seeds/mint_rl_test_long.py`
  - Minimal MinT RL demo (long prompts) exercising capabilities discovery, training client creation, sampling reload, and a simple reward.

## How to use seeds without becoming cookbook-bound

- Extract primitives and invariants, not task content:
  - Which SDK calls are used?
  - Which async/future ordering patterns are required?
  - Where do checkpoint paths appear and how are they reused?
- Change the user story and the failure modes you are trying to detect.
  - Example: keep the same call pattern but switch from arithmetic reward to "format adherence reward" or a builder-oriented agent scenario.
