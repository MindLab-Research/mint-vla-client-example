"""Merge gate test suite.

Run all tests:
    MINT_BASE_URL=http://localhost:8000 python -m pytest .claude/skills/merge-gate/tests/ -v

Run specific phase:
    python -m pytest .claude/skills/merge-gate/tests/test_dense_*.py -v  # Dense only
    python -m pytest .claude/skills/merge-gate/tests/test_moe_*.py -v    # MoE only
    python -m pytest .claude/skills/merge-gate/tests/test_stress.py -v  # Stress only
"""
