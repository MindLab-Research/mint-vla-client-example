ssh -f -N -L 8000:localhost:30496 driver

MINT_BASE_URL=http://localhost:8000 \
TINKER_API_KEY=tml-dummy \
MINT_API_KEY=tml-dummy \
python scripts/tools/rl_check.py \
  --model Qwen/Qwen3.6-27B \
  --steps 10 \
  --group-size 4 \
  --timeout-s 600