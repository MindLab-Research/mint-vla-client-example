rm /tmp/mint_dev_run.env

HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
python scripts/tools/gen_dev_placement.py --head-ip $HEAD_IP \
  --model Qwen/Qwen3.6-27B --gpu-count 4 \
  --output /tmp/mint_dev_run.env
scp /tmp/mint_dev_run.env driver:/tmp/mint_dev_run.env