ssh driver 'PY=/vePFS-Mindverse/share/code/wenxi/host-venv/bin/python3.13
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
$PY -c "
import ray
ray.init(address=\"$HEAD_IP:6379\", namespace=\"mint_wenxi_dev\", ignore_reinit_error=True, log_to_driver=False)
for name in [\"mint_config\", \"mint_task_state_store\", \"mint_model_work_scheduler\", \"mint_maintenance_cron\", \"mint_model_actor_supervisor\"]:
    try:
        a = ray.get_actor(name, namespace=\"mint_wenxi_dev\")
        ray.kill(a, no_restart=True)
        print(f\"killed {name}\")
    except Exception:
        pass
ray.shutdown()
"'