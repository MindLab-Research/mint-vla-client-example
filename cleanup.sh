ssh driver 'PY=/vePFS-Mindverse/share/code/wenxi/host-venv/bin/python3.13
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
NS=mint_wenxi_dev
$PY -c "
import ray
ray.init(address=\"$HEAD_IP:6379\", namespace=\"$NS\", ignore_reinit_error=True, log_to_driver=False)
# Kill all named actors in your namespace
for actor in ray.util.list_named_actors(all_namespaces=True):
    ns = str(actor.get(\"namespace\") or \"\")
    name = str(actor.get(\"name\") or \"\")
    if ns == \"$NS\" and name:
        try:
            a = ray.get_actor(name, namespace=ns)
            ray.kill(a, no_restart=True)
            print(f\"killed {name}\")
        except Exception as e:
            print(f\"skip {name}: {e}\")
# Also clean issue-scoped namespaces
for actor in ray.util.list_named_actors(all_namespaces=True):
    ns = str(actor.get(\"namespace\") or \"\")
    name = str(actor.get(\"name\") or \"\")
    if ns.startswith(\"mint_wenxi_issue_\") and name:
        try:
            a = ray.get_actor(name, namespace=ns)
            ray.kill(a, no_restart=True)
            print(f\"killed {name} in {ns}\")
        except Exception as e:
            print(f\"skip {name} in {ns}: {e}\")
ray.shutdown()
"'