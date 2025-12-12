#!/usr/bin/env python3
"""Kill persistent Megatron actor(s).

Use this script to:
1. Kill existing actor after code changes (actor must restart to pick up changes)
2. Free up GPU resources
3. Reset training state

Run from tinker-server root directory.
"""

import argparse
import sys

import ray


def main():
    parser = argparse.ArgumentParser(description="Kill Megatron actor(s)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Kill all Megatron actors in the namespace",
    )
    parser.add_argument(
        "--name",
        default="persistent_megatron_worker_group_v2",
        help="Actor name to kill (default: persistent singleton)",
    )
    args = parser.parse_args()

    # Initialize Ray
    namespace = "tinker"
    if not ray.is_initialized():
        ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)

    print(f"Ray initialized, namespace={namespace}")

    if args.all:
        # Kill all actors with "megatron" in the name
        from ray.util.state import list_actors

        actors = list_actors(filters=[("state", "=", "ALIVE")])
        killed = 0
        for actor in actors:
            if "megatron" in actor.name.lower() if actor.name else False:
                try:
                    handle = ray.get_actor(actor.name, namespace=namespace)
                    print(f"Killing actor: {actor.name}")
                    try:
                        ray.get(handle.shutdown.remote(), timeout=30)
                    except Exception as e:
                        print(f"  Shutdown method failed: {e}")
                    ray.kill(handle)
                    killed += 1
                except Exception as e:
                    print(f"  Failed to kill {actor.name}: {e}")
        print(f"Killed {killed} Megatron actors")
    else:
        # Kill specific actor
        actor_name = args.name
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
            print(f"Found actor: {actor_name}")

            # Try graceful shutdown first
            try:
                print("Attempting graceful shutdown...")
                ray.get(actor.shutdown.remote(), timeout=30)
                print("Graceful shutdown completed")
            except Exception as e:
                print(f"Graceful shutdown failed: {e}")

            # Force kill
            print("Force killing actor...")
            ray.kill(actor)
            print(f"Killed actor: {actor_name}")

        except ValueError:
            print(f"Actor not found: {actor_name}")
            sys.exit(0)  # Not an error - actor may not exist
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
