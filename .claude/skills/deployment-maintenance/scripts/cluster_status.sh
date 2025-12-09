#!/bin/bash
# Check Ray cluster status on Volcano ML Platform
# Usage: ./cluster_status.sh

echo "=== Ray Cluster Tasks ==="
echo ""

# List ray-related tasks
volc ml_task list --name ray --output json 2>/dev/null | python3 -c "
import json
import sys

try:
    data = json.load(sys.stdin)
    tasks = data.get('Result', {}).get('Items', [])
    if not tasks:
        print('No Ray tasks found.')
        sys.exit(0)

    print(f'{'Task Name':<25} {'Status':<12} {'Task ID':<25} {'Created'}')
    print('-' * 80)
    for task in tasks:
        name = task.get('TaskName', 'N/A')[:24]
        status = task.get('State', 'N/A')
        task_id = task.get('Id', 'N/A')
        created = task.get('CreateTime', 'N/A')[:19]
        print(f'{name:<25} {status:<12} {task_id:<25} {created}')
except json.JSONDecodeError:
    print('Failed to parse task list. Run: volc ml_task list')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null || volc ml_task list --name ray

echo ""
echo "For detailed info: volc ml_task get --id TASK_ID"
