#!/bin/bash
# usage: ./run.sh flight0|survey|stop [extra run_mission.py flags, e.g. --no-gui --no-rviz --replan]
cd "$(dirname "$0")"
mode=${1:-flight0}; shift
case $mode in
  flight0) exec python3 ../bridge_autonomy/run_mission.py config.yaml --flight flight0 "$@" ;;
  survey)  exec python3 ../bridge_autonomy/run_mission.py config.yaml "$@" ;;
  stop)    exec python3 ../bridge_autonomy/run_mission.py config.yaml --stop ;;
  *) echo "usage: $0 flight0|survey|stop [flags]"; exit 1 ;;
esac
