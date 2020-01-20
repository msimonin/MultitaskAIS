# Build the trajectories (input for Pierre's normality model)


Docker swarm command:
```
docker service create \
  -e PYTHONUNBUFFERED=1 \
  -e SESAME_UID=<uid> \
  -e SESAME_GID=<gid> \
  --restart-condition=none \
  --network host \
  --mount type=volume,volume-opt=o=addr=srv-bigdata.rennes.grid5000.fr,volume-opt=device=:/srv/bigdata,volume-opt=type=nfs,source=bigdata,target=/bigdata \
  --name traj \
  --reserve-cpu 32\
  msimonin/sesame_build_trajectories  --master "local[32]" main.py "/bigdata/groups/sesame/ais_britany/raw/2011/*/*/*.cdv" brittany
```
