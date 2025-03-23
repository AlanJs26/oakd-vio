# those following 3 lines would need to be done only one time
export XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it --rm \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  --device-cgroup-rule='c 189:* rmw' \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="XAUTHORITY=$XAUTH" \
  --volume="$XAUTH:$XAUTH" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMP_WAIT_POLICY=passive \
  --network host \
  -v ./depthai-core:/root/depthai-core \
  -v ./helloworld-oak:/root/helloworld-oak \
  luxonis/depthai-library:latest \
  bash
# luxonis/depthai-ros:jazzy-latest \
# bash
# python3 /depthai-python/examples/ColorCamera/rgb_preview.py
