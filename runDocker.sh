#!/usr/bin/bash
# Run Docker container with the IMAGE ai-surveillance, with the required settings to connect to the host X server
# and to use the host camera, creating a volume to store the results to access them from the host.

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Run IMAGE surveillance
docker run -m 8GB -it --rm -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v surveillance:/usr/src/app/results --device=/dev/video0:/dev/video0 -it juanabascal/ai-surveillance:latest
xhost -local:docker
