#!/usr/bin/bash
# Run Docker container with the IMAGE ai-surveillance, with the required settings to connect to the host X server
# and to use the host camera, creating a volume to store the results to access them from the host.

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Run IMAGE surveillance with the following settings:
# -m 8GB: 8GB of RAM
# -it: interactive mode
# --rm: remove container after exit
# -e DISPLAY=$DISPLAY: set DISPLAY environment variable
# -v $XSOCK:$XSOCK: mount XSOCK
# -v $XAUTH:$XAUTH: mount XAUTH
# -e XAUTHORITY=$XAUTH: set XAUTHORITY environment variable
# -v surveillance:/usr/src/app/src/results/: mount volume surveillance
# --device=/dev/video0:/dev/video0: mount host camera
# -v /etc/localtime:/etc/localtime:ro: mount host time
# -it juanabascal/ai-surveillance:latest: run IMAGE juanabascal/ai-surveillance:latest
# -p 587:587: expose port 587 to send emails
# -v $(pwd)/../mailtrap:/usr/src/mailtrap: mount mailtrap config file (token, ...)
docker run -m 8GB -it --rm -p 587 -e DISPLAY=$DISPLAY -v $(pwd)/../mailtrap:/usr/src/mailtrap -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v surveillance:/usr/src/app/src/results/ -v /etc/localtime:/etc/localtime:ro --device=/dev/video0:/dev/video0 -it juanabascal/ai-surveillance:mobnetv2
xhost -local:docker