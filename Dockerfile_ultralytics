# Image
# FROM python:3.9-slim-buster
# Ultralytics image at /usr/src/ultralytics/
FROM ultralytics/ultralytics:latest-cpu 

# Setting up working directory 
WORKDIR /usr/src/app
COPY run_cam_surveillance_yolov8.py requirements.txt runSurveillance.sh ./
COPY config  ./config/
COPY utils ./utils/
COPY models/ ./models/
COPY data/ ./data/
RUN chmod +x runSurveillance.sh

# Install system packages
RUN apt-get update && apt-get install python3  && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Ultralytics and opencv
#RUN pip install ultralytics && \
RUN pip install --no-cache-dir -r requirements.txt  

# Too large image -> search for reduced library versions and 
# install ultralytics from source

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)  

ENV QT_X11_NO_MITSHM=1

CMD ["bash"]
#ENTRYPOINT [ "python", "./run_cam_surveillance_yolov8.py"]
