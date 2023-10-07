# docker build -t juanabascal/ai-surveillance .

# Image
FROM python:3.9-slim-buster

# Setting up working directory 
WORKDIR /usr/src/app
COPY requirements.txt ./

# Install system packages
# RUN apt-get update && apt-get install python3  && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk \
     && rm -rf /var/lib/apt/lists/*

# Ultralytics and opencv
#RUN pip install ultralytics && \
RUN pip install --no-cache-dir -r requirements.txt  

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)  

COPY data/ ./data/
COPY src/ ./src/
COPY runSurveillance.sh ./
RUN chmod +x runSurveillance.sh
#COPY models/ ./models/

ENV QT_X11_NO_MITSHM=1

CMD ["bash"]
# cd src
#ENTRYPOINT [ "python", "./run_cam_surveillance.py"]
