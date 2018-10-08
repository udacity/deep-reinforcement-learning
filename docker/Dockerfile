FROM ubuntu:18.04

################################################################################
#  Install python3.6 and pip
################################################################################
RUN apt-get update && apt-get install -y \
  sudo \
  git \
  python3.6 \
  python3-pip \
  python3-dev

RUN pip3 install --upgrade pip

################################################################################
#  Install minimal OpenAI gym
################################################################################
RUN apt-get update && apt-get install -y \
  zlib1g-dev \
  libjpeg-dev \
  cmake \
  swig \
  python-pyglet \
  python3-opengl \
  libboost-all-dev \
  libsdl2-dev \
  libosmesa6-dev \
  patchelf \
  ffmpeg \
  xvfb

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /workdir
COPY ./src/python/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

################################################################################
#  Install openAI gym
################################################################################
WORKDIR /build
RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip3 install -e '.[classic_control, box2d]'
RUN rm /build -r
#################################################################################
##  Setup ipython kernel
#################################################################################
#RUN python3 -m ipykernel install --user --name drlnd --display-name "drlnd"

#################################################################################
##  Setup jupyter
#################################################################################
COPY ./docker/run_jupyter.sh /run_jupyter.sh
COPY ./docker/jupyter_notebook_config.py /.jupyter/jupyter_notebook_config.py
RUN chmod +x /run_jupyter.sh && \
    python3 -m ipykernel.kernelspec
ENV PASSWORD='zeus'

################################################################################
#  add user to docker group
################################################################################
ARG UID
RUN useradd docker -l -u $UID -s /bin/bash -m
USER docker

################################################################################
#  setup environment
################################################################################
WORKDIR /workdir
ENV ROOT_DIR=/workdir
ENTRYPOINT ["/bin/bash"]
