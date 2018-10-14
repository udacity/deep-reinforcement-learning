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
WORKDIR /usr/lib
RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip3 install -e '.[classic_control, box2d]'


################################################################################
#  add user to docker group
################################################################################
#ARG UID
#RUN useradd docker -l -u $UID -s /bin/bash -m
#USER docker

################################################################################
#  Setup jupyter
################################################################################
WORKDIR /workdir
COPY ./docker/run_jupyter.sh /run_jupyter.sh
COPY ./docker/jupyter_notebook_config.py /root/.jupyter/
RUN chmod +x /run_jupyter.sh
ENV PASSWORD='zeus'

################################################################################
#  Setup ipython kernel
################################################################################
RUN python3 -m ipykernel install --user --name drlnd --display-name "drlnd"

################################################################################
#  setup environment
################################################################################
WORKDIR /workdir
ENV ROOT_DIR=/workdir
ENTRYPOINT ["/bin/bash"]
