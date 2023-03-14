ARG base_image="nvidia/cuda:11.2.1-devel-ubuntu20.04"
FROM ${base_image}

# Install pyenv requirements.
# https://github.com/pyenv/pyenv/wiki/Common-build-problems#requirements
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install \
        build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev git cmake protobuf-compiler libprotobuf-dev \
        openmpi-bin openmpi-common && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install pyenv.
RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT=/opt/pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

# Install Python.
ARG python_version="3.9.7"
RUN pyenv install ${python_version} && \
    pyenv global ${python_version}

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH ${HOME}/.poetry/env:${PATH}
