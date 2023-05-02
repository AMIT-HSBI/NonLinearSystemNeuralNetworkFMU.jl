# Build with
# docker build \
#  --build-arg USERNAME=$(whoami) \
#  --build-arg USER_UID=$(id -u) \
#  --build-arg USER_GID=$(id -g) \
#  -t julia-with-omc .

FROM julia

ARG USERNAME=juliauser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Get OpenModelica repo and key
RUN apt-get update && \
    apt-get install -qy \
      ca-certificates \
      curl \
      gnupg \
      lsb-release \
      wget \
      tar && \
    curl -fsSL http://build.openmodelica.org/apt/openmodelica.asc | gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg && \
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] https://build.openmodelica.org/apt \
      $(lsb_release -cs) nightly" | tee /etc/apt/sources.list.d/openmodelica.list > /dev/null

# Install OpenModelica nightly
RUN apt update && \
    apt install -qy openmodelica

# Install CMake v3.24.2
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.tar.gz && \
    tar -xvf cmake-3.24.2-linux-x86_64.tar.gz
ENV PATH="/cmake-3.24.2-linux-x86_64/bin:${PATH}"
RUN cmake --version

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

ENV USER=$USERNAME
ENV SHELL /bin/bash

USER $USERNAME
