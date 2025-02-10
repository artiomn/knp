# Build image Dockerfile for x86_64.
#
# © 2024 AO Kaspersky Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM gcc:14.1.0

ARG TARGETARCH

ENV OCLINT_FILE=oclint-22.02-llvm-13.0.1-x86_64-linux-ubuntu-20.04.tar.gz \
    PVS_FILE=pvs-studio-7.32.83291.394-amd64.deb \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /azp

RUN \
  echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes \
  && if [ "${TARGETARCH}" = "arm64" ]; then export CUDA_ARCH="arm64"; export DEBIAN_VERSION="ubuntu2204"; else export CUDA_ARCH="x86_64"; export DEBIAN_VERSION=debian$(cat /etc/debian_version | sed 's/\..*//'); fi \
  && mkdir -m 0755 -p /etc/apt/keyrings/ \
  && wget "https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb" \
  && dpkg -i cuda-keyring_1.1-1_all.deb \
  && rm cuda-keyring_1.1-1_all.deb \
  && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/${CUDA_ARCH}/3bf863cc.pub | gpg --dearmor -o /etc/apt/keyrings/cuda.gpg \
  && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/${CUDA_ARCH}/ /" | \
    tee /etc/apt/sources.list.d/cuda-${DEBIAN_VERSION}-${CUDA_ARCH}.list \
  && apt-get update \
  && apt-get -y --no-install-recommends --show-progress install \
    apt-transport-https \
    bandit \
    bison \
    build-essential \
    ca-certificates \
    clang \
    clang-tidy \
    cmake \
    cloc \
    cpplint \
    cppcheck \
#    cppzmq-dev \
    curl \
    cuda-12-8 \
    cuda-libraries-dev-12-8 \
    libnpp-dev-12-8 \
    cuda-nvtx-12-8 \
    cuda-toolkit-12-8 \
    cuda-tools-12-8 \
    cuda-compiler-12-8 \
    libcusparse-dev-12-8 \
    libcublas-dev-12-8 \
    flake8 \
    flex \
    fonts-liberation \
    # gcovr \
    gdb \
    gdbserver \
    git \
    gnupg-agent \
    gosu \
    graphviz \
    gsfonts \
    jq \
    # libboost-all-dev \
    libcurl4 \
#    libgtest-dev \
#    libhdf5-dev \
#    libssl-dev \
    libunwind8 \
#    libzmq5-dev \
    iputils-ping \
    iproute2 \
    less \
    meson \
    mypy \
    netcat-openbsd \
    net-tools \
    libopencv-dev \
    p7zip-full \
    pre-commit \
    protobuf-compiler \
    pylint \
    python3 \
    python3-dev \
    python3-build \
    python3-mypy \
    python3-opencv \
    python3-pip \
    # python3-pytest \
    python3-requests \
    python3-setuptools \
    python3-types-requests \
    python3-wheel \
    rapidjson-dev \
    rsync \
    software-properties-common \
    ssh \
    strace \
    sudo \
    tar \
    telnet \
    tcpdump \
    valgrind \
    vim \
    wget \
    zip \
    $(apt list|awk '/boost.*1.81.*-dev/ {print($1);}') \
  # && apt-get purge -y libboost\* \
  && apt-get clean \
  # && git clone --recursive --depth 1 --branch boost-1.84.0 https://github.com/boostorg/boost /usr/local/boost/1.84.0 \
  # && cd /usr/local/boost/1.84.0 \
  # && PY_VER="$(python3 --version|cut -f2 -d' '|cut -f1-2 -d.)" \
  # && echo "using python : ${PY_VER} : /usr/bin/python3 : /usr/include/python${PY_VER} : /usr/lib/python${PY_VER} ;" \
  #      >> tools/build/example/user-config.jam \
  # && cp tools/build/example/user-config.jam ~ \
  # && ./bootstrap.sh  --with-python=/usr/bin/python3 --with-python-version=${PY_VER} --with-python-root=/usr/lib/python${PY_VER} \
  # && ./b2 -a -j 6 cxxflags="-fPIC" cflags="-fPIC" link=static,shared threading=multi \
  && cd /azp \
  # New Doxygen version. \
  && git clone --depth 1 https://github.com/doxygen/doxygen.git \
  && cd doxygen \
  && cmake -G "Unix Makefiles" -DCMAKE_POSITION_INDEPENDENT_CODE=ON . \
  && cmake --build . --parallel 4 \
  && cmake --install . \
  && cd .. \
  && rm -rf doxygen \
  && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
  && add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        mantic \
        stable" \
  && export VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | jq .name -r) \
  && export DESTINATION=/usr/local/bin/docker-compose \
  && curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-`uname -s`-`uname -m` -o "${DESTINATION}" \
  && chmod +x ${DESTINATION} \
  && ln -sf ${DESTINATION} /usr/bin/docker-compose

# COPY "${OCLINT_FILE}" .
RUN \
  pip3 install --break-system-packages gcovr pytest \
  && pip3 uninstall --break-system-packages -y certifi \
#  && tar xf "${OCLINT_FILE}" \
#  && rsync -a oclint-22.02/* /usr/local/ \
  && rm -rf /tmp/*

COPY LICENSE.txt .
COPY NOTICE.txt .

#COPY "${PVS_FILE}" .
#RUN dpkg -i "${PVS_FILE}"

#COPY --chmod=0555 ./start.sh .

#CMD ["./start.sh"]
