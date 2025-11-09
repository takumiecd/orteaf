# syntax=docker/dockerfile:1.4
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        llvm \
        cmake \
        ninja-build \
        git \
        python3 \
        python3-pip \
        doxygen \
        graphviz \
        ccache \
        pkg-config \
        curl \
 && rm -rf /var/lib/apt/lists/*

# Install yaml-cpp 0.8.0 from source to match macOS build
RUN git config --global advice.detachedHead false \
 && git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git /tmp/yaml-cpp \
 && cmake -S /tmp/yaml-cpp -B /tmp/yaml-cpp/build \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DYAML_CPP_BUILD_TOOLS=OFF \
 && cmake --build /tmp/yaml-cpp/build -j$(nproc) \
 && cmake --install /tmp/yaml-cpp/build \
 && rm -rf /tmp/yaml-cpp \
 && ldconfig

WORKDIR /workspace

CMD ["/bin/bash"]
