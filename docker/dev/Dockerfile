# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        ninja-build \
        git \
        python3 \
        python3-pip \
        doxygen \
        graphviz \
        ccache \
        pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Build and install yaml-cpp 0.8.0 (supports std::string_view)
# This ensures consistency with macOS Homebrew version
RUN git config --global advice.detachedHead false \
 && git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git /tmp/yaml-cpp \
 && cd /tmp/yaml-cpp \
 && cmake -S . -B build \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DYAML_CPP_BUILD_TOOLS=OFF \
 && cmake --build build -j$(nproc) \
 && cmake --install build \
 && cd / \
 && rm -rf /tmp/yaml-cpp \
 && ldconfig

# Default workdir for mounting the project
WORKDIR /workspace

CMD ["/bin/bash"]
