FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Replaces the tmux devcontainer feature
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    stow \
    curl \
    wget \
    curl \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /projects/bwood/classes/brain_state_pred

CMD ["/bin/bash"]
