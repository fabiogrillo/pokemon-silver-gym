# Dockerized playback demo: run a pretrained generalist from the egg-delivered state and watch it
# head for Falkner. CPU-only inference — no GPU needed. The ROM and (optionally) the checkpoint are
# MOUNTED at runtime, never baked into the image.
#
# Build:
#   docker build -t silver-falkner-agent .
#
# Run (mount the ROM + a checkpoint dir, persist outputs to ./runs):
#   docker run --rm \
#     -v "$PWD/pokemon_rom.gbc:/app/pokemon_rom.gbc" \
#     -v "$PWD/runs:/app/runs" \
#     -e MODEL=/app/runs/checkpoints/agent_076/agent_076_final.zip \
#     silver-falkner-agent --map
#
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MODEL=/app/runs/checkpoints/agent_076/agent_076_final.zip \
    STATE=/app/saves/egg_delivered_clean.state \
    MAX_STEPS=20000

# SDL2 + GL runtime libs for PyBoy (it runs headless, but the bundled SDL2 still loads these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsdl2-2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (small), then the rest of the runtime deps.
COPY requirements-play.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements-play.txt

# Application code + the corridor map asset + save states. ROM and checkpoints are mounted at runtime.
COPY agents/ agents/
COPY env/ env/
COPY saves/ saves/
COPY assets/ assets/

ENTRYPOINT ["python", "-m", "agents.rl.play"]
CMD ["--map"]
