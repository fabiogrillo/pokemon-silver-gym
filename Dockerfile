# Dockerized playback demo: run the pretrained gym agent (agent_087, 100% badge) from inside the
# Violet City Gym and watch it beat Falkner. CPU-only inference — no GPU needed. The demo checkpoint
# is BAKED into the image; only the ROM is mounted at runtime (it is not distributable).
#
# Build:
#   docker build -t silver-falkner-agent .
#
# Run (mount your ROM; the agent_087 checkpoint is already inside the image):
#   docker run --rm \
#     -v "$PWD/pokemon_silver.gbc:/app/pokemon_rom.gbc" \
#     silver-falkner-agent
#
# Override the checkpoint by mounting -v "$PWD/runs:/app/runs" and -e MODEL=/app/runs/.../<ckpt>.zip
#
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MODEL=/app/assets/checkpoints/agent_087_final.zip \
    STATE=/app/saves/violet_city_gym.state \
    MAX_STEPS=6000

# SDL2 + GL runtime libs for PyBoy (it runs headless, but the bundled SDL2 still loads these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsdl2-2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (small), then the rest of the runtime deps.
COPY requirements-play.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements-play.txt

# Application code + demo save states + the assets the agent needs at runtime (collision grids,
# stitched map, demo checkpoints). Copied selectively so the comparison videos in assets/ don't
# bloat the image. The ROM is mounted at runtime (not bundled — legal). The checkpoints live under
# assets/ (not runs/) so the ./runs volume mount in docker-compose can't shadow them.
COPY agents/ agents/
COPY env/ env/
COPY saves/violet_city_gym.state saves/egg_delivered_clean.state saves/
COPY assets/collision/ assets/collision/
COPY assets/maps/ assets/maps/
COPY assets/checkpoints/ assets/checkpoints/

ENTRYPOINT ["python", "-m", "agents.rl.play"]
CMD []
