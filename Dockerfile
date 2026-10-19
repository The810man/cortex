# The brain's own machine. `cortex run` builds and starts it: see src/cortex/container.py.
FROM python:3.12-slim
# FreeCAD, without a window: the brain builds solid bodies in it with `run_freecad`. It is large
# and comes first, so that a change to the brain's own code never fetches it again.
RUN apt-get update && apt-get install -y --no-install-recommends freecad && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
# pip is for the brain: in full mode it may install what its scripts need. Gone when it stops.
RUN uv sync --locked --no-dev && uv pip install --python /opt/venv/bin/python pip

ENV PATH=/opt/venv/bin:$PATH CORTEX_CONTAINED=1
ENTRYPOINT ["cortex"]
