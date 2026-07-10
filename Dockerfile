# syntax=docker/dockerfile:1
#
# oropt GUI image (roadmap item U3, the shippable slice).
#
#   docker build -t oropt .
#   docker run --rm -p 8501:8501 oropt          # then open http://localhost:8501
#
# What this image IS:  the oropt package + Streamlit GUI, self-contained for the
# zero-solver `demo` backend (`demo.enabled: true`) and for post-processing /
# config-editing work.
#
# What this image is NOT:  it does NOT bundle OpenRadioss. The MUMPS-implicit
# solver image (`openradioss-mumps:*`) is not publicly distributed with this
# repo — users build/load it separately — so real solves from inside this
# container require docker-out-of-docker (see docs/docker_image.md, mode c).
#
# Layering on the solver image (experimental): BASE_IMAGE lets you rebase this
# image onto a locally built solver image (`--build-arg
# BASE_IMAGE=openradioss-mumps:20260520`) IF that image ships Python >= 3.10
# and pip. The supported path remains BASE_IMAGE=python:3.12-slim + the
# docker.sock mount; see docs/docker_image.md before trying to rebase.
ARG BASE_IMAGE=python:3.12-slim

# --------------------------------------------------------------------------
# Stage 1 — build wheels for oropt and every runtime dependency, so the final
# stage installs from local wheels only (no compilers, no pip network access,
# no setuptools build cruft in the runtime layers).
# --------------------------------------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml ./
COPY oropt/ oropt/

# [gui] pulls streamlit + stpyvista on top of the core numpy/scipy/pyvista
# stack. Deliberately NOT installed:
#   * [dev]        — pytest/ruff/mypy have no place in a runtime image.
#   * [growthmesh] — TetGen (growth-mesh PREPARE step) is AGPL-licensed;
#                    evaluate the licence before baking it into an image you
#                    redistribute. Users who need it can extend this image
#                    with `pip install tetgen` (or `.[growthmesh]`).
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[gui]"

# --------------------------------------------------------------------------
# Stage 2 — runtime.
# --------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

# VTK/pyvista wheels on a slim (X-less) base need these shared libraries for
# headless rendering: libgl1 (OpenGL dispatch), libxrender1/libx11-6 (X client
# libs the VTK wheel links even off-screen), libgomp1 (OpenMP, used by VTK and
# scipy kernels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libx11-6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optional: a static docker CLI, ONLY needed for docker-out-of-docker (mode c
# in docs/docker_image.md) where the containerised GUI launches the dockerised
# OpenRadioss-MUMPS solver through a mounted /var/run/docker.sock.
#   docker build --build-arg INSTALL_DOCKER_CLI=1 -t oropt .
# x86_64 only; adjust the URL for other architectures.
ARG INSTALL_DOCKER_CLI=0
ARG DOCKER_CLI_VERSION=27.5.1
RUN if [ "$INSTALL_DOCKER_CLI" = "1" ]; then \
        python -c "import urllib.request; urllib.request.urlretrieve('https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz', '/tmp/docker.tgz')" \
        && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
        && mv /tmp/docker/docker /usr/local/bin/docker \
        && rm -rf /tmp/docker.tgz /tmp/docker; \
    fi

# Install oropt + GUI deps from the wheels built in stage 1 (offline, cache-free).
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels "oropt[gui]"

# Non-root user. -m so streamlit can write its ~/.streamlit config/state.
RUN useradd -m -u 1000 oropt \
    && mkdir -p /cases \
    && chown oropt:oropt /cases

# /app holds the same oropt source that was installed above, so the documented
# repo-root invocation (`streamlit run oropt/gui/app.py`) works verbatim.
WORKDIR /app
COPY --chown=oropt:oropt oropt/ oropt/

USER oropt

# Headless rendering + container-friendly Streamlit defaults.
ENV PYVISTA_OFF_SCREEN=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Streamlit's built-in health endpoint; python stands in for curl (not on slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

# Extra streamlit flags can be appended at `docker run oropt <flags>`.
ENTRYPOINT ["streamlit", "run", "oropt/gui/app.py", \
            "--server.address", "0.0.0.0", "--server.port", "8501"]
