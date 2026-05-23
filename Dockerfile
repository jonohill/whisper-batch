# syntax=docker/dockerfile:1

FROM debian:bookworm-slim AS whisper-build

ARG WHISPER_CPP_REF=v1.8.4
ARG GGML_NATIVE=OFF

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" \
        https://github.com/ggml-org/whisper.cpp.git .

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE="${GGML_NATIVE}" \
        -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_SERVER=ON \
    && cmake --build build --target whisper-server -j "$(nproc)"


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="whisper-batch" \
      org.opencontainers.image.description="OpenAI-compatible HTTP transcription service over a warm whisper.cpp pool" \
      org.opencontainers.image.source="https://github.com/ggml-org/whisper.cpp"

# ffmpeg/ffprobe for audio handling; libgomp1 for the OpenMP the binary links.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=whisper-build /src/build/bin/whisper-server /usr/local/bin/whisper-server

# Install the package with its HTTP server extra (CPU core stays dependency-free;
# the extra pulls FastAPI/uvicorn/python-multipart, all prebuilt wheels).
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

# Run unprivileged.
RUN useradd --create-home --uid 10001 app
USER app

# Mount the model here (kept out of the image — models are large and swappable):
#   docker run -v /path/to/models:/models:ro -e WHISPER_MODEL=/models/ggml-base.en.bin ...
ENV WHISPER_MODEL=/models/ggml-base.en.bin
VOLUME ["/models"]

EXPOSE 8000

# /health needs no auth, so this works even with WHISPER_BATCH_API_KEY set.
# Generous start-period: the pool loads the model into every warm server first.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"]

# CPU-only target -> default to --no-gpu. Override/extend at `docker run` time,
# e.g. `docker run <img> -w 4 -t 2 --no-gpu`.
ENTRYPOINT ["whisper-batch-server", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--no-gpu"]
