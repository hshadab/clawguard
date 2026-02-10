# ClawGuard Docker build
#
# Build context: parent directory containing the jolt-atlas workspace
#   docker build -t clawguard -f clawguard/Dockerfile ..
#
# Run:
#   docker run -p 8080:8080 clawguard

# --- Builder stage ---
FROM rust:1.88-bookworm AS builder

# Install the RISC-V target for ZK proof generation
RUN rustup target add riscv32im-unknown-none-elf

WORKDIR /build

# Copy workspace root files
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Copy workspace members needed by clawguard
COPY onnx-tracer/ onnx-tracer/
COPY zkml-jolt-core/ zkml-jolt-core/
COPY clawguard/ clawguard/

# Build clawguard in release mode
RUN cargo build --release --package clawguard

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/clawguard /app/clawguard

# Copy data directory (registry, cross-reference files)
COPY clawguard/data/ /app/data/

EXPOSE 8080

CMD ["/app/clawguard", "serve", "--bind", "0.0.0.0:8080", "--rate-limit", "30"]
