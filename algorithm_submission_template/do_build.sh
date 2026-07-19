#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"
FASTSLIDE_DIR="${REPO_ROOT}/fastslide"
FASTSLIDE_WHEEL_DIR="${SCRIPT_DIR}/build/fastslide_wheels"
BAZELISK_BIN="${BAZELISK_BIN:-bazelisk}"
FASTSLIDE_BAZEL_TARGET="${FASTSLIDE_BAZEL_TARGET:-//python:fastslide_wheel_cp311}"
FASTSLIDE_BAZEL_PLATFORM="${FASTSLIDE_BAZEL_PLATFORM:-//platforms:linux_x86_64}"

build_fastslide_wheel() {
  if [ ! -d "$FASTSLIDE_DIR" ]; then
    echo "=+= FastSlide source directory not found at ${FASTSLIDE_DIR}" >&2
    exit 1
  fi

  if ! command -v "$BAZELISK_BIN" >/dev/null 2>&1; then
    if [ "$BAZELISK_BIN" = "bazelisk" ] && command -v bazel >/dev/null 2>&1; then
      BAZELISK_BIN="bazel"
    else
      echo "=+= Bazelisk is required to build the FastSlide Linux wheel." >&2
      echo "=+= Install bazelisk or set BAZELISK_BIN=/path/to/bazelisk, then rerun this script." >&2
      exit 1
    fi
  fi

  echo "=+= Building FastSlide wheel with ${BAZELISK_BIN}"
  (
    cd "$FASTSLIDE_DIR"
    "$BAZELISK_BIN" build \
      --config=hermetic \
      --platforms="$FASTSLIDE_BAZEL_PLATFORM" \
      "$FASTSLIDE_BAZEL_TARGET"
  )

  mkdir -p "$FASTSLIDE_WHEEL_DIR"
  rm -f "${FASTSLIDE_WHEEL_DIR}"/fastslide-*.whl

  local wheel
  wheel=$(find "${FASTSLIDE_DIR}/bazel-bin/python" -maxdepth 1 \( -type f -o -type l \) -name 'fastslide-*.whl' | sort | tail -n 1)
  if [ -z "$wheel" ]; then
    echo "=+= FastSlide wheel was not produced under ${FASTSLIDE_DIR}/bazel-bin/python" >&2
    exit 1
  fi

  cp "$wheel" "$FASTSLIDE_WHEEL_DIR/"
  echo "=+= FastSlide wheel staged at ${FASTSLIDE_WHEEL_DIR}/$(basename "$wheel")"
}

build_fastslide_wheel

docker build \
  --platform=linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$REPO_ROOT" 2>&1
