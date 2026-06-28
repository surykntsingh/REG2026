#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"

docker build \
  --platform=linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$REPO_ROOT" 2>&1
