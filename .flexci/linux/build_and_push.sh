#!/bin/bash -uex
# thanks to https://github.com/pfnet/pytorch-pfn-extras/blob/master/.flexci/linux/build_and_push.sh

IMAGE_BASE="${1:-}"
IMAGE_PUSH=1
if [ "${IMAGE_BASE}" = "" ]; then
  IMAGE_BASE="pfhedge"
  IMAGE_PUSH=0
fi

docker_build_and_push() {
    IMAGE_TAG="${1}"; shift
    IMAGE_NAME="${IMAGE_BASE}:${IMAGE_TAG}"

    pushd "$(dirname ${0})"
    docker build -t "${IMAGE_NAME}" "$@" .
    popd

    if [ "${IMAGE_PUSH}" = "0" ]; then
      echo "Skipping docker push."
    else
      docker push "${IMAGE_NAME}"
    fi
}

WAIT_PIDS=""

docker_build_and_push python38 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04" \
    --build-arg python_version="3.8.12" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

docker_build_and_push python39 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04" \
    --build-arg python_version="3.9.7" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

docker_build_and_push python39 \
    --build-arg base_image="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04" \
    --build-arg python_version="3.10.4" \
    &
WAIT_PIDS="$! ${WAIT_PIDS}"

# Wait until the build complete.
for P in ${WAIT_PIDS}; do
    wait ${P}
done
