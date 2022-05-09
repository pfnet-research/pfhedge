#!/bin/bash
# Thanks to https://github.com/pfnet/pytorch-pfn-extras/blob/master/.flexci/linux/script.sh

# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .flexci/linux/script.sh python39".
#
# Environment variables:
# - PFHEDGE_FLEXCI_IMAGE_NAME ... The Docker image name (without tag) to be
#       used for CI.
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.

# Fail immedeately on error or unbound variables.
set -eu

# note: These values can be overridden per project using secret environment
# variables of FlexCI.
PFHEDGE_FLEXCI_IMAGE_NAME=${PFHEDGE_FLEXCI_IMAGE_NAME:-asia.gcr.io/pfn-public-ci/pfhedge}
PFHEDGE_FLEXCI_GCS_BUCKET=${PFHEDGE_FLEXCI_GCS_BUCKET:-chainer-artifacts-pfn-public-ci}

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/../.."; pwd)"

  # Initialization.
  prepare_docker &
  wait

  # Prepare docker args.
  docker_args=(
    docker run --rm --ipc=host --privileged --runtime=nvidia
    --env CUDA_VISIBLE_DEVICES
    --volume="${SRC_ROOT}:/src"
    --volume="/tmp/output:/output"
    --workdir="/src"
  )

  # Run target-specific commands.
  case "${TARGET}" in
    python* )
      run "${docker_args[@]}" \
          "${PFHEDGE_FLEXCI_IMAGE_NAME}:${TARGET}" \
          /src/.flexci/linux/test.sh
      gsutil -m -q cp -r /tmp/output/htmlcov gs://${PFHEDGE_FLEXCI_GCS_BUCKET}/pfhedge/pytest-cov/${CI_JOB_ID}/htmlcov
      echo "pytest-cov output: https://storage.googleapis.com/${PFHEDGE_FLEXCI_GCS_BUCKET}/pfhedge/pytest-cov/${CI_JOB_ID}/htmlcov/index.html"
      ;;
    prep )
      # Build and push docker images for unit tests.
      run "${SRC_ROOT}/.flexci/linux/build_and_push.sh" \
          "${PFHEDGE_FLEXCI_IMAGE_NAME}"
      ;;
    * )
      echo "${TARGET}: Invalid target."
      exit 1
      ;;
  esac
}

################################################################################
# Utility functions
################################################################################

# run executes a command.  If DRYRUN is enabled, run just prints the command.
run() {
  echo '+' "$@" >&2
  if [ "${DRYRUN:-}" == '' ]; then
    "$@"
  fi
}

# Configure docker to pull images from gcr.io.
prepare_docker() {
  run gcloud auth configure-docker
}

################################################################################
# Bootstrap
################################################################################
main "$@"
