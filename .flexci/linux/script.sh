#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .flexci/linux/script.sh python39".
#
# Environment variables:
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.

# Fail immedeately on error or unbound variables.
set -eu

# note: These values can be overridden per project using secret environment
# variables of FlexCI.
PFHEDGE_FLEXCI_GCS_BUCKET=${PFHEDGE_FLEXCI_GCS_BUCKET:-pfhedge-artifacts-pfn-public-ci}

declare -A python_versions=(
  ["python38"]="3.8.12"
  ["python39"]="3.9.7"
  ["python310"]="3.10.4"
)

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/../.."; pwd)"

  echo "TARGET: ${TARGET}"
  echo "SRC_ROOT: ${SRC_ROOT}"

  # Run target-specific commands.
  case "${TARGET}" in
    python* )
      python_version=${python_versions["${TARGET}"]}
      # Prepare docker args.
      docker_args=(
        docker run --rm --ipc=host --privileged --runtime=nvidia
        --env CUDA_VISIBLE_DEVICES
        --env python_version=${python_version}
        --volume="${SRC_ROOT}:/src"
        --volume="/tmp/output:/output"
        --workdir="/src"
      )
      run "${docker_args[@]}" \
          "nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04" \
          bash /src/.flexci/linux/test.sh
      gsutil -m -q cp -r /tmp/output/htmlcov gs://${PFHEDGE_FLEXCI_GCS_BUCKET}/pfhedge/pytest-cov/${CI_JOB_ID}/htmlcov
      echo "pytest-cov output: https://storage.googleapis.com/${PFHEDGE_FLEXCI_GCS_BUCKET}/pfhedge/pytest-cov/${CI_JOB_ID}/htmlcov/index.html"
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

################################################################################
# Bootstrap
################################################################################
main "$@"
