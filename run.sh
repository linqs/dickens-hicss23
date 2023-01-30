#!/usr/bin/env bash

# Run all of the experiments.

function main() {
  trap exit SIGINT

  # Setup,
  # Downloads data and PSL jar.
  ./scripts/setup.sh

  # Run Bikeshare experiments.
  ./scripts/run_experiments.sh
}

main "$@"