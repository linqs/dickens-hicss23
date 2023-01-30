#!/usr/bin/env bash

# Run all the experiments.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly MODELS_DIR="${BASE_DIR}/bikeshare-models"
readonly RESULTS_DIR="${BASE_DIR}/results"

readonly BIKESHARE_MODELS='autoregressive autoregressive_exogenous collective collective_cluster collective_source_destination collective_spatial'

readonly EXPERIMENTS='Online Incremental'
readonly DATASETS='Bay_Area Blue_Bikes Metro'

declare -A MODEL_OPTIONS
MODEL_OPTIONS[autoregressive]='-D sgd.learningrate=1.0 -D sgd.maxiterations=2000'
MODEL_OPTIONS[autoregressive_arithmetic]='-D sgd.learningrate=1.0 -D sgd.maxiterations=2000'
MODEL_OPTIONS[autoregressive_exogenous]='-D sgd.learningrate=1.0 -D sgd.maxiterations=2000'
MODEL_OPTIONS[collective]='-D sgd.learningrate=1.0 -D sgd.maxiterations=2000'

readonly INFERENCE_OPTIONS='-D sgd.extension=ADAM -D inference.initialvalue=ATOM -D partialgrounding.powerset=true -D reasoner.tolerance=1e-12f'
readonly STANDARD_OPTIONS=''

function run() {
  local model_name=$1

  # Declare paths to output files.
  local out_directory=""
  local out_path=""
  local err_path=""
  local experiment_options=""

  for experiment in ${EXPERIMENTS}; do
    for dataset in ${DATASETS}; do
        echo "Running PSL ${model_name} for experiment ${experiment} on dataset ${dataset}."

        # Declare paths to output files.
        out_directory="${RESULTS_DIR}/${experiment}/${dataset}/${model_name}"
        out_path="${out_directory}/out.txt"
        err_path="${out_directory}/out.err"
        experiment_options="${MODEL_OPTIONS[${model_name}]} ${STANDARD_OPTIONS} ${INFERENCE_OPTIONS}"

        # cp model and data files to cli directory
        python3 scripts/create_ar_rules.py ${dataset} | cat - "${MODELS_DIR}/${model_name}/bikeshare.psl" > "${BASE_DIR}/cli/bikeshare.psl"
        
        cp "${MODELS_DIR}/${model_name}/bikeshare-eval.data" "${BASE_DIR}/cli/bikeshare-eval.data"

        if [[ -e "${out_path}" ]]; then
          echo "Output file already exists, skipping: ${out_path}"
        else
          mkdir -p ${out_directory}
          pushd . > /dev/null
             cd "${BASE_DIR}/cli"

             # Set the data split.
             sed -i "s/bikeshare\/eval\/[0-9]\+/${dataset}\/eval\/000/g" "bikeshare-eval.data"

             ./run.sh $experiment $dataset ${experiment_options} > "${out_path}" 2> "${err_path}"

             mv inferred-predicates "${out_directory}/"
             cp "./bikeshare-eval.data" "${out_directory}/bikeshare-eval.data"
             cp "./bikeshare.psl" "${out_directory}/bikeshare.psl"
             cp "./run_client.sh" "${out_directory}/run_client.sh"
             cp "./run_server.sh" "${out_directory}/run_server.sh"
             mv "./out_server.txt" "${out_directory}/out_server.txt"
             mv "./out_server.err" "${out_directory}/out_server.err"
             mv "./client_output" "${out_directory}/client_output"
          popd > /dev/null
        fi
      done
  done
}

function default_models() {
  local model_name=$1
  
  run ${model_name}
}

function main() {
  trap cleanup SIGINT

  for model_name in ${BIKESHARE_MODELS} ; do
      default_models "${model_name}"
  done
}

function cleanup() {
  for pid in $(jobs -p); do
    pkill -P ${pid}
    kill ${pid}
  done
  pkill -P $$
  exit
}

main "$@"
