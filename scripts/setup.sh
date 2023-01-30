#!/usr/bin/env bash

# Note that you can change the version of PSL used with the PSL_VERSION option in the run inference and run wl scripts.
readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..)
readonly CLI_DIR="${BASE_DIR}/cli"

readonly PSL_VERSION='3.0.0-SNAPSHOT'

readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
# Floor by multiples of 5 and then reserve an additional 5 GB.
readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
# readonly JAVA_MEM_GB=24

# Common to all examples.
function standard_setup() {
      pushd . > /dev/null
          cd "${CLI_DIR}"

          # Increase memory allocation.
          sed -i "s/java -jar/java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar/" run_server.sh
          sed -i "s/java -jar/java -Xmx8G -Xms8G -jar/" run_client.sh

          # Set the PSL version.
          sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${PSL_VERSION}'/" run_server.sh
          sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${PSL_VERSION}'/" run_client.sh
      popd > /dev/null
}

function individual_setup() {
    # Fetch Data Script
      pushd . > /dev/null
        cd "${BASE_DIR}/data"
        local fetchDataScript="./fetchData.sh"
        if [ ! -e "${fetchDataScript}" ]; then
            return
        fi

        ${fetchDataScript}
      popd > /dev/null
}

function main() {
   trap exit SIGINT

   standard_setup
   individual_setup

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
