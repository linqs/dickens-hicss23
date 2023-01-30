#!/bin/bash

# If the raw data does not already exist, then this script will download it from the linqs data server.
# Then, if the PSL formatted data does not already exist, then this script will construct it from the raw data.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly DATA_CONSTRUCTION_SCRIPT="${THIS_DIR}/data-construction/construct.py"

readonly DATA_URL='https://linqs-data.soe.ucsc.edu/public/dickens-hicss23/bikeshare_raw.zip'
#readonly DATA_URL='https://linqs-data.soe.ucsc.edu/public/dickens-hicss23/bluebikes_raw.zip'
#readonly DATA_URL='https://linqs-data.soe.ucsc.edu/public/dickens-hicss23/metro_raw.zip'
readonly DATA_FILE='bikeshare_raw.zip'
readonly EXTRACTED_DATA_FILE='bikeshare_raw'
readonly DATA_DIR="${THIS_DIR}/../data"

function main() {
    set -e
    trap exit SIGINT

    check_requirements

    fetch_file "${DATA_URL}" "${DATA_DIR}/${DATA_FILE}"
    extract_zip "${DATA_DIR}/${DATA_FILE}" "${DATA_DIR}/${EXTRACTED_DATA_FILE}"

    python3 ${DATA_CONSTRUCTION_SCRIPT} ${DATA_DIR}
}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download dataset'
      exit 10
   fi

   type tar > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: tar required to extract dataset'
      exit 11
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2

   if [[ -e "${path}" ]]; then
      echo "Data file found cached, skipping download."
      return
   fi

   echo "Downloading ${url} with command: $FETCH_COMMAND"
   $(get_fetch_command) "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${url} file"
      exit 30
   fi
}

function extract_zip() {
   local path=$1
   local expectedDir=$2

   if [[ -e "${expectedDir}" ]]; then
      echo "Extracted data found cached, skipping extract."
      return
   fi

   echo "Extracting data from ${path}."
   unzip "${path}" -d ${DATA_DIR}
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract data from ${path}"
      exit 40
   fi
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
