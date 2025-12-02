#!/bin/bash

# parameters
datajuicer_path= # path to data-juicer
config_path= # path to config file

# hostname
hostname=$(hostname)

# into datajuicer_path
cd "$datajuicer_path" || { echo "Could not change directory to $datajuicer_path"; exit 1; }

# copy and generate new config file for current host

config_basename=$(basename "$config_path")
config_dirname=$(dirname "$config_path")
config_extension="${config_basename##*.}"
config_basename="${config_basename%.*}"

new_config_file="${config_dirname}/${config_basename}_$hostname.$config_extension"
cp "$config_path" "$new_config_file" || { echo "Could not copy config file"; exit 1; }

echo "$new_config_file"

if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_I_SUFFIX=".bak"
else
    SED_I_SUFFIX=""
fi

if grep -q "dataset_path: .*\.json" "$new_config_file"; then
    # .json data_path
    sed -i$SED_I_SUFFIX "s|\(dataset_path: \)\(.*\)\(/[^/]*\)\(.json\)|\1\2\3_$hostname\4|" "$new_config_file"
else
    # dir dataset_path
    sed -i$SED_I_SUFFIX "s|\(dataset_path: '\)\(.*\)'\(.*\)|\1\2_$hostname'\3|" "$new_config_file"
fi

if grep -q "export_path: .*\.json" "$new_config_file"; then
    # .json data_path
    sed -i$SED_I_SUFFIX "s|\(export_path: \)\(.*\)\(/[^/]*\)\(.json\)|\1\2\3_$hostname\4|" "$new_config_file"
else
    # dir export_path
    sed -i$SED_I_SUFFIX "s|\(export_path: '\)\(.*\)'\(.*\)|\1\2_$hostname'\3|" "$new_config_file"
fi

# run to process data
python tools/process_data.py --config "$new_config_file"
