# Note for dataset path

The videos/images path here support both absolute path and relative path.
Please use an address that can be accessed on all nodes (such as an address within a NAS file-sharing system).
For relative paths, these should be relative to the directory where the dataset file is located (the dataset_path parameter in the config).
 - if the dataset_path parameter is a directory, then it's relative to dataset_path
 - if the dataset_path parameter is a file, then it's relative to data_path parameter's corresponding dirname
