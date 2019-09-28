# Panel Classification

## Directory tree (src)
- config
  - BENCHMARK_CONFIG.json
  - FNOL_CONFIG.json
  - GET_SCORE_CONFIG.json
  - PREPROCESS_CONFIG.json
  - WORKER_CONFIG.json
- utils
  - fnol_generator.py
  - tf_record_generator.py
  - utils.py
  - worker_fnol.py
  - benchmark_utils.py
- 1.1_data_prepracessing.py
- 1.2_benchmark_fnol.py
- 2_train.py
- 3_get_scores_fnol.py


1. A numbered list
    1. A nested numbered list
    2. Which is numbered
2. Which is numbered

## Process

1. Preprocessing the data 
    1. Preparing the training data **(1.1_data_prepracessing.py + PREPROCESS_CONFIG.json + FNOL_CONFIG.json)**
    To prepare the data for training, there are three necessary steps:
          - Splitting the data to train, validate, and testing sets
          - Creating Fnol for each group of data
          - Creating the tf_records 

    2. Creating the FNOL for the benchmark data **(2_benchmark_fnol + BENCHMARK_CONFIG.json)**
    Use the Fnol_param.pkl created in the previous step to create FNOL data for te benchmark dataset.


2. Training the model **(train.py + WORKER_CONFIG.json)**

3. Generating the scores for test and validation data **(3_get_scores_fnol.py + GET_SCORE_CONFIG.json)**

## Code Manual

### Main codes:
- 1.1_data_prepracessing.py : Use for spliting, fnol generating, and tf_record generating. Capable of generating multi_year and multi_panel data.

- 1.2_benchmark_fnol.py : Use for generating FNOL on the benchmark data
- 2_train.py: Use for training the Panel(s) classification model
- 3_get_scores_fnol.py: Use for generating the lables on Validation and the test set

### Config files:

- PREPROCESS_CONFIG.json : (Used by 1.1_data_prepracessing.py) To preprocess your data you only need to change the following arguments:
    - use_fnol: (0 | 1) To generate the fnol and create the respective tf_record for the data set the argument equal to one
    - type_ : String for the type of images your using ("" | "cropped" | "dilated_cropped") 
    - split_name : String with the name of the folder created for the splitted data (in case of using more than one panel, the split name should a general name to cover all the used panels)
    - year: List of string(s) with the list of data year you want to use ( year : ["2017", "2018"] will create a merge set of all the images for our selected panel(s) from both 2017 and 2018 and store the result in a folder called 2017-2018/[split_name]
    - panels_list: List of string(s) with list of all the panels you want to train your model on
    
    
- FNOL_CONFIG.json : (Used by 1.1_data_prepracessing.py) To generate the fnol you only need to change the following argument:
    
    - split_name: String with the name of the folder created for the splitted data
    
- BENCHMARK_CONFIG.json : (Used by 1.2_benchmark_fnol.py) To generate the fnol for the benchmark data you need to change the following arguments:
    - year: String with the year(s) of data been used in the data generating process (seperated by "-", ex: 2017-2018)
    - split_name: string with the name of the folder created for the splitted data. It will use this split folder to load the fnol_param.pkl file which been created in the preprocessing process.
    - benchmark_name : String with the name of the panel in the benchmark dataset. It will generate FNOL data using the fnol_param.pkl for test.parquet file inside this folder.
    
- WORKER_CONFIG.json : (Used by 2_train.py) To train the model on the preprocessed data you need to change the following arguments:
    - data_year: String with the year(s) of data been used in the data generating process (seperated by "-", ex: 2017-2018)
    - split_name: String with the name of the folder created for the splitted data

- GET_SCORE_CONFIG.json : (Used by 3_get_scores_fnol.py) To generate run the code on the testing and validation data and generate the scores  you need to chnage the following arguments:
    - split_name: String with the name of the folder created for the splitted data
    - checkpoint_name: String with the name of the checkpoint you want to test your data on
    - model_name: String with the name of the folder created for the model. (split_name + "_regularized")
    - year: String with the year(s) of data been used in the data generating process (seperated by "-", ex: 2017-2018)
    
    

## Notes
**benchmark_utils.py** can be used to:
  - change the benchmark's image paths regarding their directory in OCI
  - create a filtered version of the benchmark
  - create a balance version of the benchmark data (filtered/non_filtered)


- The models will be store in "/ai-team-shared-5/users/bahar/multi_gpu/models_fnol/". 
- The logs will be store in "/ai-team-shared-5/users/bahar/multi_gpu/logs_fnol".
- The splits will be store in "/ai-team-shared-5/users/bahar/training/splits"
- You can find the  benchmark data (test.parquet) in "/ai-team-shared-5/users/bahar/benchmark/data"

**You can change the paths using the config files**






