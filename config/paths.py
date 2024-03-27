

def get_paths(machine, keys=["EXPBASEPATH", "DATABASEPATH", "DATACOLLECTIONPATH", "MODELCOLLECTIONPATH", "LLMSCRIPTPATH", "LLMMODELPATH"]):
    """
    EXPBASEPATH: base path to save experiements
    DATABASEPATH: base path of datasets. Should contain '/benchmark'
    DATACOLLECTIONPATH: path to data_config_evaluation.yaml
    MODELCOLLECTIONPATH: path to model_collections.yaml
    LLMSCRIPTPATH: for open source llms, path to train_bash.py
    LLMMODELPATH: for open source llms run locally, path to model checkpoints.
    RAWJSONPATH: for generating dataset, path to the base directory of raw json files
    LOGBASEPATH: base path to log openai cost
    """
    if machine == 'machine':
        EXPBASEPATH = None # [REQUIRED]
        DATABASEPATH = None # [REQUIRED]
        DATACOLLECTIONPATH = None # [REQUIRED]
        MODELCOLLECTIONPATH = None # [REQUIRED]
        LLMSCRIPTPATH = None # [REQUIRED]
        RAWJSONPATH = None # [REQUIRED]
        LLMMODELPATH = None # [REQUIRED]

    
    path_mapping = {
    "EXPBASEPATH": EXPBASEPATH,
    "DATABASEPATH": DATABASEPATH, 
    "DATACOLLECTIONPATH": DATACOLLECTIONPATH, 
    "MODELCOLLECTIONPATH": MODELCOLLECTIONPATH,
    "LLMSCRIPTPATH": LLMSCRIPTPATH,
    "LLMMODELPATH": LLMMODELPATH,
    "RAWJSONPATH": RAWJSONPATH,
    }
    
    tar_paths = [path_mapping[k] for k in keys]
    return tar_paths
