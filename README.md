# TabIS

Official code and data for paper "Uncovering Limitations of Large Language Models in Information Seeking from Tables" (under reviewing).

## Data
Our data is available here: [link](https://drive.google.com/file/d/1bucVa_z4h71Secd5B4ImEZSlnvJUf0j-/view?usp=sharing).


## Getting Started

### Config Preparation
Please complete the following config files before conducting the evaluation.

```
config/model_collections.yaml      config the models used to eval the datasets.
config/paths.py                    config some important path, such as model path.
```


### Run Evaluation
```
python tab_benchmark/eval_benchmark.py 
                         --task_name mc-sis \
                         --eval_tasks SIS \
                         --eval_models llama2-7b-chat \
                         --datasets mc-totto/mc-hitab \
                         --spec_names all \
                         --max_data_num 4000 \
                         --pool_num 10 \
                         --gpu_devices 0,1,2,3,4,5,6,7 \
                         --shot one \
                         --serialization md \
                         --seed 0 \
                         --machine L5 \
                         --eval_group_key option_types
```
