# TabIS

Official code and data for paper "Uncovering Limitations of Large Language Models in Information Seeking from Tables" (Findings of ACL 2024).

## Log
```
2024/06/11  Update data with improved quality & evaluation code
2024/06/12  Update results of Llama3 on B-TIS
```

## Data
Our data is available here: [link](https://drive.google.com/file/d/1MFwMTHgxOTh7QFFRQ3cszoMCFxhNv9ui/view?usp=sharing).

### Data Format
- **prompt**: The input prompt used to test LLMs; all prompts are in a single-choice question format with two options (one-shot).
- **response**: The standard answer, represented as either "A" or "B".
- **option_types**: The strategy used to generate the incorrect option, categorized as either "mod-input", "mod-answer", "exam-judge", or "human-labeled".
- **reasoning**: The reasoning path of LLMs when generating the incorrect options, specifically distinguishing the option for "exam-judge".
- **components**: Sub-components of the prompt for data reproduction.

## Evaluation

### Config Preparation
Please complete the following config files before conducting the evaluation.

```
config/model_collections.yaml      config the model path (huggingface) for evaluation.
```

To access OpenAI APIs, please specify your key:
```
export OPENAI_KEY="YOUR_OPENAI_KEY"
```


### Run Evaluation
```
python eval_benchmark.py --exp_path /path/to/exp \
                         --eval_models gpt-3.5-turbo gpt-4-1106-preview \
                         --dataset_path /path/to/datasets \
                         --max_data_num 5000 \
                         --pool_num 10 \
                         --gpu_devices 4,5,6,7 \
                         --max_memory_per_device 10 \
                         --seed 0 \
                         --eval_group_key option_types
```
Please put the test datasets (xxx.json) under `dataset_path`.


## Results on B-TIS

<table>
    <tr>
        <th rowspan="2" class="center-text">Model</th>
        <th colspan="5" class="center-text">ToTTo</th>
        <th colspan="5" class="center-text">HiTab</th>
    </tr>
    <tr>
        <th>EJ</th>
        <th>MI</th>
        <th>MO</th>
        <th>HA</th>
        <th>Avg.</th>
        <th>EJ</th>
        <th>MI</th>
        <th>MO</th>
        <th>HA</th>
        <th>Avg.</th>
    </tr>
    <tr>
        <td colspan="11"><b>proprietary model</b></td>
    </tr>
    <tr>
        <td>Gemini-pro</td>
        <td>70.2</td>
        <td>93.3</td>
        <td>87.9</td>
        <td>76.9</td>
        <td>85.6</td>
        <td>53.1</td>
        <td>67.6</td>
        <td>79.1</td>
        <td>67.4</td>
        <td>66.6</td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo-instruct</td>
        <td>60.7</td>
        <td>81.8</td>
        <td>80.6</td>
        <td>55.7</td>
        <td>75.1</td>
        <td>62.3</td>
        <td>71.9</td>
        <td>78.4</td>
        <td>45.7</td>
        <td>68.3</td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo-1106</td>
        <td>56.9</td>
        <td>77.8</td>
        <td>76.6</td>
        <td>64.8</td>
        <td>72.1</td>
        <td>42.5</td>
        <td>64.6</td>
        <td>71.3</td>
        <td>48.6</td>
        <td>57.5</td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo-16k</td>
        <td>58.4</td>
        <td>84.5</td>
        <td>82.8</td>
        <td>59.1</td>
        <td>76.7</td>
        <td>48.4</td>
        <td>67.5</td>
        <td>75.4</td>
        <td>43.8</td>
        <td>61.2</td>
    </tr>
    <tr>
        <td>GPT-4-turbo-1106</td>
        <td>79.8</td>
        <td>93.5</td>
        <td>96.4</td>
        <td>85.2</td>
        <td><b>91.2</b></td>
        <td>73.5</td>
        <td>85.2</td>
        <td>91.8</td>
        <td>77.1</td>
        <td><b>82.4</b></td>
    </tr>
    <tr>
        <td colspan="11"><b>open source model</b></td>
    </tr>
    <tr>
        <td>Llama2-7b-chat</td>
        <td>54.3</td>
        <td>52.4</td>
        <td>53.1</td>
        <td>60.2</td>
        <td>53.6</td>
        <td>44.3</td>
        <td>54.8</td>
        <td>47.8</td>
        <td>39.1</td>
        <td>47.8</td>
    </tr>
    <tr>
        <td>TableLlama-7b</td>
        <td>53.2</td>
        <td>54.7</td>
        <td>53.9</td>
        <td>58.0</td>
        <td>54.3</td>
        <td>43.8</td>
        <td>53.3</td>
        <td>48.9</td>
        <td>41.0</td>
        <td>47.7</td>
    </tr>
    <tr>
        <td>Mistral-7b-instruct-v0.2</td>
        <td>52.8</td>
        <td>77.4</td>
        <td>81.0</td>
        <td>70.5</td>
        <td>73.2</td>
        <td>40.9</td>
        <td>63.5</td>
        <td>72.4</td>
        <td>47.6</td>
        <td>56.9</td>
    </tr>
    <tr>
        <td>Llama2-13b-chat</td>
        <td>52.4</td>
        <td>66.7</td>
        <td>66.7</td>
        <td>60.2</td>
        <td>63.3</td>
        <td>45.0</td>
        <td>52.2</td>
        <td>64.8</td>
        <td>53.3</td>
        <td>53.4</td>
    </tr>
    <tr>
        <td>Mixtral-8*7b-instruct</td>
        <td>55.8</td>
        <td>88.7</td>
        <td>88.1</td>
        <td>73.9</td>
        <td>80.6</td>
        <td>51.6</td>
        <td>75.1</td>
        <td>77.1</td>
        <td>52.4</td>
        <td>65.6</td>
    </tr>
    <tr>
        <td>Llama2-70b-chat</td>
        <td>52.1</td>
        <td>70.9</td>
        <td>79.6</td>
        <td>65.9</td>
        <td>70.0</td>
        <td>46.8</td>
        <td>60.0</td>
        <td>68.0</td>
        <td>50.5</td>
        <td>56.9</td>
    </tr>
    <tr>
        <td>Tulu2-70b-DPO</td>
        <td>64.4</td>
        <td>91.7</td>
        <td>93.1</td>
        <td>78.4</td>
        <td>85.7</td>
        <td>55.5</td>
        <td>72.5</td>
        <td>81.4</td>
        <td>61.0</td>
        <td>68.2</td>
    </tr>
    <tr>
        <td>StructLM-7b</td>
        <td>47.6</td>
        <td>68.8</td>
        <td>70.1</td>
        <td>64.8</td>
        <td>64.6</td>
        <td>38.4</td>
        <td>60.3</td>
        <td>57.7</td>
        <td>49.5</td>
        <td>50.8</td>
    </tr>
    <tr>
        <td>StructLM-13b</td>
        <td>57.3</td>
        <td>85.9</td>
        <td>83.0</td>
        <td>70.5</td>
        <td>77.8</td>
        <td>45.0</td>
        <td>68.4</td>
        <td>68.9</td>
        <td>50.5</td>
        <td>58.9</td>
    </tr>
    <tr>
        <td>StructLM-34b</td>
        <td>61.4</td>
        <td>87.1</td>
        <td>86.3</td>
        <td>71.6</td>
        <td>80.4</td>
        <td>45.4</td>
        <td>61.7</td>
        <td>70.2</td>
        <td>52.4</td>
        <td>57.7</td>
    </tr>
    <tr>
        <td>codellama-7b</td>
        <td>47.2</td>
        <td>61.2</td>
        <td>61.2</td>
        <td>56.8</td>
        <td>58.0</td>
        <td>33.6</td>
        <td>59.4</td>
        <td>56.3</td>
        <td>41.0</td>
        <td>47.9</td>
    </tr>
    <tr>
        <td>codellama-13b</td>
        <td>49.4</td>
        <td>58.4</td>
        <td>57.8</td>
        <td>61.4</td>
        <td>56.5</td>
        <td>42.7</td>
        <td>56.2</td>
        <td>52.5</td>
        <td>40.0</td>
        <td>49.0</td>
    </tr>
    <tr>
        <td>codellama-34b</td>
        <td>54.7</td>
        <td>81.8</td>
        <td>81.2</td>
        <td>64.8</td>
        <td>74.8</td>
        <td>44.5</td>
        <td>61.7</td>
        <td>72.1</td>
        <td>44.8</td>
        <td>57.3</td>
    </tr>
    <tr>
        <td>Llama3-8b-instruct</td>
        <td>62.9</td>
        <td>89.4</td>
        <td>92.5</td>
        <td>77.3</td>
        <td>84.3</td>
        <td>55.3</td>
        <td>71.0</td>
        <td>80.9</td>
        <td>58.1</td>
        <td>67.3</td>
    </tr>
    <tr>
        <td>Llama3-70b-instruct</td>
        <td>83.9</td>
        <td>94.0</td>
        <td>96.4</td>
        <td>90.9</td>
        <td><b>92.6</b></td>
        <td>72.4</td>
        <td>83.2</td>
        <td>91.5</td>
        <td>74.3</td>
        <td><b>81.1</b></td>
    </tr>
</table>