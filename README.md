# XAI_scruples

- max payload of the web server is too restrictive
- own server has to be running (minor changes were necessary)

## Next steps
- LIME pertubations dependent on word count
- run SP LIME on subset
- fix anchors
- integrate SHAP
- benchmark and report

## LIME Parameters
|              | runtime influence | default    | proposal        | comment                                                                                            |
|--------------|-------------------|------------|-----------------|----------------------------------------------------------------------------------------------------|
| num_samples  | Y                 | 5000       | 1000 < K < 5000 | Number of pertubations (neighborhood size for linear approx)  guess based on average word count?   |
| top_labels   | Y                 | deactivate | 3               |                                                                                                    |
| num_features | N                 | 10         | 10              | for text classifiers: words                                                                        |
|              |                   |            |                 |                                                                                                    |
|              |                   |            |                 |                                                                                                    |

### Runtime LIME

- single explanation with 10 pertubations: 14s CPU, 
- single explanation with 5000 pertubations: 6532s CPU, 348s GPU


### Problems

- length of texts for anchors + BERT: exceeds max length of 512 tokens, solutions? https://www.thepythoncode.com/article/text-summarization-using-huggingface-transformers-python 
-  increase max length? how does norms handle this in Roberta? config for embeddings 

### Fixed problems

- version conflict anchors and scruples with scikit-learn, check deactivated in setup.py of scruples


