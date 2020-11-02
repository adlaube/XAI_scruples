# XAI_scruples

- max payload of the web server is too restrictive
- own server has to be running (minor changes were necessary)

## Next steps
- LIME pertubations dependent on word count
- create set of LIME explanations
- ?

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


### Anchors

- runtime weirdly high


### SHAP

- not compatible with embedding layers: https://github.com/slundberg/shap/issues/595



