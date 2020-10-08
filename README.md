# XAI_scruples

- LIME library requests all pertubations at once which exceeds maximum REST payload of the webservice
- own server has to be running (minor changes were necessary)


# Input needed
- Pertubations count?
- output format html? ID in file name, LIME does not store IDs
- machine available?

# Parameters
|              | runtime influence | default    | proposal        | comment                                                                                            |
|--------------|-------------------|------------|-----------------|----------------------------------------------------------------------------------------------------|
| num_samples  | Y                 | 5000       | 1000 < K < 5000 | Number of pertubations (neighborhood size for linear approx)  guess based on average word count?   |
| top_labels   | Y                 | deactivate | 3               |                                                                                                    |
| num_features | N                 | 10         | 10              | for text classifiers: words                                                                        |
|              |                   |            |                 |                                                                                                    |
|              |                   |            |                 |                                                                                                    |

# Problem #1

- SP API lässt pertubation sample size nicht umstellen, ist default auf 5000
- single explanation with 10 pertubations: 14s CPU, 
- single explanation with 5000 pertubations: 6532s CPU, 348s GPU