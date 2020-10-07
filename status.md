# Status
- LIME library requests all pertubations at once which exceeds maximum REST payload of the webservice
- own server has to be running (minor changes were necessary)


# Input needed
- Pertubations count?
- output format html?
- machine available?


# Problem #1

- SP API lässt pertubation sample size nicht umstellen, ist default auf 5000
- single explanation with 10 pertubations: 14s CPU, 
- single explanation with 5000 pertubations: 6532s CPU, 348s GPU

