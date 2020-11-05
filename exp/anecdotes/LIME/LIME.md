# LIME anaylsis on 50 sample instances

## Settings:
- number of pertubations as a function of word count: max(word_count^1.2,5000)
- 3 most likely classes are explained
- 10 feature contributions per instance are given


## General Observations
- "I" is always part of the explanation
- LIME does not seem to find "combined words"
- hardly any peaks in feature contributions
- NORMS probabilities are rather conservative even if it is a clear case (based on average, western standards)


|Link | Class | Prediction | Explanation | Comment
|--- | --- | ---|---| --- |
|[69](LIME_69.html) | OTHER | OK | 8 |
|[110](./LIME_110.html) | OTHER | OK | 6 |
|[138](./LIME_138.html) | OTHER | OK | 10 | long story, accurate explanation, abusive relationship
|[146](./LIME_146.html) | AUTHOR | OK | 1 | author cheating but uses different phrasing, no cheating related features in the explanation but model gets it
|[203](./LIME_203.html) | OTHER | OK | 4 | author sort of shames, gets insulted, shame features are in the explanation, insult ones are not
|[204](LIME_204.html) | OTHER | OK | 3 | father-son relationship issues, "gaslighting" has the highest feature contribution for all classes but does not match prediction
|[220](LIME_220.html) | OTHER | NOK | 8 | dog education, 
|[69](LIME_69.html) | OTHER | OK | 8 |
|[69](LIME_69.html) | OTHER | OK | 8 |
|[69](LIME_69.html) | OTHER | OK | 8 |
./LIME_203.html
./LIME_204.html
./LIME_220.html
./LIME_221.html
./LIME_315.html
./LIME_319.html
./LIME_401.html
./LIME_434.html
./LIME_439.html
./LIME_506.html
./LIME_551.html
./LIME_567.html
./LIME_645.html
./LIME_670.html
./LIME_671.html
./LIME_762.html
./LIME_807.html
./LIME_877.html
./LIME_944.html
./LIME_946.html
./LIME_1138.html
./LIME_1240.html
./LIME_1317.html
./LIME_1491.html
./LIME_1509.html
./LIME_1559.html
./LIME_1616.html
./LIME_1636.html
./LIME_1652.html
./LIME_1675.html
./LIME_1851.html
./LIME_1853.html
./LIME_1911.html
./LIME_1949.html
./LIME_2077.html
./LIME_2092.html
./LIME_2212.html
./LIME_2243.html
./LIME_2291.html
./LIME_2304.html
./LIME_2357.html
./LIME_2437.html
./LIME_2470.html
./LIME_2478.html