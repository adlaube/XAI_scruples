# LIME anaylsis on 50 sample instances

## Settings:
- number of pertubations as a function of word count: max(word_count^1.2,5000)
- all 5 classes are explained
- 10 feature contributions per instance are given


## General Observations
- personal pronouns are always part of the explanations
- feature contributions of single words are high rather than features that rely on context
- never, always are often part of the explanation [877](LIME_877.html)
- hardly any peaks in feature contributions
- feeling related features contribute towards NOBODY (only empathetic parties - no one the asshole?) [944](LIME_944.html)
- family, relationships or pregnancies are almost always part of the explanation, abbreviations are understoof (gf, ex) [138](LIME_138.html) & [1138](LIME_1138.html)
- number of pertubations is sufficient also for long stories