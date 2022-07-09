# Experimental data for *"Anisotropy in the peripheral visual field based on pupil response to the glare illusion"*
Copyright 2022 Yuta Suzuki

### Article information
Istiqomah, N., Suzuki, Y., Kinzuka, Y., Minami, T. & Nakauchi, S. Anisotropy in the peripheral visual field based on pupil response to the glare illusion. Heliyon e09772 (2022) [doi:10.1016/j.heliyon.2022.e09772].

[doi:10.1016/j.heliyon.2022.e09772]: https://doi.org/10.1016/j.heliyon.2022.e09772

## Requirements
Python
- pre-processing (https://github.com/suzuki970/PupilAnalysisToolbox)
- numpy
- scipy
- os
- json

R
- library(rjson)
- library(ggplot2)
- library(ggpubr)
- library(Cairo)
- library(gridExtra)
- library(effsize)
- library(BayesFactor)
- library(rjson)
- library(reshape)
- library(lme4)
- library(permutes)

## Raw data
raw data can be found at **'[Python]PreProcessing/results'**

## Pre-processing
- Raw data (.asc) are pre-processed by **'[Python]PreProcessing/parseData.py'**

	- Pre- processed data is saved as **‘data_original.json’**

- Artifact rejection and data epoch are performed by **'[Python]PreProcessing/dataAnalysis.py'**

```
>> python parseData.py	
>> python dataAnalysis.py
```


## Figure and statistics
-  **'[Rmd]Results/figure.Rmd'**‘ is to generate figures and statistical results.
