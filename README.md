# Experimental data for *"Pupil response anisotropy of the peripheral visual field by the glare illusion"*
Copyright 2021 Yuta Suzuki

### Article information
Istiqomah, N., Suzuki, Y., Kinzuka, Y., Minami, T. & Nakauchi, S. Pupil response anisotropy of the peripheral visual field by the glare illusion. in prep.

## Requirements
Python
- pre-peocessing (https://github.com/suzuki970/PupilAnalysisToolbox)
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
- *‘figure.Rmd’* is to generate figures and statistical results.
