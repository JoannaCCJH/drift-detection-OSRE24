# drift-detection-OSRE24

## Overview
Drift including both data drift and concept drift will affect the model's performance over time. This project aims to focus on data drift (or feature drift), specifically analyzing  
The goal is to design a pipeline to evaluate drift detection algorithms on system traces.  The project is mainly divided into two parts: dataset construction and algorithm benchmark. 

[PART 1: Dataset Construction] I constructed a data drift dataset from the Tencent I/O block trace that includes both drift and non drift data. The raw traces are processed into timestamps with different features. Several off-line drift detection algorithms including Kolmogorov-Smirnov, Cramer-von Mises, KL-Divergence, and Jenson-Shannon Distance are used to. A voting strategy and some post-processing tracks are used to clean the datasets as well. Since trace data contains noise, which will impact the drift detection ability, I also implement 

[PART 2: Benchmark Drift Detection Algorithms]

For more specific details about the project, please refer to the final blog attached to the end of the page.

## Project Structure

```bash
project-root/
│
├── config/
│   ├── config.yaml            # configuration for dataset construction
│
├── drifts/                    # implementation of dataset constrcution code
│
├── online-drift-detection-benchmark/ 
│
├── output/                    # 
│   ├── 1063_1m_winlen_60_stepsize_30_none
│
├── generate_dataset.py        # run this script to generate labeled dataset
```

## Getting Started

To generate labeled dataset, please following the steps below:
1. configure the setting `config/config.yaml`
2. run `python generate_dataset.py`


## Links
[Trovi Artifact]

[Final Blog]