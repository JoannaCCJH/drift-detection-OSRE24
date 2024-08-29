# drift-detection-OSRE24

## Overview
Drift, including both data drift and concept drift will affect the model's performance over time.  This project focuses specifically on data drift (feature drift), aiming to design a pipeline for evaluating drift detection algorithms on system traces. The project is divided into two main parts: dataset construction and algorithm benchmarking.

**[PART 1: Dataset Construction]**
I constructed a data drift dataset using the Tencent I/O block trace, which includes both drift and non-drift labels. The raw traces were processed into timestamps with various features. To detect where drift happens and construct the datasets, I employed several offline drift detection algorithms, including Kolmogorov-Smirnov, Cramer-von Mises, KL-Divergence, and Jensen-Shannon Distance. A voting strategy, along with post-processing techniques, was applied to build and clean the datasets. Given that trace data often contains noise, which can affect the accuracy of drift detection, additional preprocessing steps such as fourier transform and moving average were taken to mitigate this issue.

**[PART 2: Benchmark Drift Detection Algorithms]**

For more specific details about the project, please refer to the final blog attached at the end of this page.


## Project Structure

```bash
project-root/
│
├── config/
│   ├── config.yaml            # configuration for dataset construction
│
├── drifts/                    # implementation of dataset construction code and 
│
├── online-drift-detection-benchmark/ 
│
├── output/                    # labeled dataset output
│   ├── 1063_1m_winlen_60_stepsize_30_none
│   ├── ...
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