# CUDA-based Link Prediction in Large-Scale Graphs

This repository contains the source code and additional resources for the research paper "Leveraging CUDA for Local Similarity-Based Link Prediction in Large-Scale Graphs" by Ahmad Al Musawi. The project focuses on enhancing the performance of link prediction algorithms in large-scale networks using CUDA (Compute Unified Device Architecture) for parallel computing.

## Overview

The project explores the application of CUDA to accelerate the computation of similarity scores between nodes in large-scale graphs. This approach is crucial for efficiently handling expansive datasets commonly found in social networks, biological systems, and other complex network structures.

## Key Features

- **CUDA Optimization:** Implementation of link prediction algorithms optimized with CUDA for improved performance on large-scale graphs.
- **Similarity Metrics:** Several similarity measures, including the common neighbors approach, are implemented and tested.
- **Performance Analysis:** Evaluation of the correlation between speedup, network size, and complexity.
- **Scalable Solutions:** Demonstrates the scalability and effectiveness of parallel processing in graph analytics.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (compatible version with your GPU)
- C++ Compiler
- Python (for comparative Python implementation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/almusawiaf/LinkPredictionInCUDA.git
