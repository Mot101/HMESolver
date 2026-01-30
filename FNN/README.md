
# Fully Connected Neural Network (FNN) 

## Overview
This notebook documents the structure, design decisions, and execution steps of the **Fully Connected Neural Network (FNN)** project for handwritten mathematical symbol recognition and solving.

Due to the complexity of handling both **polynomial solving** and **general mathematical expression evaluation**, the project was split into two independent pipelines.


## Project Structure

```
FNN/
 ├── FNN/
 └── FNN1/
```

- **FNN**: Polynomial recognition and solving  
- **FNN1**: Mathematical expression evaluation  

This separation improves stability, clarity, and debugging.



## Execution Environment

- Ubuntu (Linux)
- Tested using Windows Subsystem for Linux (WSL)
- C++17 compatible compiler (`g++`)
- OpenCV 4


## Running the Polynomial Solver (FNN)

### Step 1: Navigate to the folder
```bash
cd /mnt/c/Users/admin/Downloads/FNN/FNN
```

### Step 2: Compile
```bash
g++ -std=c++17 -O2 main.cpp Preprocessing.cpp Extraction.cpp Calculator.cpp Network.cpp PolySolver.cpp `pkg-config --cflags --libs opencv4` -o pipeline
```

### Step 3: Run
```bash
./pipeline
```


## Input Image Selection

At the beginning of `main.cpp`, input images are defined as:

```cpp
vector<string> exprCandidates = {
    "data/test_images/expr2.png",
    "data/test_images/expr2.jpg",
    "data/test_images/expr2.jpeg"
};
```

You may modify this list to test different images.



## Test Images Directory

All test images must be placed in:

```
data/test_images/
```


## Debugging and Intermediate Results

During execution, the pipeline creates a directory:

```
debug_patches/
```

This directory contains:
- Preprocessed images and also extracted symbol patches

These outputs helped us to verify preprocessing and extraction quality.


## Running the Calculating Mathematical Expressions (FNN)

### Step 1: Navigate to the folder
```bash
cd /mnt/c/Users/admin/Downloads/FNN/FNN1
```

### Step 2: Compile
```bash
g++ -std=c++17 -O2 main1.cpp Preprocessing1.cpp Extraction1.cpp Calculator1.cpp Network1.cpp `pkg-config --cflags --libs opencv4` -o pipeline
```

### Step 3: Run
```bash
./pipeline
```


## Input Image Selection

At the beginning of `main.cpp`, input images are defined as:

```cpp
vector<string> exprCandidates = {
    "data/test_images/expr1.png",
    "data/test_images/expr1.jpg",
    "data/test_images/expr1.jpeg"
};
```

You may modify this list to test different images.


## Test Images Directory

Same for FNN1, all test images must be placed in:

```
data/test_images/
```
