# Handwritten Mathematical Expression

## 1. Problem statement

Write, using only the built-in C++ libraries, a neural network with which you can solve handwritten mathematical equations.

To solve this problem, our team decided to compare two models: Fully connected and Convolutional neural models. Plus, to work with incoming images, a photo preprocessing unit was added, and to solve the resulting equation, an Expression Tree (Shunting-yard Algorithm + Distributive Property) and a solver based on the Newton-Raphson method were implemented.

## 2. Dataset

we looked at several datasets available on the Internet and mixed 3 datasets:
- MNIST is a classic dataset of numbers 0-9
- https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/data - Contains more than 100,000 images, although the quality of some characters raises questions
- https://github.com/wblachowski/bhmsds/tree/master - 27,000 images of 18 basic characters in high quality 
Using these datasets, we have compiled a total of 18 characters.: ( ) 0 1 2 3 4 5 6 7 8 9 = * / - + x, for each 8000 train/2000 test character (with the exception of * and /). This dataset allowed us to achieve good model accuracy. To evaluate the results, a separate validation folder (100 samples per class) has been created for symbols that are not included in these 3 datasets, collected from different people.

## 3. Image preprocessering

Preprocessing uses a third-party library, as this is not our main task. We use _opepncv_

In the preprocessing stage, each symbol image is converted to a unified format before being fed into the model. First, the file is read and converted to grayscale (for inference symbols via _BGR2GRAY_, and for training samples directly with _IMREAD_GRAYSCALE_). Then, background normalization is applied: the average intensity is computed over four corner patches, and if the background is detected as bright (value > 128), the image is inverted to enforce a consistent contrast (“dark symbol on a light background”). After that, the image is resized to a fixed 44×44 resolution (for NN 28×28) while preserving the aspect ratio: a scale factor is computed, resize is applied, and the remaining area is filled with padding whose value is chosen to match the background intensity estimated from the corners. Finally, pixels are converted to float and normalized to the [0, 1] range by dividing by 255, and the image is flattened into a 1D vector of length 44×44 (28×28) for further processing; for training data, a one-hot label vector is also created and the train dataset is shuffled.

## 4. Expression Tree

After the CNN recognizes individual symbols, the resulting equation string is passed to the expression tree module, which converts it into a canonical algebraic form. First, _addExplicitMult_ inserts explicit dot operators where multiplication is implicit (e.g., 2x → 2*x, x(y) → x*(y)). Then _buildTree_ constructs an expression tree using operator precedence and parentheses, while also handling unary minus by encoding it as a dedicated operator. Next, normalize recursively transforms the tree into a _MultiPolynomial_ representation: constants and variables become monomials, addition/subtraction merges coefficients, multiplication performs term-by-term expansion with exponent accumulation, exponentiation is supported only for non-negative integers (reading the exponentiation symbol was not implemented due to the missing character set as datasets), and division is allowed only by a monomial (otherwise an empty polynomial is returned to signal an unsupported case). If an equality sign is present, the left and right sides are normalized and moved to one side (pL - pR = 0): if there are no variables, the code evaluates both sides and checks whether the equality holds; if there is exactly one variable, the solver searches for real roots on [-1000, 1000] using Newton’s method; and if there are multiple variables, it returns the simplified polynomial equation set to zero. (see [CNN/src/expression_tree.cpp](./CNN/src/expression_tree.cpp)

## 5. NN

The full description can be found [here](./FNN/README.md)

The implementation is split into two independent pipelines.

### 5.1. Project Structure
- `FNN/`  — polynomial recognition and solving
- `FNN1/` — mathematical expression evaluation

### 5.2. Requirements
- Linux (tested on WSL)
- C++17 compiler (`g++`)
- OpenCV 4

### 5.3. Running the Polynomial Solver (FNN)

##### Step 1: Navigate to the folder
```bash
cd /mnt/c/Users/admin/Downloads/FNN/FNN
```

##### Step 2: Compile
```bash
g++ -std=c++17 -O2 main.cpp Preprocessing.cpp Extraction.cpp Calculator.cpp Network.cpp PolySolver.cpp `pkg-config --cflags --libs opencv4` -o pipeline
```

##### Step 3: Run
```bash
./pipeline
```


#### 5.3.1. Input Image Selection

At the beginning of `main.cpp`, input images are defined as:

```cpp
vector<string> exprCandidates = {
    "data/test_images/expr2.png",
    "data/test_images/expr2.jpg",
    "data/test_images/expr2.jpeg"
};
```

You may modify this list to test different images.



#### 5.3.2 Test Images Directory

All test images must be placed in:

```
data/test_images/
```


#### 5.3.3. Debugging and Intermediate Results

During execution, the pipeline creates a directory:

```
debug_patches/
```

This directory contains:
- Preprocessed images and also extracted symbol patches

These outputs helped us to verify preprocessing and extraction quality.


### 5.4. Running the Calculating Mathematical Expressions (FNN)


##### Step 1: Navigate to the folder
```bash
cd /mnt/c/Users/admin/Downloads/FNN/FNN1
```

##### Step 2: Compile
```bash
g++ -std=c++17 -O2 main1.cpp Preprocessing1.cpp Extraction1.cpp Calculator1.cpp Network1.cpp `pkg-config --cflags --libs opencv4` -o pipeline
```

##### Step 3: Run
```bash
./pipeline
```


#### 5.4.1. Input Image Selection

At the beginning of `main.cpp`, input images are defined as:

```cpp
vector<string> exprCandidates = {
    "data/test_images/expr1.png",
    "data/test_images/expr1.jpg",
    "data/test_images/expr1.jpeg"
};
```

You may modify this list to test different images.



## 6. CNN
(The entire discussion in this paragraph is about the CNN/project folder)

The files related to the CNN solver are located in the CNN folder. The header and source files are located in [CNN/include](./CNN/include) and [CNN/src](./CNN/src), the executable files are in CNN

### 6.1. Architecture

A scheme similar to LeNet-5 was implemented. ReLU is selected as the activation functions between the layers. Weights in Convolution Layer and FC are initialized using the Kaiming algorithm

### 6.2 Implementation

The skeleton of the base layer has been created - BaseLayer.h. All the following layers are inheritors of this class. 
The source files have the structure *layer name*.cpp, for example ConvolutionLayer.cpp (see CNN/src). 
Each layer has vectors for storing gradients, the last input (in layers where it is necessary to use last input for backward pass)/output; variables for storing input and output vector sizes (the code is implemented on one-dimensional vectors, but all calculations assume that we have 3d vectors, so 3 variables are stored).
The following methods are implemented for each layer:
- forward / backward pass
- zero_gradients for zeroing layer gradient vectors
- get_output_size() - returns a vector of output dimensions of the output layer
(in layers where there are weights)
- load/save/update weights/filters/biases - to load/save/update weights so as not to train every time. Update to update the weights during the backward pass

Implemented layers: Convolution, ReLU, Pool, FC, Softmax, DropOut (not used in the final version)
The model is controlled by the CNN class (stored in [CNN.cpp](./CNN/src/CNN.cpp) ). CNN class methods:
- add_layer(BaseLayer* layer) — adds a layer to the architecture (layers vector) in the data-flow order.
- forward(input, train) — forward pass: applies layer->forward sequentially for all layers, avoiding unnecessary copies.
- backward(d_out) — backward pass: propagates gradients from the last layer to the first via layer->backward.
- train_step(...) — one training step on a batch: zeros gradients, sets targets for the last layer, accumulates loss, runs backprop, and updates parameters with l2_reg.
- train(...) — training loop over epochs and batches: optionally shuffles data, halves the learning rate every 3 epochs, logs loss and metrics.
- predict(input, train) — inference: a wrapper around forward.
- get_output_size() — returns the output size of the last layer.
- calculate_test_loss(inputs, targets, train_flag) — computes total loss on a dataset without updating weights.
- evaluate(images, labels, train_flag) — computes accuracy using argmax predictions and one-hot labels.
- predict_classes(images, class_names, train_flag) — returns predicted class names for a batch of images.
- predict_pairs(images, true_labels, class_names, train_flag) — returns pairs (true class, predicted class).
- ~CNN_model() — frees memory by deleting all layers stored in layers.

Implemented additional functions: 
- read_jpg - processing incoming images in jpg/jped format
- read_mnist (not used) - processing mnist dataset
- symbols.cpp - detection and allocation of characters into separate symbols_*number* files.jpg to the symbols folder

Executable files:
- train - model training. Inside the assembly of our architecture (Convolution (6,5,1,2,44,44,1) -> ReLU -> Pool(2,2) -> Convolution(16,5,1,2) -> ReLU -> Pool(2,2) -> FC(120) -> ReLU -> FC(84) -> ReLU -> FC(18) -> Softmax ) and saving the weights to the weights folder/
Usage: ./cnn_train <train_data_dir> <test_data_dir> <epochs> <batch_size> <learning_rate> <l2_reg> <0/1 for the record training_log.txt>
- expression_tree.cpp - a complete solution cycle using weights from weights 
Usage: ./cnn_solver <image_path>
- extractor.cpp - Prediction of the input equation and output of the predicted equation as a string
Usage: ./cnn_extractor
- metrics.cpp - Calculation of confusion matrix and F1 per class based on validation/ saved in csv files
Usage: ./cnn_metrics

## 7. Limatations 

This version of the program assumes that the input is given equations in 1 line, without stuck symbols / symbols with a break / special mathematical symbols (including fractions in the form of a column)
The CNN folder contains examples of equations (example_1-4.jpg) that we can solve.

## 8. Report 

The report, along with a literature review and a more detailed description of the project, is available in this file[./Report.pdf].

## 9. Roles in the team

- Ayauly Chayakova and Maram Marsaoui - Implementation of the NN 
- Abeer Obeid - Study of various approaches to solving the problem, writing a report, preparing a dataset
- Le Quyi - Implementation of the Expression Tree module
- Matvei Kudashev - Implementation of the CNN
