# Handwritten Mathematical Expression

## 1. Problem statement

Write, using only the built-in C++ libraries, a neural network with which you can solve handwritten mathematical equations.

To solve this problem, our team decided to compare two models: Fully connected and Convolutional neural models. Plus, to work with incoming images, a photo preprocessing unit was added, and to solve the resulting equation, an Expression Tree (Shunting-yard Algorithm + Distributive Property) and a solver based on the Newton-Raphson method were implemented.

## 2. Image preprocessering

Preprocessing uses a third-party library, as this is not our main task. We use _opepncv_

In the preprocessing stage, each symbol image is converted to a unified format before being fed into the model. First, the file is read and converted to grayscale (for inference symbols via _BGR2GRAY_, and for training samples directly with _IMREAD_GRAYSCALE_). Then, background normalization is applied: the average intensity is computed over four corner patches, and if the background is detected as bright (value > 128), the image is inverted to enforce a consistent contrast (“dark symbol on a light background”). After that, the image is resized to a fixed 44×44 resolution (for NN 28×28) while preserving the aspect ratio: a scale factor is computed, resize is applied, and the remaining area is filled with padding whose value is chosen to match the background intensity estimated from the corners. Finally, pixels are converted to float and normalized to the [0, 1] range by dividing by 255, and the image is flattened into a 1D vector of length 44×44 (28×28) for further processing; for training data, a one-hot label vector is also created and the train dataset is shuffled.

## 3. Expression Tree

After the CNN recognizes individual symbols, the resulting equation string is passed to the expression tree module, which converts it into a canonical algebraic form. First, _addExplicitMult_ inserts explicit dot operators where multiplication is implicit (e.g., 2x → 2*x, x(y) → x*(y)). Then _buildTree_ constructs an expression tree using operator precedence and parentheses, while also handling unary minus by encoding it as a dedicated operator. Next, normalize recursively transforms the tree into a _MultiPolynomial_ representation: constants and variables become monomials, addition/subtraction merges coefficients, multiplication performs term-by-term expansion with exponent accumulation, exponentiation is supported only for non-negative integers (reading the exponentiation symbol was not implemented due to the missing character set as datasets), and division is allowed only by a monomial (otherwise an empty polynomial is returned to signal an unsupported case). If an equality sign is present, the left and right sides are normalized and moved to one side (pL - pR = 0): if there are no variables, the code evaluates both sides and checks whether the equality holds; if there is exactly one variable, the solver searches for real roots on [-1000, 1000] using Newton’s method; and if there are multiple variables, it returns the simplified polynomial equation set to zero. (see CNN/expression_tree.cpp)

## 4. NN

TO DO: add info

## 5. CNN

The files related to the CNN solver are located in the CNN folder. The header and source files are located in CNN/include and CNN/src, the executable files are in CNN

### 5.1. Architecture

A scheme similar to LeNet-5 was implemented. ReLU is selected as the activation functions between the layers. Weights in Convolution Layer and FC are initialized using the Kaiming algorithm

### 5.2 Implementation

The skeleton of the base layer has been created - BaseLayer.h. All the following layers are inheritors of this class. 
The source files have the structure *layer name*.cpp, for example ConvolutionLayer.cpp (see CNN/src). 
Each layer has vectors for storing gradients, the last input (in layers where it is necessary to use last input for backward pass)/output; variables for storing input and output vector sizes (the code is implemented on one-dimensional vectors, but all calculations assume that we have 3d vectors, so 3 variables are stored).
The following methods are implemented for each layer:
- forward / backward pass
- zero_gradients for zeroing layer gradient vectors
- get_output_size() - returns a vector of output dimensions of the output layer
(in layers where there are weights)
- load/save/update weights/filters/biases - to load/save/update weights so as not to train every time. Update to update the weights during the backward pass

