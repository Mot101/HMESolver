CXX      = g++
#CXXFLAGS = -Wall -fsanitize=address -g
CXXFLAGS = -O3 -march=native -DNDEBUG
INCLUDES = -I./include

SRC_train  = src/CNN.cpp src/ConvolutionLayer.cpp src/Pool.cpp src/FC.cpp src/DropOutLayer.cpp src/train.cpp src/read_mnist.cpp src/ReluLayer.cpp src/read_jpg.cpp src/symbols.cpp src/SoftmaxLayer.cpp
SRC_extractor = src/CNN.cpp src/ConvolutionLayer.cpp src/Pool.cpp src/FC.cpp src/DropOutLayer.cpp src/extractor.cpp src/read_mnist.cpp src/ReluLayer.cpp src/read_jpg.cpp src/symbols.cpp src/SoftmaxLayer.cpp
SRC_metrics = src/CNN.cpp src/ConvolutionLayer.cpp src/Pool.cpp src/FC.cpp src/DropOutLayer.cpp src/metrics.cpp src/read_mnist.cpp src/ReluLayer.cpp src/read_jpg.cpp src/symbols.cpp src/SoftmaxLayer.cpp
SRC_solver = src/CNN.cpp src/ConvolutionLayer.cpp src/Pool.cpp src/FC.cpp src/DropOutLayer.cpp src/expression_tree.cpp src/read_mnist.cpp src/ReluLayer.cpp src/read_jpg.cpp src/symbols.cpp src/SoftmaxLayer.cpp
OBJ_train  = $(SRC_train:.cpp=.o)
OBJ_extractor = $(SRC_extractor:.cpp=.o)
OBJ_metrics = $(SRC_metrics:.cpp=.o)
OBJ_solver = $(SRC_solver:.cpp=.o)

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

TARGET_TRAIN = cnn_train
TARGET_EXTRACTOR = cnn_extractor
TARGET_METRICS = cnn_metrics
TARGET_SOLVER = cnn_solver

# create folders weights and symbols if they don't exist

TARGET_FOLDERS = weights symbols

all: $(TARGET_TRAIN) $(TARGET_EXTRACTOR) $(TARGET_METRICS) $(TARGET_SOLVER) $(TARGET_FOLDERS)

$(TARGET_TRAIN): $(OBJ_train)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(TARGET_EXTRACTOR): $(OBJ_extractor)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(TARGET_METRICS): $(OBJ_metrics)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(TARGET_SOLVER): $(OBJ_solver)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(TARGET_FOLDERS):
	mkdir -p weights
	mkdir -p symbols

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_CFLAGS) -c $< -o $@

# clean only symbols folder

clean_symbols:
	rm -rf symbols/*

clean_weights:
	rm -rf weights/*	

clean:
	rm -f $(OBJ_train) $(OBJ_extractor) $(OBJ_metrics) $(OBJ_solver) $(TARGET_TRAIN) $(TARGET_EXTRACTOR) $(TARGET_METRICS) $(TARGET_SOLVER) 


