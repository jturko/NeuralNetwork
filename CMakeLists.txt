cmake_minimum_required(VERSION 3.5.2)

# set the project name
project(NeuralNetwork VERSION 1.0)

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# add library 
#add_library(nn SHARED  ${sources})

# add the executable
add_executable(NeuralNetwork ${PROJECT_SOURCE_DIR}/main.cc ${sources} ${headers})

