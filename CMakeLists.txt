cmake_minimum_required(VERSION 3.28)
project(TensorFlow_ProofOfConcept)

set(CMAKE_CXX_STANDARD 26)


include_directories(include)
include_directories("/home/physiker/.cache/bazel/_bazel_physiker/6edee2aa40659590dd6c942b17b1bc5a/execroot/org_tensorflow/bazel-out/k8-opt/bin")

add_subdirectory(source)
add_subdirectory(test)