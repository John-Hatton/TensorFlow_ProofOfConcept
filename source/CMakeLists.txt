file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
)


### Remove this later. Only works on my machine...
include_directories(/mnt/c/devel/sdk/tensorflow)

# Define the imported library
add_library(tensorflow_cc SHARED IMPORTED)

# Specify the location of the shared library
set_property(TARGET tensorflow_cc PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libraries/tensorflow/libtensorflow_cc.so)

add_library(tensorflow_framework SHARED IMPORTED)

# Specify the location of the shared library
set_property(TARGET tensorflow_framework PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libraries/tensorflow/libtensorflow_framework.so)

# Optionally, if your shared library has dependencies, you can also set them
set_property(TARGET tensorflow_cc PROPERTY INTERFACE_LINK_LIBRARIES tensorflow_framework)



target_link_libraries(${PROJECT_NAME} PRIVATE tensorflow_cc)





