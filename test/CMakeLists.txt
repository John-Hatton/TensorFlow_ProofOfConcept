project(Google_Tests)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(TensorFlow_ProofOfConcept_Tests
        ${SOURCE_FILES}

        # Where .cpp files from your project go

)

target_link_libraries(TensorFlow_ProofOfConcept_Tests gtest gtest_main)

#
##
#
########
#
##
#

###################
#					#
#    Google Test	#
#					#
###################



include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#
##
#
########
#
##
#