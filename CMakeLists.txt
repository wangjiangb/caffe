PROJECT( Caffe )
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

#=================Version======================
SET( CAFFE_VERSION_MAJOR 0 )
SET( CAFFE_VERSION_MINOR 1 )
SET( CAFFE_VERSION_PATCH 0 )
SET( CAFFE_VERSION_STRING "${CAFFE_VERSION_MAJOR}.${CAFFE_VERSION_MINOR}.${CAFFE_VERSION_PATCH}" )

#==============Set Directories=================
SET( CAFFE_SOURCE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
SET( include_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" )
SET( example_DIR "${CMAKE_CURRENT_BINARY_DIR}/examples" )
SET( test_DIR "${CMAKE_CURRENT_BINARY_DIR}/src/gtests" )

SET( EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" )
SET( LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib" )

#=================Options=======================
OPTION( DEBUG_MODE
        "If on, will build everything with debug flags."
        ON )
OPTION( DO_TESTING
        "If on, will make unit tests."
        ON )
OPTION( DO_PROFILING
        "If on, and using g++, will perform profiling."
        OFF )

#==============Custom Cmake Modules=============
SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules" )

#==============Packages=========================
#FIND_PACKAGE( Boost REQUIRED )
FIND_PACKAGE( CUDA REQUIRED )
FIND_PACKAGE( MKL REQUIRED )
FIND_PACKAGE( Protobuf REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED)
FIND_PACKAGE(GLog REQUIRED)
FIND_PACKAGE(LevelDB REQUIRED)
FIND_PACKAGE(GTest REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -Xcompiler -fPIC -O2 -use_fast_math)

#==============Additional Include and Library=========================
SET(ADDITIONAL_INCLUDE 
    ${GTEST_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIRS}
    ${LevelDB_INCLUDE_PATH}
    ${MKL_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})

SET(ADDITIONAL_LIBRARY
  ${CMAKE_THREAD_LIBS_INIT}
  ${OPENCV_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${LevelDB_LIBRARY}
  ${LIB_MKL_RT}
  ${LIB_PTHREAD}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${Boost_LIBRARIES}
  -lsnappy
  -lopencv_highgui
  -lopencv_core
  -lopencv_imgproc)

#===========Set release/debug settings==========
IF( CMAKE_COMPILER_IS_GNUCXX )
   SET( CMAKE_CXX_FLAGS_DEBUG "-g3 -Wall -Wno-sign-compare" )
   SET( LINK_FLAGS_DEBUG "-rdynamic" )
   SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wno-deprecated -Wno-sign-compare" )
   SET( LINK_FLAGS_RELEASE "-rdynamic" )
   IF( DO_PROFILING )
      SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg" )
      SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg" )
   ENDIF()
ENDIF()

#============Add Include directories============
INCLUDE_DIRECTORIES( ${include_DIR} ${ADDITIONAL_INCLUDE})


#=============Google Test Main===================
#ADD_SUBDIRECTORY("src/gtest")

#=============Process Subdirectories============
ADD_SUBDIRECTORY( "src/caffe" )
#ADD_SUBDIRECTORY( "src/test" )
ADD_SUBDIRECTORY( "examples/" )
