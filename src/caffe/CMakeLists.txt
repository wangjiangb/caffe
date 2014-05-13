file(GLOB CUDA_LAYER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cu")

file(GLOB CPP_LAYER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cpp")

SET( SRC_FILES
   ${CPP_LAYER_FILES}
   "blob.cpp"
   "common.cpp"
   "net.cpp"
   "syncedmem.cpp"
   "layer_factory.cpp"
   "solver.cpp"
   "util/io.cpp"
   "util/im2col.cpp"
   "util/math_functions.cpp"
)


SET(CUDA_FILES
${CUDA_LAYER_FILES}
"util/im2col.cu"
"util/math_functions.cu"
)


ADD_SUBDIRECTORY( "proto/" )

#----Including the protobuffer header----------
CUDA_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/../"
${PROTOBUF_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
CUDA_ADD_LIBRARY(CaffeLayers  ${CUDA_FILES})
target_link_libraries(CaffeLayers CaffeProto)

ADD_LIBRARY(Caffe ${SRC_FILES})
target_include_directories(Caffe PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/../"
${PROTOBUF_INCLUDE_DIR})
target_link_libraries(Caffe CaffeProto CaffeLayers)

INSTALL(
   TARGETS Caffe
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
)
