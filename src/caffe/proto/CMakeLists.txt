SET(PROTO_FILES
"caffe.proto"
)


PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${PROTO_FILES} )

ADD_LIBRARY(CaffeProto SHARED ${PROTO_SRC} )
target_include_directories(CaffeProto PUBLIC ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(CaffeProto ${PROTOBUF_LIBRARY})
