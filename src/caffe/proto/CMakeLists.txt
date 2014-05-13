SET(PROTO_FILES
"caffe.proto"
)


PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${PROTO_FILES} )

ADD_LIBRARY(CaffeProto ${PROTO_SRC} )

target_link_libraries(CaffeProto ${PROTOBUF_LIBRARY})