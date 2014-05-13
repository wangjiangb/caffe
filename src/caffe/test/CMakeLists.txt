
set(TEST_FILES
"test_blob"
"test_common"
"test_convolution_layer"
"test_data_layer"
"test_euclidean_loss_layer"
"test_filler"
"test_flatten_layer"
"test_im2col_layer"
"test_innerproduct_layer"
"test_lrn_layer"
"test_multinomial_logistic_loss_layer"
"test_neuron_layer"
"test_padding_layer"
"test_platform"
"test_pooling_layer"
"test_protobuf"
"test_softmax_layer"
"test_softmax_with_loss_layer"
"test_stochastic_pooing"
"test_syncedmem"
"test_util_blas"
)

foreach(TEST_FILE ${TEST_FILES})
  add_executable(${TEST_FILE} ${TEST_FILE}.cpp)
  target_link_libraries(${TEST_FILE} Caffe CaffeProto CaffeLayers ${GTEST_LIBRARIES} ${ADDITIONAL_LIBRARY})
  target_include_directories(${TEST_FILE} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/../" "${CMAKE_CURRENT_SOURCE_DIR}/../..")
  add_test( ${TEST_FILE} ${EXECUTABLE_OUTPUT_PATH}/${TEST_FILE})
endforeach(TEST_FILE)
