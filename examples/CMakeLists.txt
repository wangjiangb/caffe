#SET(CUDA_LIB_DIRS
#  "${CUDA_SDK_ROOT_DIR}/lib/x86_64-linux-gnu")

#LINK_DIRECTORIES(${CUDA_LIB_DIRS})

SET(MORE_LIBS
${ADDITIONAL_LIBRARY}
)

SET(MORE_INCS
)

SET(EXAMPLE_FILES
  "convert_cifar_data"
  "convert_imageset"
  "convert_mnist_data"
  "demo_compute_image_mean"
  "dump_network"
  "evaluate_network"
  "finetune_net"
  "net_speed_benchmark"
  "quantize_network"
  "test_net"
  "train_net"
)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
  add_executable(${EXAMPLE_FILE} ${EXAMPLE_FILE}.cpp)
  target_link_libraries(${EXAMPLE_FILE} Caffe CaffeProto CaffeLayers ${MORE_LIBS})
  target_include_directories(${EXAMPLE_FILE} PUBLIC "${MORE_INCS}")
endforeach(EXAMPLE_FILE)
