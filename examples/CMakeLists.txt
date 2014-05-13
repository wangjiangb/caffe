#SET(CUDA_LIB_DIRS
#  "${CUDA_SDK_ROOT_DIR}/lib/x86_64-linux-gnu")

#LINK_DIRECTORIES(${CUDA_LIB_DIRS})

SET(MORE_LIBS
${CMAKE_THREAD_LIBS_INIT}
${OPENCV_LIBRARIES}
${GLOG_LIBRARIES}
${LevelDB_LIBRARY}
${MKL_LIBRARIES}
-lcudart
-lcublas
-lcurand
-lpthread
-lsnappy
-lopencv_highgui
-lopencv_core
-lopencv_imgproc
-lmkl_rt
-lmkl_core
-lmkl_gnu_thread
${Boost_LIBRARIES}
)

SET(MORE_INCS
${OPENCV_INCLUDE_DIRS}
${LevelDB_INCLUDE_PATH}
${MKL_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)



add_executable(convert_cifar_data convert_cifar_data.cpp)
target_link_libraries(convert_cifar_data Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(convert_cifar_data PUBLIC "${MORE_INCS}")

add_executable(convert_imageset convert_imageset.cpp)
target_link_libraries(convert_imageset Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(convert_imageset PUBLIC "${MORE_INCS}")


add_executable(convert_mnist_data convert_mnist_data.cpp)
target_link_libraries(convert_mnist_data Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(convert_mnist_data PUBLIC "${MORE_INCS}")


add_executable(demo_compute_image_mean demo_compute_image_mean.cpp)
target_link_libraries(demo_compute_image_mean Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(demo_compute_image_mean PUBLIC "${MORE_INCS}")


add_executable(dump_network dump_network.cpp)
target_link_libraries(dump_network Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(dump_network PUBLIC "${MORE_INCS}")

add_executable(evaluate_network evaluate_network.cpp)
target_link_libraries(evaluate_network Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(evaluate_network PUBLIC "${MORE_INCS}")

add_executable(finetune_net finetune_net.cpp)
target_link_libraries(finetune_net Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(finetune_net PUBLIC "${MORE_INCS}")

add_executable(net_speed_benchmark net_speed_benchmark.cpp)
target_link_libraries(net_speed_benchmark Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(net_speed_benchmark PUBLIC "${MORE_INCS}")

add_executable(quantize_network quantize_network.cpp)
target_link_libraries(quantize_network Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(quantize_network PUBLIC "${MORE_INCS}")

add_executable(test_net test_net.cpp)
target_link_libraries(test_net Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(test_net PUBLIC "${MORE_INCS}")

add_executable(train_net train_net.cpp)
target_link_libraries(train_net Caffe CaffeProto CaffeLayers ${MORE_LIBS})
target_include_directories(train_net PUBLIC "${MORE_INCS}")
