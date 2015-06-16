add_executable( dense_inference util.cpp dense_inference.cpp )
add_executable( dense_inference_additive util.cpp dense_inference_additive.cpp )
target_link_libraries( dense_inference densecrf )
target_link_libraries( dense_inference_additive densecrf )
