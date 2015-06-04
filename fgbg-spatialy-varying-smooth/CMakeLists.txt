add_executable( fgbg-svs-gco fgbg-svs-gco.cpp )
add_executable( fgbg-3channel-svs-gco fgbg-3channel-svs-gco.cpp )
target_link_libraries( fgbg-3channel-svs-gco libgco ${OpenCV_LIBS} )
target_link_libraries( fgbg-svs-gco libgco ${OpenCV_LIBS} )
