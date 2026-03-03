set(OP_NAME macro_notch)

# Pure Python op — no C++ compilation needed.
# This file only registers the Python sources for make install.
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
