#project(thirdparty)

add_subdirectory(flute)

set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentation for Limbo")
set(ENABLE_TEST OFF CACHE BOOL "Turn off test for Limbo")
add_subdirectory(Limbo)
add_subdirectory(munkres-cpp)
add_subdirectory(OpenTimer)

# HeteroSTA ships as a pre-built GPU binary (x86_64 / CUDA only).
# On CPU-only or non-x86_64 builds (e.g. Apple Silicon) pass
# -DENABLE_HETEROSTA=OFF to skip the download and avoid a FATAL_ERROR.
option(ENABLE_HETEROSTA "Download and build HeteroSTA timing engine" ON)
if(ENABLE_HETEROSTA)
  add_subdirectory(HeteroSTA)
endif()

install(
    DIRECTORY NCTUgr.ICCAD2012 DESTINATION thirdparty
    )

