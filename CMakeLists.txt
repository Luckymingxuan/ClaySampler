cmake_minimum_required(VERSION 3.20)
project(ClaySampler LANGUAGES CXX)

add_executable(claysampler
    src/main.cpp
    src/sfz/instrument.cpp
    src/io/audio.cpp
    src/io/midi.cpp
    src/render/plan.cpp
    src/cuda/renderer.cpp
)
set_target_properties(claysampler PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_link_libraries(claysampler PRIVATE ${CMAKE_DL_LIBS})
if(MINGW)
    target_link_options(claysampler PRIVATE -municode)
endif()
if(MSVC)
    target_compile_options(claysampler PRIVATE /W4 /utf-8)
else()
    target_compile_options(claysampler PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/cuda/render.cu src/cuda/render_types.h)
file(READ src/cuda/render.cu CLAY_CUDA_SOURCE)
file(READ src/cuda/render_types.h CLAY_CUDA_TYPES)
configure_file(src/cuda/kernel_source.h.in generated/kernel_source.h @ONLY)
target_include_directories(claysampler PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
install(TARGETS claysampler RUNTIME DESTINATION bin)
