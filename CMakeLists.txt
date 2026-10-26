cmake_minimum_required(VERSION 3.18)  # 3.18+ needed for clean CUDA language support
# LANGUAGES C CXX (not just CXX): VTK's MPI detection (pulled in transitively
# via ITKVtkGlue) needs the C language enabled, even though this project's
# own code is all C++. Declaring only CXX here broke configuration - a real
# regression caught by actually re-running cmake after this file's GPU
# changes, not just reasoning about them.
project(ct_segmentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU support is OPTIONAL and auto-detected: the project builds and runs
# exactly as before on machines with no CUDA-capable GPU/toolkit. Only
# when a CUDA compiler is found does the extra ct_segmentation_gpu
# executable (and the gpu/ filters it uses) get built at all.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA compiler found - ct_segmentation_gpu will be built")
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "No CUDA compiler found - skipping GPU executable (CPU tools still build normally)")
endif()

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(${CMAKE_SOURCE_DIR})

# Qt6 Widgets + OpenGL needed for the interactive Qt/VTK viewer (ct_viewer).
# Qt6 MOC must run automatically for Q_OBJECT classes.
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(VTK REQUIRED COMPONENTS
    CommonCore CommonDataModel RenderingCore RenderingOpenGL2
    InteractionStyle InteractionWidgets ImagingCore RenderingFreeType
    GUISupportQt RenderingAnnotation FiltersCore FiltersGeometry)

add_executable(ct_segmentation main.cpp)
target_link_libraries(ct_segmentation ${ITK_LIBRARIES})

add_executable(debug_canny tools/debug_canny.cpp)
target_link_libraries(debug_canny ${ITK_LIBRARIES})

add_executable(ct_segmentation_3d main_volume.cpp)
target_link_libraries(ct_segmentation_3d ${ITK_LIBRARIES})

add_executable(test_roi_processor tools/test_roi_processor.cpp)
target_link_libraries(test_roi_processor ${ITK_LIBRARIES})

# ct_preprocess: One-time preprocessing and multi-plane slice extraction
add_executable(ct_preprocess
    tools/ct_preprocess.cpp
    volume/SliceStore.cpp)
target_link_libraries(ct_preprocess ${ITK_LIBRARIES})

# test_lazy_loading: Unit & verification tests for SliceStore and SliceCache
add_executable(test_lazy_loading
    tools/test_lazy_loading.cpp
    volume/SliceStore.cpp)
target_link_libraries(test_lazy_loading ${ITK_LIBRARIES})


# ct_viewer: Qt6 + VTK interactive window with lazy slice loading
add_executable(ct_viewer
    main_viewer.cpp
    volume/SliceStore.cpp
    viewer/OrthoSliceView.cpp
    viewer/CTViewerWindow.cpp)
target_link_libraries(ct_viewer
    ${ITK_LIBRARIES}
    ${VTK_LIBRARIES}
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets)
vtk_module_autoinit(TARGETS ct_viewer MODULES ${VTK_LIBRARIES})

if(HAVE_CUDA)
    link_directories(BEFORE /usr/lib/x86_64-linux-gnu)

    add_executable(ct_segmentation_gpu
        main_volume_gpu.cpp
        volume/SliceStore.cpp
        gpu/MedianFilter3D.cu
        gpu/GaussianFilter3D.cu
        gpu/Morphology3D.cu
        gpu/Segmentation3D.cu)
    set_target_properties(ct_segmentation_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89")  # 89 for RTX 40-series (Ada Lovelace, RTX 4050)
    target_link_libraries(ct_segmentation_gpu /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${ITK_LIBRARIES})

    # Enable GPU acceleration in ct_viewer when CUDA is available
    target_compile_definitions(ct_viewer PRIVATE HAVE_CUDA=1)
    target_sources(ct_viewer PRIVATE
        gpu/MedianFilter3D.cu
        gpu/GaussianFilter3D.cu
        gpu/Morphology3D.cu
        gpu/Segmentation3D.cu)
    set_target_properties(ct_viewer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89")
    target_link_libraries(ct_viewer /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
endif()
