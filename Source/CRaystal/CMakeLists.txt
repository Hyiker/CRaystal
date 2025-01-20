add_library(CRaystal SHARED)

target_sources(CRaystal PRIVATE
    CRaystal.h

    Core/Camera.cu
    Core/Camera.h
    Core/CameraProxy.h
    Core/Sensor.h

    Core/Buffer.cu
    Core/Buffer.h
    Core/Enum.h
    Core/Error.h
    Core/Error.cpp
    Core/Macros.h
    Core/Object.h
    Core/Spectrum.cpp
    Core/Spectrum.h
    Core/SpectrumTabData.h
    Core/Vec.cpp
    Core/Vec.h

    Math/CRayMath.h
    Math/MathDefs.h
    Math/Ray.h

    Utils/CudaUtils.h
    Utils/Image.cpp
    Utils/Image.h
    Utils/Logger.cpp
    Utils/Logger.h
)

target_precompile_headers(CRaystal
    PRIVATE
    CRaystal.h
)

target_compile_features(CRaystal
    PUBLIC cxx_std_20
    PRIVATE cxx_std_20
)

set_target_properties(CRaystal PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(CRaystal
    PUBLIC

    # MSVC flags.
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
    /Zi # generate debug symbols
    /MP # enable multi-processor compilation

    # Configure warnings
    /WX # warnings as errors
    /W4 # increase warning level
    /wd4251 # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
    /wd4819 # The file contains a character that cannot be represented in the current code page(936)
    /wd4244 # 'conversion' conversion from 'type1' to 'type2', possible loss of data
    /wd4267 # 'var' : conversion from 'size_t' to 'type', possible loss of data
    /wd4100 # unreferenced formal parameter
    /wd4201 # nonstandard extension used: nameless struct/union
    /wd4245 # conversion from 'type1' to 'type2', signed/unsigned mismatch
    /wd4189 # local variable is initialized but not referenced
    /wd4127 # conditional expression is constant
    /wd4701 # potentially uninitialized local variable 'name' used
    /wd4703 # potentially uninitialized local pointer variable 'name' used
    /wd4324 # structure was padded due to alignment specifier
    /wd4505 # unreferenced local function has been removed
    /wd4702 # unreachable code
    /wd4389 # signed/unsigned mismatch
    /wd4459 # declaration of 'identifier' hides global declaration
    /wd4268 # 'identifier' : 'const' static/global data initialized with compiler generated default constructor fills the object with zeros
    /external:templates- # allows warnings from system headers when they occur in a template that's instantiated in your code
    /external:W1 # Controls the warning level for external code (e.g., system headers or third-party libraries), where <level> sets the desired warning level (0 to disable, 1 to enable, etc.).

    # Enable warnings that are disabled with /W4
    /we4263 # 'function': member function does not override any base class virtual member function
    /we4264 # 'virtual_function': no override available for virtual member function from base 'class'; function is hidden
    /we5038 # data member 'member1' will be initialized after data member 'member2'
    /we5259 # 'specialized-type': explicit specialization requires 'template <>'
    /we5263 # calling 'std::move' on a temporary object prevents copy elision
    >

    # Clang/GCC flags.
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
    -fms-extensions # enable MS extensions (among other things allow anonymous structs)
    -fvisibility=hidden # hide symbols by default
    -Wall # set warning level
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-switch
    -Wno-missing-braces
    -Wno-invalid-offsetof
    >

    # Clang flags.
    $<$<CXX_COMPILER_ID:Clang>:
    -Wno-unused-private-field
    -Wno-braced-scalar-init
    -Wno-self-assign-overloaded
    >

    # GCC flags.
    $<$<CXX_COMPILER_ID:GNU>:
    -fpermissive
    -Wno-sign-compare
    -Wno-literal-suffix
    -Wno-class-memaccess
    -Wno-strict-aliasing
    -Wno-maybe-uninitialized
    -Wno-stringop-truncation
    >

    # NVCC flags.
    $<$<AND:$<PLATFORM_ID:Windows>,$<COMPILE_LANGUAGE:CUDA>>:
    -Xcompiler=/utf-8
    >

    $<$<COMPILE_LANGUAGE:CUDA>:
    -diag-suppress=20012
    -diag-suppress=1394
    >

    PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj> # big object files
)

target_link_options(CRaystal
    PUBLIC

    # MSVC flags.
    $<$<CXX_COMPILER_ID:MSVC>:/DEBUG> # generate debug information
)

target_compile_definitions(CRaystal
    PUBLIC
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:Debug>:_DEBUG>

    # Windows.
    $<$<PLATFORM_ID:Windows>:NOMINMAX> # do not define min/max macros
    $<$<PLATFORM_ID:Windows>:UNICODE> # force character map to unicode

    # MSVC C++ library.
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
    $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS>

    # Clang.
    $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions

    # CRaystal feature flags.
    CRAYSTAL_HAS_CUDA=$<BOOL:${CRAYSTAL_HAS_CUDA}>
    CRAYSTAL_ENABLE_F64=$<BOOL:${CRAYSTAL_ENABLE_F64}>
    CRAYSTAL_ENABLE_SPECTRUM=$<BOOL:${CRAYSTAL_ENABLE_SPECTRUM}>

    PRIVATE

    CRAYSTAL_PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
    CRAYSTAL_MODULE=1
    $<$<PLATFORM_ID:Windows>:IMGUI_API=__declspec\(dllexport\)>
    $<$<PLATFORM_ID:Linux>:IMGUI_API=__attribute__\(\(visibility\("default"\)\)\)>
)

target_include_directories(CRaystal
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}

    PRIVATE
    ../../External/stb
)

set_target_properties(CRaystal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CRAYSTAL_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${CRAYSTAL_LIBRARY_OUTPUT_DIRECTORY}
)

target_link_libraries(CRaystal
    PUBLIC
    fmt::fmt glm::glm
    PRIVATE
    tinyexr
)
