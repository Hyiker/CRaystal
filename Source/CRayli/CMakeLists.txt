add_craystal_executable(CRayli)

target_sources(CRayli PRIVATE
    CRayli.cpp
)

set_target_properties(CRayli PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
