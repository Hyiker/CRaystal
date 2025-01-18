function(add_craystal_test target)
    add_craystal_executable(${target})

    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
endfunction()

add_craystal_test(CameraTest)
target_sources(CameraTest PRIVATE
    CameraTest.cpp
)

add_craystal_test(CoreTest)
target_sources(CoreTest PRIVATE
    CoreTest.cpp
)

add_craystal_test(ImageTest)
target_sources(ImageTest PRIVATE
    ImageTest.cpp
)
