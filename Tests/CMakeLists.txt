function(add_craystal_test target)
    add_craystal_executable(${target})

    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
endfunction()

add_craystal_test(CoreTest)
target_sources(CoreTest PRIVATE
    CoreTest.cpp
)
