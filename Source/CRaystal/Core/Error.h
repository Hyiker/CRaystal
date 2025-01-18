#pragma once

#include <fmt/format.h>

#include <cassert>
#include <exception>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

#include "Macros.h"

#define CRAYSTAL_ASSERT(x) assert(x)

#define CRAYSTAL_ERROR(x)

namespace CRay {

#if CRAYSTAL_MSVC
#pragma warning(push)
#pragma warning(disable : 4275)  // allow dllexport on classes dervied from STL
#endif

#if CRAYSTAL_NVCC
#pragma nv_diag_suppress 1388  // allow dllexport on classes dervied from STL
#endif

class CRAYSTAL_API Exception : public std::exception {
   public:
    Exception() noexcept {}
    Exception(std::string_view what)
        : mpWhat(std::make_shared<std::string>(what)) {}
    Exception(const Exception& other) noexcept { mpWhat = other.mpWhat; }
    virtual ~Exception() override {}
    virtual const char* what() const noexcept override {
        return mpWhat ? mpWhat->c_str() : "";
    }

   protected:
    // Message is stored as a reference counted string in order to allow copy
    // constructor to be noexcept.
    std::shared_ptr<std::string> mpWhat;
};


#if CRAYSTAL_NVCC
#pragma nv_diag_default 1388
#endif

#if CRAYSTAL_MSVC
#pragma warning(pop)
#endif

/**
 * Exception to be thrown when an error happens at runtime.
 */
class CRAYSTAL_API RuntimeError : public Exception {
   public:
    RuntimeError() noexcept {}
    RuntimeError(std::string_view what) : Exception(what) {}
    RuntimeError(const RuntimeError& other) noexcept { mpWhat = other.mpWhat; }
    virtual ~RuntimeError() override {}
};

/**
 * Exception to be thrown on CRAYSTAL_ASSERT.
 */
class CRAYSTAL_API AssertionError : public Exception {
   public:
    AssertionError() noexcept {}
    AssertionError(std::string_view what) : Exception(what) {}
    AssertionError(const AssertionError& other) noexcept {
        mpWhat = other.mpWhat;
    }
    virtual ~AssertionError() override {}
};

//
// Exception helpers.
//

/// Throw a RuntimeError exception.
/// If ErrorDiagnosticFlags::AppendStackTrace is set, a stack trace will be
/// appended to the exception message. If ErrorDiagnosticFlags::BreakOnThrow is
/// set, the debugger will be broken into (if attached).
[[noreturn]] CRAYSTAL_API void throwException(const std::source_location& loc,
                                              std::string_view msg);

namespace detail {
/// Overload to allow CRAYSTAL_THROW to be called with a message only.
[[noreturn]] inline void throwException(const std::source_location& loc,
                                        std::string_view msg) {
    ::CRay::throwException(loc, msg);
}

/// Overload to allow CRAYSTAL_THROW to be called with a format string and
/// arguments.
template <typename... Args>
[[noreturn]] inline void throwException(const std::source_location& loc,
                                        fmt::format_string<Args...> fmt,
                                        Args&&... args) {
    ::CRay::throwException(loc, fmt::format(fmt, std::forward<Args>(args)...));
}
}  // namespace detail
}  // namespace CRay

/// Helper for throwing a RuntimeError exception.
/// Accepts either a string or a format string and arguments:
/// CRAYSTAL_THROW("This is an error message.");
/// CRAYSTAL_THROW("Expected {} items, got {}.", expectedCount, actualCount);
#define CRAYSTAL_THROW(...) \
    ::CRay::detail::throwException(std::source_location::current(), __VA_ARGS__)

/// Helper for throwing a RuntimeError exception if condition isn't met.
/// Accepts either a string or a format string and arguments.
/// CRAYSTAL_CHECK(device != nullptr, "Device is null.");
/// CRAYSTAL_CHECK(count % 3 == 0, "Count must be a multiple of 3, got {}.",
/// count);
#define CRAYSTAL_CHECK(cond, ...)                 \
    do {                                          \
        if (!(cond)) CRAYSTAL_THROW(__VA_ARGS__); \
    } while (0)

/// Helper for marking unimplemented functions.
#define CRAYSTAL_UNIMPLEMENTED() CRAYSTAL_THROW("Unimplemented")

/// Helper for marking unreachable code.
#define CRAYSTAL_UNREACHABLE() CRAYSTAL_THROW("Unreachable")

//
// Assertions.
//

namespace CRay {
/// Report an assertion.
/// If ErrorDiagnosticFlags::AppendStackTrace is set, a stack trace will be
/// appended to the exception message. If ErrorDiagnosticFlags::BreakOnAssert is
/// set, the debugger will be broken into (if attached).
[[noreturn]] CRAYSTAL_API void reportAssertion(const std::source_location& loc,
                                               std::string_view cond,
                                               std::string_view msg = {});

namespace detail {
/// Overload to allow CRAYSTAL_ASSERT to be called without a message.
[[noreturn]] inline void reportAssertion(const std::source_location& loc,
                                         std::string_view cond) {
    ::CRay::reportAssertion(loc, cond);
}

/// Overload to allow CRAYSTAL_ASSERT to be called with a message only.
[[noreturn]] inline void reportAssertion(const std::source_location& loc,
                                         std::string_view cond,
                                         std::string_view msg) {
    ::CRay::reportAssertion(loc, cond, msg);
}

/// Overload to allow CRAYSTAL_ASSERT to be called with a format string and
/// arguments.
template <typename... Args>
[[noreturn]] inline void reportAssertion(const std::source_location& loc,
                                         std::string_view cond,
                                         fmt::format_string<Args...> fmt,
                                         Args&&... args) {
    ::CRay::reportAssertion(loc, cond,
                            fmt::format(fmt, std::forward<Args>(args)...));
}
}  // namespace detail
}  // namespace CRay
