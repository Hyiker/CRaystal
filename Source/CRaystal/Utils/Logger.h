#pragma once
#include <fmt/format.h>

#include <filesystem>
#include <stdexcept>
#include <string_view>

#include "Core/Macros.h"
namespace CRay {

class CRAYSTAL_API Logger {
   public:
    enum class Level : int {

        Disabled,  // disabled logger, namely suppress all output.
        Fatal,     // fatal error message, which will terminate the program.
        Error,     // error message.
        Warning,   // warning message.
        Info,      // normal information output.
        Debug,     // verbose message for debugging.
        Count
    };

    static void init();
    static void log(Level level, const std::string_view message);
    static void shutdown();

   private:
    Logger();
};

void logBeforeInitialized(Logger::Level level, const std::string_view message);

template <typename... Args>
inline void logBeforeInitialized(Logger::Level level,
                                 fmt::format_string<Args...> format,
                                 Args &&...args) {
    logBeforeInitialized(level,
                         fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline void logDebug(fmt::format_string<Args...> format, Args &&...args) {
    Logger::log(Logger::Level::Debug,
                fmt::format(format, std::forward<Args>(args)...));
}
inline void logDebug(const std::string_view message) {
    Logger::log(Logger::Level::Debug, message);
}

template <typename... Args>
inline void logInfo(fmt::format_string<Args...> format, Args &&...args) {
    Logger::log(Logger::Level::Info,
                fmt::format(format, std::forward<Args>(args)...));
}
inline void logInfo(const std::string_view message) {
    Logger::log(Logger::Level::Info, message);
}

template <typename... Args>
inline void logWarning(fmt::format_string<Args...> format, Args &&...args) {
    Logger::log(Logger::Level::Warning,
                fmt::format(format, std::forward<Args>(args)...));
}
inline void logWarning(const std::string_view message) {
    Logger::log(Logger::Level::Warning, message);
}

template <typename... Args>
inline void logError(fmt::format_string<Args...> format, Args &&...args) {
    Logger::log(Logger::Level::Error,
                fmt::format(format, std::forward<Args>(args)...));
}
inline void logError(const std::string_view message) {
    Logger::log(Logger::Level::Error, message);
}

template <typename... Args>
[[noreturn]] inline void logFatal(fmt::format_string<Args...> format,
                                  Args &&...args) {
    const auto &message = fmt::format(format, std::forward<Args>(args)...);
    Logger::log(Logger::Level::Fatal, message);
    throw std::runtime_error(message);
}

[[noreturn]] inline void logFatal(const std::string_view message) {
    Logger::log(Logger::Level::Fatal, message);
    throw std::runtime_error(message.data());
}

}  // namespace CRay
