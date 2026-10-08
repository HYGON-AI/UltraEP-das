#pragma once

#include <exception>
#include <sstream>
#include <string>

#include "../platform/runtime.hpp"

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        std::stringstream ss;
        ss << name << " exception (" << file << ":" << line << "): " << error;
        message = ss.str();
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#define EPExceptionWithLineInfo(name, message) EPException(name, __FILE__, __LINE__, message)

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_HOST_UNREACHABLE
#define EP_HOST_UNREACHABLE(reason) (throw EPException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef EP_DEVICE_ASSERT
#if defined(__HIP_DEVICE_COMPILE__)
#define EP_DEVICE_TRAP() __builtin_trap()
#else
#define EP_DEVICE_TRAP() asm("trap;")
#endif
#define EP_DEVICE_ASSERT(cond)                                                             \
    do {                                                                                   \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            EP_DEVICE_TRAP();                                                                  \
        }                                                                                  \
    } while (0)
#endif

#ifndef EP_UNIFIED_ASSERT
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define EP_UNIFIED_ASSERT(cond) EP_DEVICE_ASSERT(cond)
#else
#define EP_UNIFIED_ASSERT(cond) EP_HOST_ASSERT(cond)
#endif
#endif

#ifndef DEVICE_RUNTIME_CHECK
#define DEVICE_RUNTIME_CHECK(cmd)                                                                                  \
    do {                                                                                                           \
        const auto e = (cmd);                                                                                      \
        if (e != ultra_ep::platform::kRuntimeSuccess) {                                                            \
            std::stringstream ss;                                                                                  \
            ss << static_cast<int>(e) << " (" << ultra_ep::platform::runtime_error_name(e) << ", "                \
               << ultra_ep::platform::runtime_error_string(e) << ")";                                             \
            throw EPException(ultra_ep::platform::backend_name(), __FILE__, __LINE__, ss.str());                   \
        }                                                                                                        \
    } while (0)
#endif

// Temporary source-compatibility alias. New code should use the backend-neutral
// name; keeping this avoids coupling the runtime-abstraction step to PTX/SHMEM
// migration work scheduled for later phases.
#ifndef CUDA_RUNTIME_CHECK
#define CUDA_RUNTIME_CHECK(cmd) DEVICE_RUNTIME_CHECK(cmd)
#endif

#ifndef CUDA_DRIVER_CHECK
#define CUDA_DRIVER_CHECK(cmd)                                                \
    do {                                                                      \
        const auto e = (cmd);                                                 \
        if (e != CUDA_SUCCESS) {                                              \
            std::stringstream ss;                                             \
            const char *name, *info;                                          \
            cuGetErrorName(e, &name), cuGetErrorString(e, &info);             \
            ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
            throw EPException("CUDA driver", __FILE__, __LINE__, ss.str());   \
        }                                                                     \
    } while (0)
#endif
