#pragma once

#include <cstdint>
#include <memory>

#include "Macros.h"

namespace CRay {

/** Simple CUDA device buffer RAII wrapper.
 */
class CRAYSTAL_API DeviceBuffer {
   public:
    DeviceBuffer(int sizeInBytes);
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data();
    const void* data() const;
    int size() const;

    void copyFromHost(const void* hostData);
    void copyToHost(void* hostData) const;
    void memset(unsigned char value);

   private:
    void free();

    void* mpDeviceData;  ///< device pointer
    int mSize;           ///< Buffer size in bytes
};

}  // namespace CRay
