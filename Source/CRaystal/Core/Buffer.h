#pragma once

#include <cstdint>
#include <memory>

#include "DeviceArray.h"
#include "Macros.h"

namespace CRay {

/** Simple CUDA device buffer RAII wrapper.
 */
class CRAYSTAL_API DeviceBuffer {
   public:
    using Ref = std::shared_ptr<DeviceBuffer>;

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

    template <typename T>
    DeviceArray<T> getDeviceArray() const {
        return DeviceArray<T>(static_cast<T*>(mpDeviceData), mSize / sizeof(T));
    }

   private:
    void free();

    void* mpDeviceData = nullptr;  ///< device pointer
    int mSize = 0;                 ///< Buffer size in bytes
};

}  // namespace CRay
