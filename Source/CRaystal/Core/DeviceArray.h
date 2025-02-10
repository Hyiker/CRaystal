#pragma once
#include <cinttypes>

#include "Macros.h"
namespace CRay {
/** Device view of an array data memory.
 */
template <typename T>
class DeviceArray {
   public:
    CRAYSTAL_HOST DeviceArray() = default;
    CRAYSTAL_HOST DeviceArray(T* ptr, int size) : mPtr(ptr), mSize(size) {}

    CRAYSTAL_DEVICE_HOST int size() const { return mSize; }
    CRAYSTAL_DEVICE_HOST T* data() const { return mPtr; }
    CRAYSTAL_DEVICE T operator[](int index) { return mPtr[index]; }
    CRAYSTAL_DEVICE T operator[](int index) const { return mPtr[index]; }

   private:
    T* mPtr = nullptr;  ///< Device pointer.
    int mSize = 0;      ///< Array size.
};
}  // namespace CRay
