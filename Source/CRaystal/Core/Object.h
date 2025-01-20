#pragma once
#include "Macros.h"

namespace CRay {
/** Base class for all host object.
 *  Owns a proxy pointer to device data, upload before kernel execution.
 */
class CRAYSTAL_API HostObject {
   public:
    HostObject() = default;

    HostObject(const HostObject&) = delete;
    HostObject(HostObject&&) noexcept = default;

    HostObject& operator=(const HostObject& other) = delete;
    HostObject& operator=(HostObject&& other) noexcept = default;

    virtual void updateDeviceData() const {}
    virtual void readbackDeviceData() {}

    virtual ~HostObject() = default;
};
}  // namespace CRay
