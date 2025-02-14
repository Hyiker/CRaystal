#pragma once
#include "Core/Macros.h"

namespace CRay {
/** Cuda device compatible std::variant.
 */
template <typename... Types>
class CVariant {
   public:
    CRAYSTAL_DEVICE_HOST CVariant() : typeIndex(-1) {}

    CRAYSTAL_DEVICE_HOST CVariant(const CVariant& other)
        : typeIndex(other.typeIndex) {
        if (typeIndex >= 0) {
            copyConstructDispatch(typeIndex, storage, other.storage);
        }
    }

    CRAYSTAL_DEVICE_HOST CVariant(CVariant&& other) noexcept
        : typeIndex(other.typeIndex) {
        if (typeIndex >= 0) {
            moveConstructDispatch(typeIndex, storage, other.storage);
            other.typeIndex = -1;
        }
    }

    template <typename T, typename = std::enable_if_t<(
                              getTypeIndex<std::remove_reference_t<T>>() >= 0)>>
    CRAYSTAL_DEVICE_HOST CVariant(T&& value) {
        using RawT = std::remove_reference_t<T>;
        typeIndex = getTypeIndex<RawT>();
        // Avoid infinite recursion construction
        if constexpr (std::is_rvalue_reference_v<T&&>) {
            moveConstructAt(storage, std::move(value));
        } else {
            constructAt(storage, value);
        }
    }

    CRAYSTAL_DEVICE_HOST ~CVariant() { destroy(); }

    CRAYSTAL_DEVICE_HOST CVariant& operator=(const CVariant& other) {
        if (this != &other) {
            destroy();
            typeIndex = other.typeIndex;
            if (typeIndex >= 0) {
                copyConstructDispatch(typeIndex, storage, other.storage);
            }
        }
        return *this;
    }

    CRAYSTAL_DEVICE_HOST CVariant& operator=(CVariant&& other) noexcept {
        if (this != &other) {
            destroy();
            typeIndex = other.typeIndex;
            if (typeIndex >= 0) {
                moveConstructDispatch(typeIndex, storage, other.storage);
                other.typeIndex = -1;
            }
        }
        return *this;
    }

    CRAYSTAL_DEVICE_HOST int index() const { return typeIndex; }

    CRAYSTAL_DEVICE_HOST bool valueless_by_exception() const {
        return typeIndex < 0;
    }

    template <typename T>
    CRAYSTAL_DEVICE_HOST T* get_if() {
        static_assert(getTypeIndex<T>() >= 0, "Invalid type");
        if (typeIndex == getTypeIndex<T>()) {
            return reinterpret_cast<T*>(storage);
        }
        return nullptr;
    }

    template <typename T>
    CRAYSTAL_DEVICE_HOST const T* get_if() const {
        static_assert(getTypeIndex<T>() >= 0, "Invalid type");
        if (typeIndex == getTypeIndex<T>()) {
            return reinterpret_cast<const T*>(storage);
        }
        return nullptr;
    }

    template <typename T>
    CRAYSTAL_DEVICE_HOST bool holds_alternative() const {
        return typeIndex == getTypeIndex<T>();
    }

    template <typename T, typename... Args>
    CRAYSTAL_DEVICE_HOST T& emplace(Args&&... args) {
        static_assert(getTypeIndex<T>() >= 0, "Invalid type");
        destroy();
        typeIndex = getTypeIndex<T>();
        new (storage) T(std::forward<Args>(args)...);
        return *reinterpret_cast<T*>(storage);
    }

   private:
    CRAYSTAL_DEVICE_HOST void destroy() {
        if (typeIndex >= 0) {
            destroyDispatch(typeIndex, storage);
            typeIndex = -1;
        }
    }

   private:
    static constexpr size_t MaxAlign = std::max({alignof(Types)...});
    static constexpr size_t MaxSize = std::max({sizeof(Types)...});

    alignas(MaxAlign) unsigned char storage[MaxSize];
    int typeIndex;

    template <typename T, typename... Rest>
    struct IndexOf;

    template <typename T, typename First, typename... Rest>
    struct IndexOf<T, First, Rest...> {
        static constexpr int value =
            std::is_same_v<T, First> ? 0 : (1 + IndexOf<T, Rest...>::value);
    };

    template <typename T>
    struct IndexOf<T> {
        static constexpr int value = -1;
    };

    template <typename T>
    static constexpr int getTypeIndex() {
        return IndexOf<T, Types...>::value;
    }

    template <size_t I>
    CRAYSTAL_DEVICE_HOST static void destroyImpl(void* ptr) {
        using T = std::tuple_element_t<I, std::tuple<Types...>>;
        static_cast<T*>(ptr)->~T();
    }

    template <size_t I>
    CRAYSTAL_DEVICE_HOST static void copyConstructImpl(void* dst,
                                                       const void* src) {
        using T = std::tuple_element_t<I, std::tuple<Types...>>;
        new (dst) T(*static_cast<const T*>(src));
    }

    template <size_t I>
    CRAYSTAL_DEVICE_HOST static void moveConstructImpl(void* dst, void* src) {
        using T = std::tuple_element_t<I, std::tuple<Types...>>;
        new (dst) T(std::move(*static_cast<T*>(src)));
    }

    template <size_t I = 0>
    CRAYSTAL_DEVICE_HOST static void destroyDispatch(int index, void* ptr) {
        if constexpr (I >= sizeof...(Types)) {
            return;
        } else {
            if (index == I) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                static_cast<T*>(ptr)->~T();
            } else {
                destroyDispatch<I + 1>(index, ptr);
            }
        }
    }

    template <size_t I = 0>
    CRAYSTAL_DEVICE_HOST static void copyConstructDispatch(int index, void* dst,
                                                           const void* src) {
        if constexpr (I >= sizeof...(Types)) {
            return;
        } else {
            if (index == I) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                new (dst) T(*static_cast<const T*>(src));
            } else {
                copyConstructDispatch<I + 1>(index, dst, src);
            }
        }
    }

    template <size_t I = 0>
    CRAYSTAL_DEVICE_HOST static void moveConstructDispatch(int index, void* dst,
                                                           void* src) {
        if constexpr (I >= sizeof...(Types)) {
            return;
        } else {
            if (index == I) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                new (dst) T(std::move(*static_cast<T*>(src)));
            } else {
                moveConstructDispatch<I + 1>(index, dst, src);
            }
        }
    }

    template <typename T>
    CRAYSTAL_DEVICE_HOST static void constructAt(void* ptr, const T& src) {
        new (ptr) T(src);
    }

    template <typename T>
    CRAYSTAL_DEVICE_HOST static void moveConstructAt(void* ptr, T&& src) {
        new (ptr) T(std::move(src));
    }
};

template <typename T, typename... Types>
CRAYSTAL_DEVICE_HOST bool holds_alternative(const CVariant<Types...>& v) {
    return v.template holds_alternative<T>();
}

template <typename T, typename... Types>
CRAYSTAL_DEVICE_HOST T* get_if(CVariant<Types...>* v) {
    return v ? v->template get_if<T>() : nullptr;
}

template <typename T, typename... Types>
CRAYSTAL_DEVICE_HOST const T* get_if(const CVariant<Types...>* v) {
    return v ? v->template get_if<T>() : nullptr;
}

}  // namespace CRay
