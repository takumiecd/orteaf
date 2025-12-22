#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace orteaf::internal::base {

/**
 * @brief Small buffer-optimized vector with inline storage.
 *
 * SmallVector keeps up to N elements inline before falling back to heap storage.
 * It provides a std::vector-like API with contiguous storage semantics.
 *
 * Exception safety:
 * - Basic operations provide the strong guarantee when element operations are
 *   noexcept.
 * - Insert/emplace overloads that shift elements require nothrow move/copy
 *   construction to avoid leaving gaps in partially modified storage.
 *
 * Invariants:
 * - data_ always points to valid storage of size capacity_.
 * - size_ is the number of constructed elements in [data_, data_ + size_).
 * - Inline storage is used when size_ <= N and heap storage has been released.
 */
template <typename T, std::size_t N>
class SmallVector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    /// @brief Constructs an empty SmallVector using inline storage when available.
    SmallVector() noexcept {
        resetInlinePointer();
    }

    /// @brief Constructs with count copies of value (copy-constructible only).
    SmallVector(size_type count, const T& value) requires std::is_copy_constructible_v<T> : SmallVector() {
        resize(count, value);
    }

    SmallVector(size_type count, const T& value) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Constructs with count default-constructed elements.
    explicit SmallVector(size_type count) : SmallVector() {
        resize(count);
    }

    /// @brief Constructs from an initializer list (copy-constructible only).
    SmallVector(std::initializer_list<T> init) requires std::is_copy_constructible_v<T> : SmallVector() {
        assign(init.begin(), init.end());
    }

    SmallVector(std::initializer_list<T> init) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Copy-constructs from another SmallVector (copy-constructible only).
    SmallVector(const SmallVector& other) requires std::is_copy_constructible_v<T> : SmallVector() {
        assign(other.begin(), other.end());
    }

    SmallVector(const SmallVector& other) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Move-constructs from another SmallVector.
    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) : SmallVector() {
        moveFrom(other);
    }

    /// @brief Destroys the vector and releases heap storage if used.
    ~SmallVector() {
        clear();
        if (usingHeapStorage()) {
            deallocate(data_);
        }
    }

    /// @brief Copy-assigns from another SmallVector (copy-constructible only).
    SmallVector& operator=(const SmallVector& other) requires std::is_copy_constructible_v<T> {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }

    SmallVector& operator=(const SmallVector& other) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Move-assigns from another SmallVector.
    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_assignable_v<T>) {
        if (this != &other) {
            clear();
            releaseHeapStorage();
            moveFrom(other);
        }
        return *this;
    }

    /// @brief Assigns from an initializer list (copy-constructible only).
    SmallVector& operator=(std::initializer_list<T> init) requires std::is_copy_constructible_v<T> {
        assign(init.begin(), init.end());
        return *this;
    }

    SmallVector& operator=(std::initializer_list<T> init) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Replaces contents with count copies of value (copy-constructible only).
    void assign(size_type count, const T& value) requires std::is_copy_constructible_v<T> {
        clear();
        if constexpr (has_inline_storage) {
            if (usingHeapStorage() && count <= stack_capacity_value) {
                releaseHeapStorage();
            }
        }
        ensureCapacity(count);
        size_type i = 0;
        try {
            for (; i < count; ++i) {
                ::new (static_cast<void*>(data_ + i)) T(value);
            }
        } catch (...) {
            destroyRange(data_, data_ + i);
            throw;
        }
        size_ = count;
    }

    void assign(size_type count, const T& value) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Replaces contents with a range (copy-constructible only).
    template <typename InputIt>
    void assign(InputIt first, InputIt last) requires std::is_copy_constructible_v<T> {
        clear();
        using category = typename std::iterator_traits<InputIt>::iterator_category;
        if constexpr (std::is_base_of_v<std::forward_iterator_tag, category>) {
            const size_type count = static_cast<size_type>(std::distance(first, last));
            if constexpr (has_inline_storage) {
                if (usingHeapStorage() && count <= stack_capacity_value) {
                    releaseHeapStorage();
                }
            }
            ensureCapacity(count);
            size_type i = 0;
            try {
                for (; first != last; ++first, ++i) {
                    ::new (static_cast<void*>(data_ + i)) T(*first);
                }
            } catch (...) {
                destroyRange(data_, data_ + i);
                throw;
            }
            size_ = count;
        } else {
            if constexpr (has_inline_storage) {
                if (usingHeapStorage()) {
                    releaseHeapStorage();
                }
            }
            for (; first != last; ++first) {
                emplaceBack(*first);
            }
        }
    }

    template <typename InputIt>
    void assign(InputIt, InputIt) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Replaces contents with an initializer list (copy-constructible only).
    void assign(std::initializer_list<T> init) requires std::is_copy_constructible_v<T> {
        assign(init.begin(), init.end());
    }

    void assign(std::initializer_list<T>) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Unchecked element access.
    reference operator[](size_type index) noexcept {
        return data_[index];
    }

    /// @brief Unchecked element access.
    const_reference operator[](size_type index) const noexcept {
        return data_[index];
    }

    /// @brief Bounds-checked element access.
    reference at(size_type index) {
        if (index >= size_) {
            throw std::out_of_range("SmallVector::at");
        }
        return data_[index];
    }

    /// @brief Bounds-checked element access.
    const_reference at(size_type index) const {
        if (index >= size_) {
            throw std::out_of_range("SmallVector::at");
        }
        return data_[index];
    }

    /// @brief Returns pointer to contiguous storage.
    pointer data() noexcept {
        return data_;
    }

    const_pointer data() const noexcept {
        return data_;
    }

    /// @brief Returns iterator to the first element.
    iterator begin() noexcept {
        return data_;
    }

    /// @brief Returns iterator to the first element.
    const_iterator begin() const noexcept {
        return data_;
    }

    /// @brief Returns const iterator to the first element.
    const_iterator cbegin() const noexcept {
        return data_;
    }

    /// @brief Returns iterator to one past the last element.
    iterator end() noexcept {
        return data_ + size_;
    }

    /// @brief Returns iterator to one past the last element.
    const_iterator end() const noexcept {
        return data_ + size_;
    }

    /// @brief Returns const iterator to one past the last element.
    const_iterator cend() const noexcept {
        return data_ + size_;
    }

    /// @brief Returns true if the vector is empty.
    bool empty() const noexcept {
        return size_ == 0;
    }

    /// @brief Returns the number of elements.
    size_type size() const noexcept {
        return size_;
    }

    /// @brief Returns the current capacity.
    size_type capacity() const noexcept {
        return capacity_;
    }

    /// @brief Returns the maximum possible size.
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max();
    }

    /// @brief Ensures capacity is at least new_capacity.
    void reserve(size_type new_capacity) {
        if (new_capacity > capacity_) {
            reallocate(new_capacity);
        }
    }

    /// @brief Resizes to count default-constructed elements.
    void resize(size_type count) {
        if (count < size_) {
            destroyRange(data_ + count, data_ + size_);
            size_ = count;
        } else if (count > size_) {
            if constexpr (has_inline_storage) {
                if (usingHeapStorage() && count <= stack_capacity_value) {
                    releaseHeapStorage();
                }
            }
            ensureCapacity(count);
            size_type i = size_;
            try {
                for (; i < count; ++i) {
                    ::new (static_cast<void*>(data_ + i)) T();
                }
            } catch (...) {
                destroyRange(data_ + size_, data_ + i);
                throw;
            }
            size_ = count;
        }
    }

    /// @brief Resizes to count copies of value (copy-constructible only).
    void resize(size_type count, const T& value) requires std::is_copy_constructible_v<T> {
        if (count < size_) {
            destroyRange(data_ + count, data_ + size_);
            size_ = count;
        } else if (count > size_) {
            if constexpr (has_inline_storage) {
                if (usingHeapStorage() && count <= stack_capacity_value) {
                    releaseHeapStorage();
                }
            }
            ensureCapacity(count);
            size_type i = size_;
            try {
                for (; i < count; ++i) {
                    ::new (static_cast<void*>(data_ + i)) T(value);
                }
            } catch (...) {
                destroyRange(data_ + size_, data_ + i);
                throw;
            }
            size_ = count;
        }
    }

    void resize(size_type count, const T& value) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Destroys all elements without releasing capacity.
    void clear() noexcept {
        destroyRange(data_, data_ + size_);
        size_ = 0;
    }

    /// @brief Reduces capacity to fit size, preferring inline storage.
    void shrinkToFit() {
        if (size_ == 0) {
            releaseHeapStorage();
            resetInlinePointer();
            return;
        }
        if constexpr (has_inline_storage) {
            if (size_ <= stack_capacity_value) {
                pointer old_data = data_;
                const bool was_heap = usingHeapStorage();
                resetInlinePointer();
                size_type i = 0;
                try {
                    for (; i < size_; ++i) {
                        ::new (static_cast<void*>(data_ + i)) T(std::move_if_noexcept(old_data[i]));
                    }
                } catch (...) {
                    destroyRange(data_, data_ + i);
                    if (was_heap) {
                        data_ = old_data;
                    } else {
                        resetInlinePointer();
                    }
                    throw;
                }
                if (was_heap) {
                    destroyRange(old_data, old_data + size_);
                    deallocate(old_data);
                }
                return;
            }
        }
        if (usingHeapStorage() && size_ < capacity_) {
            reallocate(size_);
        }
    }

    /// @brief Inserts an element at pos by forwarding args (nothrow move required).
    template <typename... Args>
    iterator emplace(const_iterator pos, Args&&... args)
        requires std::is_nothrow_move_constructible_v<T> &&
                 std::is_nothrow_constructible_v<T, Args...> {
        const size_type index = indexFromConstIterator(pos);
        ensureCapacity(size_ + 1);
        shiftRight(index, 1);
        ::new (static_cast<void*>(data_ + index)) T(std::forward<Args>(args)...);
        ++size_;
        return data_ + index;
    }

    template <typename... Args>
    reference emplaceBack(Args&&... args) {
        if (size_ == capacity_) {
            ensureCapacity(size_ + 1);
        }
        ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
        ++size_;
        return back();
    }

    /// @brief Appends a copy of value (copy-constructible only).
    void pushBack(const T& value) requires std::is_copy_constructible_v<T> {
        emplaceBack(value);
    }

    void pushBack(const T& value) requires (!std::is_copy_constructible_v<T>) = delete;

    /// @brief Appends a moved value.
    void pushBack(T&& value) {
        emplaceBack(std::move(value));
    }

    /// @brief Inserts a copy of value at pos (nothrow copy/move required).
    iterator insert(const_iterator pos, const T& value)
        requires std::is_copy_constructible_v<T> &&
                 std::is_nothrow_copy_constructible_v<T> &&
                 std::is_nothrow_move_constructible_v<T> {
        return insertCount(pos, 1, value);
    }

    /// @brief Inserts a moved value at pos (nothrow move required).
    iterator insert(const_iterator pos, T&& value)
        requires std::is_nothrow_move_constructible_v<T> {
        const size_type index = indexFromConstIterator(pos);
        ensureCapacity(size_ + 1);
        shiftRight(index, 1);
        ::new (static_cast<void*>(data_ + index)) T(std::move(value));
        ++size_;
        return data_ + index;
    }

    /// @brief Inserts count copies of value at pos (nothrow copy/move required).
    iterator insert(const_iterator pos, size_type count, const T& value)
        requires std::is_copy_constructible_v<T> &&
                 std::is_nothrow_copy_constructible_v<T> &&
                 std::is_nothrow_move_constructible_v<T> {
        return insertCount(pos, count, value);
    }

    /// @brief Inserts a range of elements at pos (nothrow copy/move required).
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last)
        requires std::is_copy_constructible_v<T> &&
                 std::is_nothrow_copy_constructible_v<T> &&
                 std::is_nothrow_move_constructible_v<T> {
        using category = typename std::iterator_traits<InputIt>::iterator_category;
        if constexpr (std::is_base_of_v<std::forward_iterator_tag, category>) {
            const size_type index = indexFromConstIterator(pos);
            const size_type count = static_cast<size_type>(std::distance(first, last));
            ensureCapacity(size_ + count);
            shiftRight(index, count);
            size_type i = 0;
            for (; first != last; ++first, ++i) {
                ::new (static_cast<void*>(data_ + index + i)) T(*first);
            }
            size_ += count;
            return data_ + index;
        }
        size_type index = indexFromConstIterator(pos);
        for (; first != last; ++first, ++index) {
            pos = insert(data_ + index, *first);
        }
        return data_ + index;
    }

    /// @brief Erases the element at pos and returns iterator to next element.
    iterator erase(const_iterator pos) {
        return erase(pos, pos + 1);
    }

    /// @brief Erases elements in [first, last) and returns iterator to next element.
    iterator erase(const_iterator first, const_iterator last) {
        if (first == last) {
            return data_ + indexFromConstIterator(first);
        }
        const size_type start = indexFromConstIterator(first);
        const size_type end = indexFromConstIterator(last);
        const size_type count = end - start;
        for (size_type i = start; i + count < size_; ++i) {
            data_[i] = std::move_if_noexcept(data_[i + count]);
        }
        destroyRange(data_ + size_ - count, data_ + size_);
        size_ -= count;
        return data_ + start;
    }

    void popBack() {
        if (size_ > 0) {
            --size_;
            data_[size_].~T();
        }
    }

    /// @brief Returns reference to the first element.
    reference front() noexcept {
        return data_[0];
    }

    const_reference front() const noexcept {
        return data_[0];
    }

    /// @brief Returns reference to the last element.
    reference back() noexcept {
        return data_[size_ - 1];
    }

    const_reference back() const noexcept {
        return data_[size_ - 1];
    }

    /// @brief Swaps contents with another SmallVector.
    void swap(SmallVector& other) noexcept(
        std::is_nothrow_swappable_v<T>&&
        std::is_nothrow_move_constructible_v<T>&&
        std::is_nothrow_move_assignable_v<T>) {
        if (this == &other) {
            return;
        }

        if (usingHeapStorage() && other.usingHeapStorage()) {
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
            std::swap(capacity_, other.capacity_);
            return;
        }

        SmallVector tmp(std::move(other));
        other = std::move(*this);
        *this = std::move(tmp);
    }

private:
    using storage_type = std::aligned_storage_t<sizeof(T), alignof(T)>;

    static constexpr size_type stack_capacity_value = N;
    static constexpr bool has_inline_storage = stack_capacity_value > 0;

    pointer inlineData() noexcept {
        if constexpr (has_inline_storage) {
            return reinterpret_cast<pointer>(stack_storage_);
        }
        return nullptr;
    }

    const_pointer inlineData() const noexcept {
        if constexpr (has_inline_storage) {
            return reinterpret_cast<const_pointer>(stack_storage_);
        }
        return nullptr;
    }

    bool usingInlineStorage() const noexcept {
        if constexpr (has_inline_storage) {
            return data_ == inlineData();
        }
        return false;
    }

    bool usingHeapStorage() const noexcept {
        return data_ != nullptr && !usingInlineStorage();
    }

    void resetInlinePointer() noexcept {
        if constexpr (has_inline_storage) {
            data_ = inlineData();
            capacity_ = stack_capacity_value;
        } else {
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    void resetToStack() noexcept {
        resetInlinePointer();
        size_ = 0;
    }

    void moveFrom(SmallVector& other) {
        if (other.usingHeapStorage()) {
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.resetToStack();
        } else {
            resetInlinePointer();
            size_ = 0;
            size_type i = 0;
            try {
                for (; i < other.size_; ++i) {
                    ::new (static_cast<void*>(data_ + i)) T(std::move_if_noexcept(other.data_[i]));
                }
            } catch (...) {
                destroyRange(data_, data_ + i);
                throw;
            }
            size_ = other.size_;
            other.clear();
        }
    }

    void releaseHeapStorage() noexcept {
        if (usingHeapStorage()) {
            deallocate(data_);
            resetInlinePointer();
        }
    }

    void ensureCapacity(size_type min_capacity) {
        if (min_capacity <= capacity_) {
            return;
        }

        size_type new_capacity = capacity_ ? capacity_ * 2 : (has_inline_storage ? stack_capacity_value : size_type{0});
        if (new_capacity == 0) {
            new_capacity = 1;
        }
        if (new_capacity < min_capacity) {
            new_capacity = min_capacity;
        }
        reallocate(new_capacity);
    }

    void reallocate(size_type new_capacity) {
        pointer new_data = allocate(new_capacity);
        size_type i = 0;
        try {
            for (; i < size_; ++i) {
                ::new (static_cast<void*>(new_data + i)) T(std::move_if_noexcept(data_[i]));
            }
        } catch (...) {
            destroyRange(new_data, new_data + i);
            deallocate(new_data);
            throw;
        }

        pointer old_data = data_;
        const size_type old_size = size_;
        const bool was_heap = usingHeapStorage();

        destroyRange(data_, data_ + size_);
        if (was_heap) {
            deallocate(old_data);
        }

        data_ = new_data;
        capacity_ = new_capacity;
        size_ = old_size;
    }

    static pointer allocate(size_type capacity) {
        if (capacity == 0) {
            return nullptr;
        }
        return static_cast<pointer>(::operator new(sizeof(T) * capacity));
    }

    static void deallocate(pointer ptr) noexcept {
        ::operator delete(ptr);
    }

    static void destroyRange(pointer first, pointer last) noexcept {
        while (first != last) {
            --last;
            last->~T();
        }
    }

    size_type indexFromConstIterator(const_iterator it) const {
        return static_cast<size_type>(it - data_);
    }

    void shiftRight(size_type index, size_type count) {
        if (count == 0) {
            return;
        }
        for (size_type i = size_; i > index; --i) {
            const size_type src = i - 1;
            const size_type dest = src + count;
            ::new (static_cast<void*>(data_ + dest)) T(std::move_if_noexcept(data_[src]));
            data_[src].~T();
        }
    }

    iterator insertCount(const_iterator pos, size_type count, const T& value)
        requires std::is_copy_constructible_v<T> &&
                 std::is_nothrow_copy_constructible_v<T> &&
                 std::is_nothrow_move_constructible_v<T> {
        if (count == 0) {
            return data_ + indexFromConstIterator(pos);
        }
        const size_type index = indexFromConstIterator(pos);
        ensureCapacity(size_ + count);
        shiftRight(index, count);
        for (size_type i = 0; i < count; ++i) {
            ::new (static_cast<void*>(data_ + index + i)) T(value);
        }
        size_ += count;
        return data_ + index;
    }

    storage_type stack_storage_[has_inline_storage ? stack_capacity_value : 1];
    size_type size_{0};
    size_type capacity_{stack_capacity_value};
    pointer data_{nullptr};
};

template <typename T, std::size_t N>
void swap(SmallVector<T, N>& lhs, SmallVector<T, N>& rhs) noexcept(noexcept(lhs.swap(rhs))) {
    lhs.swap(rhs);
}

} // namespace orteaf::internal::base
