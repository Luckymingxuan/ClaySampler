#pragma once

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace clay {

class Library {
public:
    explicit Library(const std::filesystem::path& path)
    {
#ifdef _WIN32
        handle_ = path.has_parent_path()
            ? LoadLibraryExW(std::filesystem::absolute(path).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
            : LoadLibraryW(path.c_str());
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_)
            throw std::runtime_error("cannot load library " + path.u8string());
    }

    ~Library()
    {
#ifdef _WIN32
        FreeLibrary(handle_);
#else
        dlclose(handle_);
#endif
    }

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    template<class Function> Function get(const char* name) const
    {
#ifdef _WIN32
        auto address = GetProcAddress(handle_, name);
#else
        auto address = dlsym(handle_, name);
#endif
        if (!address)
            throw std::runtime_error(std::string("missing library symbol ") + name);
        Function function;
        static_assert(sizeof function == sizeof address);
        std::memcpy(&function, &address, sizeof function);
        return function;
    }

private:
#ifdef _WIN32
    HMODULE handle_{};
#else
    void* handle_{};
#endif
};

} // namespace clay
