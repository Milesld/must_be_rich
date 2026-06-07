cmake_minimum_required(VERSION 3.16)
project(quant_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Git version
execute_process(COMMAND git describe --tags --always --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GIT_VERSION)
    add_definitions(-DQUANT_VERSION="${GIT_VERSION}")
else()
    add_definitions(-DQUANT_VERSION="0.1.0")
endif()

# pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# ── risk_checker ──────────────────────
pybind11_add_module(risk_checker
    risk_checker/checker.cpp
    risk_checker/bindings.cpp
)
target_link_libraries(risk_checker PRIVATE spdlog::spdlog)
target_include_directories(risk_checker PRIVATE ${CMAKE_SOURCE_DIR})

# ── tick_processor ────────────────────
pybind11_add_module(tick_processor
    tick_processor/decoder.cpp
    tick_processor/bindings.cpp
)
target_link_libraries(tick_processor PRIVATE spdlog::spdlog)

# ── order_router ──────────────────────
pybind11_add_module(order_router
    order_router/router.cpp
    order_router/bindings.cpp
)
target_link_libraries(order_router PRIVATE spdlog::spdlog)

# Benchmark target
add_executable(risk_benchmark
    risk_checker/benchmark.cpp
    risk_checker/checker.cpp
)
target_link_libraries(risk_benchmark PRIVATE spdlog::spdlog)
