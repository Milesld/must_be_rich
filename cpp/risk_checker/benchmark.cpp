#include "checker.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace quant;
using namespace std::chrono;

int main() {
    spdlog::set_level(spdlog::level::warn);

    RiskConfig cfg;
    cfg.single_stock_max_ratio = 0.20;
    cfg.single_stock_hard_max = 0.25;
    cfg.industry_max_ratio = 0.40;

    OrderRequest req;
    req.code = "600519";
    req.side = "buy";
    req.price = 1680.0;
    req.shares = 100;
    req.current_cash = 1'000'000.0;
    req.total_asset = 1'000'000.0;
    req.current_prices = {{"600519", 1680.0}};
    req.industry_map = {{"600519", "白酒"}};

    PreTradeRiskChecker checker(cfg);

    constexpr int N = 100'000;

    // Warmup
    for (int i = 0; i < 1000; ++i) checker.check(req);

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(N);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto t0 = high_resolution_clock::now();
        checker.check(req);
        auto t1 = high_resolution_clock::now();
        latencies.push_back(duration_cast<nanoseconds>(t1 - t0).count() / 1000.0);
    }
    auto end = high_resolution_clock::now();

    double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::sort(latencies.begin(), latencies.end());

    double p50 = latencies[N / 2];
    double p99 = latencies[(N * 99) / 100];
    double p999 = latencies[(N * 999) / 1000];
    double avg = total_ms * 1000.0 / N;

    std::cout << "=== RiskChecker C++ Benchmark (" << N << " calls) ===\n";
    std::cout << "Total:   " << total_ms << " ms\n";
    std::cout << "Avg:     " << avg << " μs\n";
    std::cout << "P50:     " << p50 << " μs\n";
    std::cout << "P99:     " << p99 << " μs  ← target < 100 μs\n";
    std::cout << "P99.9:   " << p999 << " μs\n";
    std::cout << (p99 < 100.0 ? "✓ PASS" : "✗ FAIL") << " (P99 < 100μs)\n";

    return (p99 < 100.0) ? 0 : 1;
}
