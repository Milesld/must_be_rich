#include "router.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <random>

namespace quant {

// Generate unique order ID
static std::string gen_order_id() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1000, 9999);
    std::ostringstream oss;
    oss << "ord_" << ms << "_" << dist(rng);
    return oss.str();
}

OrderRouter::OrderRouter() = default;
OrderRouter::~OrderRouter() = default;

OrderResult OrderRouter::submit(const OrderRequest& req) {
    if (!submit_fn_) {
        spdlog::error("submit callback not set");
        return {gen_order_id(), "rejected", 0, 0.0, "No submit callback configured"};
    }

    auto order_id = gen_order_id();

    // First attempt
    try {
        auto result = submit_fn_(req);
        result.order_id = order_id;
        if (result.status == "unknown") {
            // Retry: query status first, then decide
            if (query_fn_) {
                auto status = query_fn_(order_id);
                if (status.status == "submitted" || status.status == "filled") {
                    return status;
                }
            }
            // One retry
            spdlog::warn("order {} UNKNOWN, retrying once", order_id);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            result = submit_fn_(req);
            result.order_id = order_id;
        }
        return result;
    } catch (const std::exception& e) {
        spdlog::error("submit exception: {}", e.what());
        return {order_id, "rejected", 0, 0.0, e.what()};
    }
}

bool OrderRouter::cancel(std::string_view order_id) {
    if (!submit_fn_) return false;
    // Cancel is a special case of submit
    try {
        OrderRequest cancel_req;
        cancel_req.code = std::string(order_id);
        cancel_req.side = "cancel";
        auto result = submit_fn_(cancel_req);
        return result.status == "cancelled";
    } catch (...) {
        return false;
    }
}

OrderResult OrderRouter::query(std::string_view order_id) {
    if (!query_fn_) {
        return {std::string(order_id), "unknown", 0, 0.0, "No query callback"};
    }
    return query_fn_(order_id);
}

} // namespace quant
