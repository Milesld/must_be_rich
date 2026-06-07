#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <functional>
#include <optional>

namespace quant {

struct OrderRequest {
    std::string code;
    std::string side;   // "buy" | "sell"
    double price = 0.0;
    int shares = 0;
    std::string order_type = "limit";  // "limit" | "market"
};

struct OrderResult {
    std::string order_id;
    std::string status;       // "submitted" | "filled" | "cancelled" | "rejected" | "unknown"
    int filled_shares = 0;
    double filled_price = 0.0;
    std::string message;
};

using SubmitFn = std::function<OrderResult(const OrderRequest&)>;
using QueryFn = std::function<OrderResult(std::string_view order_id)>;

class OrderRouter {
public:
    OrderRouter();
    ~OrderRouter();

    // Register callbacks (injected by Python side — connects to QMT SDK)
    void set_submit_callback(SubmitFn fn) { submit_fn_ = std::move(fn); }
    void set_query_callback(QueryFn fn) { query_fn_ = std::move(fn); }

    // Submit order with retry logic.
    // Timeout: 100ms per attempt. Failure → retry once (check status first).
    OrderResult submit(const OrderRequest& req);

    // Cancel an order.
    bool cancel(std::string_view order_id);

    // Query order status.
    OrderResult query(std::string_view order_id);

private:
    SubmitFn submit_fn_;
    QueryFn query_fn_;
    int timeout_ms_ = 100;
    int max_retries_ = 1;
};

} // namespace quant
