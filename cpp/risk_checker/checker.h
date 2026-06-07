#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quant {

struct RiskCheckResult {
    bool passed = true;
    std::string reject_reason;
    std::unordered_map<std::string, bool> checks_detail;
    std::vector<std::string> warn_only;
};

struct OrderRequest {
    std::string code;
    std::string side;   // "buy" | "sell"
    double price = 0.0;
    int shares = 0;
    long account_id = 1;
    double current_cash = 0.0;
    double total_asset = 0.0;
    bool is_st = false;
    bool is_suspended = false;
    std::vector<std::string> blacklist_flags;
    // 当前持仓: {code: shares}
    std::unordered_map<std::string, int> current_positions;
    // 最新市价: {code: price}
    std::unordered_map<std::string, double> current_prices;
    // 行业: {code: industry}
    std::unordered_map<std::string, std::string> industry_map;
};

struct RiskConfig {
    double single_stock_max_ratio = 0.20;
    double single_stock_hard_max = 0.25;
    double industry_max_ratio = 0.40;
    int daily_max_orders = 20000;
    int second_max_orders = 300;
    double cancel_rate_max = 0.70;
    double single_order_max_amount = 500000.0;
    std::unordered_set<std::string> blacklist_reasons;
    std::unordered_set<std::string> blacklist_codes;
};

class PreTradeRiskChecker {
public:
    explicit PreTradeRiskChecker(const RiskConfig& cfg = RiskConfig{});

    RiskCheckResult check(const OrderRequest& req);

    void record_order();
    void record_cancel();
    void reset_daily();
    void add_blacklist(std::string_view code);
    void remove_blacklist(std::string_view code);

private:
    RiskConfig cfg_;
    int order_count_day_ = 0;
    int cancel_count_recent_ = 0;
    int order_count_recent_ = 0;

    // checks
    std::pair<bool, std::string> check_blacklist(const OrderRequest& req);
    std::pair<bool, std::string> check_suspension(const OrderRequest& req);
    std::pair<bool, std::string> check_single_stock(const OrderRequest& req);
    std::pair<bool, std::string> check_industry(const OrderRequest& req);
    std::pair<bool, std::string> check_daily_order_count();
    std::pair<bool, std::string> check_single_order_amount(const OrderRequest& req);
    std::pair<bool, std::string> check_cash(const OrderRequest& req);
    std::pair<bool, std::string> check_price_limit(const OrderRequest& req);

    static double estimate_cost(const std::string& side, double amount);
};

} // namespace quant
