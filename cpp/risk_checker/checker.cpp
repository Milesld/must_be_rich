#include "checker.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

namespace quant {

// ── helpers ──────────────────────────────

static double est_cost(const std::string& side, double amount) {
    double comm = std::max(amount * 0.00015, 5.0);
    double stamp = (side == "sell") ? amount * 0.0005 : 0.0;
    double transfer = amount * 0.00001;
    return comm + stamp + transfer;
}

static int board_price_limit(double pre_close, bool is_st, bool is_gem_or_star, bool is_bse) {
    double limit = 0.10;
    if (is_gem_or_star) limit = 0.20;
    else if (is_bse) limit = 0.30;
    // is_st handled via config
    (void)is_st;
    return static_cast<int>(pre_close * (1.0 + limit));
}

// ── checker ───────────────────────────

PreTradeRiskChecker::PreTradeRiskChecker(const RiskConfig& cfg) : cfg_(cfg) {}

RiskCheckResult PreTradeRiskChecker::check(const OrderRequest& req) {
    RiskCheckResult result;
    auto& d = result.checks_detail;
    std::string* reject = &result.reject_reason;

    auto apply = [&](const char* name, const auto& r) {
        d[name] = r.first;
        if (!r.first && reject->empty()) *reject = r.second;
    };

    apply("blacklist", check_blacklist(req));
    apply("suspension", check_suspension(req));
    apply("single_stock_limit", check_single_stock(req));
    apply("industry_limit", check_industry(req));
    apply("daily_order_limit", check_daily_order_count());
    apply("single_order_amount", check_single_order_amount(req));
    apply("cash_sufficient", check_cash(req));

    result.passed = reject->empty();
    return result;
}

void PreTradeRiskChecker::record_order() {
    order_count_day_++;
    order_count_recent_++;
}

void PreTradeRiskChecker::record_cancel() {
    cancel_count_recent_++;
}

void PreTradeRiskChecker::reset_daily() {
    order_count_day_ = 0;
    order_count_recent_ = 0;
    cancel_count_recent_ = 0;
}

void PreTradeRiskChecker::add_blacklist(std::string_view code) {
    cfg_.blacklist_codes.emplace(code);
    spdlog::info("blacklist added: {}", code);
}

void PreTradeRiskChecker::remove_blacklist(std::string_view code) {
    cfg_.blacklist_codes.erase(std::string(code));
}

// ── individual checks ──────────────────

std::pair<bool, std::string> PreTradeRiskChecker::check_blacklist(const OrderRequest& req) {
    if (cfg_.blacklist_codes.count(req.code))
        return {false, std::string(req.code) + " 在黑名单中"};
    if (req.is_st)
        return {false, std::string(req.code) + " 为ST/*ST股票，禁止交易"};
    for (const auto& f : req.blacklist_flags) {
        if (cfg_.blacklist_reasons.count(f))
            return {false, std::string(req.code) + ": " + f};
    }
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_suspension(const OrderRequest& req) {
    if (req.is_suspended)
        return {false, "股票处于停牌状态"};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_single_stock(const OrderRequest& req) {
    if (req.total_asset <= 0) return {true, "OK"};
    auto it = req.current_positions.find(req.code);
    int cur_shares = (it != req.current_positions.end()) ? it->second : 0;
    double price = 0.0;
    auto pit = req.current_prices.find(req.code);
    if (pit != req.current_prices.end()) price = pit->second;
    int new_shares = (req.side == "buy") ? cur_shares + req.shares : std::max(0, cur_shares - req.shares);
    double ratio = (new_shares * price) / req.total_asset;
    if (ratio > cfg_.single_stock_hard_max)
        return {false, "单票仓位" + std::to_string(ratio * 100).substr(0, 4) + "% > 硬上限" + std::to_string(int(cfg_.single_stock_hard_max * 100)) + "%"};
    if (ratio > cfg_.single_stock_max_ratio)
        return {false, "单票仓位" + std::to_string(ratio * 100).substr(0, 4) + "% > 上限" + std::to_string(int(cfg_.single_stock_max_ratio * 100)) + "%"};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_industry(const OrderRequest& req) {
    if (req.industry_map.empty() || req.total_asset <= 0) return {true, "OK"};
    auto it = req.industry_map.find(req.code);
    if (it == req.industry_map.end()) return {true, "OK"};
    const auto& ind = it->second;
    double ind_value = 0.0;
    for (const auto& [c, s] : req.current_positions) {
        auto iit = req.industry_map.find(c);
        if (iit != req.industry_map.end() && iit->second == ind) {
            auto pit = req.current_prices.find(c);
            double p = (pit != req.current_prices.end()) ? pit->second : 0.0;
            ind_value += s * p;
        }
    }
    double trade_val = req.price * req.shares;
    double new_val = (req.side == "buy") ? ind_value + trade_val : std::max(0.0, ind_value - trade_val);
    double ratio = new_val / req.total_asset;
    if (ratio > cfg_.industry_max_ratio)
        return {false, "行业'" + ind + "'仓位" + std::to_string(ratio * 100).substr(0, 4) + "% > 上限" + std::to_string(int(cfg_.industry_max_ratio * 100)) + "%"};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_daily_order_count() {
    if (order_count_day_ >= cfg_.daily_max_orders)
        return {false, "日申报笔数已达上限" + std::to_string(cfg_.daily_max_orders)};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_single_order_amount(const OrderRequest& req) {
    double amt = req.price * req.shares;
    if (amt > cfg_.single_order_max_amount)
        return {false, "单笔金额" + std::to_string(int(amt)) + " > 上限" + std::to_string(int(cfg_.single_order_max_amount))};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_cash(const OrderRequest& req) {
    if (req.side != "buy") return {true, "OK"};
    double amt = req.price * req.shares;
    double cost = est_cost(req.side, amt);
    double needed = amt + cost;
    if (req.current_cash < needed)
        return {false, "可用资金不足: 需要" + std::to_string(int(needed)) + "(含费" + std::to_string(int(cost)) + "), 可用" + std::to_string(int(req.current_cash))};
    return {true, "OK"};
}

std::pair<bool, std::string> PreTradeRiskChecker::check_price_limit(const OrderRequest& req) {
    // Simplified: uses pre_close from prices map
    auto it = req.current_prices.find(req.code);
    if (it == req.current_prices.end()) return {true, "OK"};
    double pre_close = it->second;
    if (pre_close <= 0) return {true, "OK"};
    bool is_gem = req.code.starts_with("300") || req.code.starts_with("301") || req.code.starts_with("688");
    bool is_bse = req.code.starts_with("920") || req.code.starts_with("83") || req.code.starts_with("88");
    double pct = is_gem ? 0.20 : (is_bse ? 0.30 : 0.10);
    double limit_up = pre_close * (1.0 + pct);
    double limit_down = pre_close * (1.0 - pct);
    if (req.price > limit_up * 1.001)
        return {false, "价格 " + std::to_string(req.price) + " > 涨停价 " + std::to_string(limit_up)};
    if (req.price < limit_down * 0.999)
        return {false, "价格 " + std::to_string(req.price) + " < 跌停价 " + std::to_string(limit_down)};
    return {true, "OK"};
}

} // namespace quant
