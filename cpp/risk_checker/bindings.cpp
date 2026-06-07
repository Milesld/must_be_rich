#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "checker.h"

namespace py = pybind11;

PYBIND11_MODULE(risk_checker, m) {
    m.doc() = "C++ PreTradeRiskChecker — sub-10μs risk checks";

    py::class_<quant::RiskConfig>(m, "RiskConfig")
        .def(py::init<>())
        .def_readwrite("single_stock_max_ratio", &quant::RiskConfig::single_stock_max_ratio)
        .def_readwrite("single_stock_hard_max", &quant::RiskConfig::single_stock_hard_max)
        .def_readwrite("industry_max_ratio", &quant::RiskConfig::industry_max_ratio)
        .def_readwrite("daily_max_orders", &quant::RiskConfig::daily_max_orders)
        .def_readwrite("second_max_orders", &quant::RiskConfig::second_max_orders)
        .def_readwrite("cancel_rate_max", &quant::RiskConfig::cancel_rate_max)
        .def_readwrite("single_order_max_amount", &quant::RiskConfig::single_order_max_amount)
        .def_readwrite("blacklist_reasons", &quant::RiskConfig::blacklist_reasons)
        .def_readwrite("blacklist_codes", &quant::RiskConfig::blacklist_codes);

    py::class_<quant::OrderRequest>(m, "OrderRequest")
        .def(py::init<>())
        .def_readwrite("code", &quant::OrderRequest::code)
        .def_readwrite("side", &quant::OrderRequest::side)
        .def_readwrite("price", &quant::OrderRequest::price)
        .def_readwrite("shares", &quant::OrderRequest::shares)
        .def_readwrite("account_id", &quant::OrderRequest::account_id)
        .def_readwrite("current_cash", &quant::OrderRequest::current_cash)
        .def_readwrite("total_asset", &quant::OrderRequest::total_asset)
        .def_readwrite("is_st", &quant::OrderRequest::is_st)
        .def_readwrite("is_suspended", &quant::OrderRequest::is_suspended)
        .def_readwrite("blacklist_flags", &quant::OrderRequest::blacklist_flags)
        .def_readwrite("current_positions", &quant::OrderRequest::current_positions)
        .def_readwrite("current_prices", &quant::OrderRequest::current_prices)
        .def_readwrite("industry_map", &quant::OrderRequest::industry_map);

    py::class_<quant::RiskCheckResult>(m, "RiskCheckResult")
        .def(py::init<>())
        .def_readwrite("passed", &quant::RiskCheckResult::passed)
        .def_readwrite("reject_reason", &quant::RiskCheckResult::reject_reason)
        .def_readwrite("checks_detail", &quant::RiskCheckResult::checks_detail)
        .def_readwrite("warn_only", &quant::RiskCheckResult::warn_only)
        .def("to_dict", [](const quant::RiskCheckResult& r) {
            py::dict d;
            d["passed"] = r.passed;
            d["reject_reason"] = r.reject_reason;
            for (const auto& [k, v] : r.checks_detail) d[py::str(k)] = v;
            py::list warns;
            for (const auto& w : r.warn_only) warns.append(w);
            d["warn_only"] = warns;
            return d;
        });

    py::class_<quant::PreTradeRiskChecker>(m, "PreTradeRiskChecker")
        .def(py::init<const quant::RiskConfig&>(), py::arg("cfg") = quant::RiskConfig{})
        .def("check", &quant::PreTradeRiskChecker::check)
        .def("record_order", &quant::PreTradeRiskChecker::record_order)
        .def("record_cancel", &quant::PreTradeRiskChecker::record_cancel)
        .def("reset_daily", &quant::PreTradeRiskChecker::reset_daily)
        .def("add_blacklist", &quant::PreTradeRiskChecker::add_blacklist)
        .def("remove_blacklist", &quant::PreTradeRiskChecker::remove_blacklist);
}
