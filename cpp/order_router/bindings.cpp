#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "router.h"

namespace py = pybind11;

PYBIND11_MODULE(order_router, m) {
    m.doc() = "C++ OrderRouter — QMT SDK C++封装，内置重试逻辑";

    py::class_<quant::OrderRequest>(m, "OrderRequest")
        .def(py::init<>())
        .def_readwrite("code", &quant::OrderRequest::code)
        .def_readwrite("side", &quant::OrderRequest::side)
        .def_readwrite("price", &quant::OrderRequest::price)
        .def_readwrite("shares", &quant::OrderRequest::shares)
        .def_readwrite("order_type", &quant::OrderRequest::order_type);

    py::class_<quant::OrderResult>(m, "OrderResult")
        .def(py::init<>())
        .def_readwrite("order_id", &quant::OrderResult::order_id)
        .def_readwrite("status", &quant::OrderResult::status)
        .def_readwrite("filled_shares", &quant::OrderResult::filled_shares)
        .def_readwrite("filled_price", &quant::OrderResult::filled_price)
        .def_readwrite("message", &quant::OrderResult::message)
        .def("__repr__", [](const quant::OrderResult& r) {
            return std::string("OrderResult(") + r.order_id + ", " + r.status
                 + ", filled=" + std::to_string(r.filled_shares) + ")";
        });

    py::class_<quant::OrderRouter>(m, "OrderRouter")
        .def(py::init<>())
        .def("set_submit_callback", &quant::OrderRouter::set_submit_callback)
        .def("set_query_callback", &quant::OrderRouter::set_query_callback)
        .def("submit", &quant::OrderRouter::submit)
        .def("cancel", &quant::OrderRouter::cancel)
        .def("query", &quant::OrderRouter::query);
}
