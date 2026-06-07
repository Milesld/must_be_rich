#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "decoder.h"

namespace py = pybind11;

PYBIND11_MODULE(tick_processor, m) {
    m.doc() = "C++ Tick行情解码器 — UDP组播接收 + FAST/二进制解码 + 共享内存";

    py::class_<quant::TickData>(m, "TickData")
        .def(py::init<>())
        .def_readwrite("code", &quant::TickData::code)
        .def_readwrite("price", &quant::TickData::price)
        .def_readwrite("volume", &quant::TickData::volume)
        .def_readwrite("amount", &quant::TickData::amount)
        .def_readwrite("timestamp_ns", &quant::TickData::timestamp_ns)
        .def_readwrite("direction", &quant::TickData::direction)
        .def_readwrite("bid_ask", &quant::TickData::bid_ask)
        .def_readwrite("trade_type", &quant::TickData::trade_type)
        .def("__repr__", [](const quant::TickData& t) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Tick(%s, price=%.2f, vol=%lld, ts=%lld)",
                     t.code.c_str(), t.price, (long long)t.volume, (long long)t.timestamp_ns);
            return std::string(buf);
        });

    py::class_<quant::TickReceiver>(m, "TickReceiver")
        .def(py::init<std::string, uint16_t>(),
             py::arg("multicast_addr"), py::arg("port") = 0)
        .def("start", &quant::TickReceiver::start)
        .def("stop", &quant::TickReceiver::stop)
        .def("recv", &quant::TickReceiver::recv)
        .def("recv_nonblock", [](quant::TickReceiver& r) -> py::object {
            quant::TickData tick;
            if (r.recv_nonblock(tick)) {
                py::dict d;
                d["code"] = tick.code;
                d["price"] = tick.price;
                d["volume"] = tick.volume;
                d["amount"] = tick.amount;
                d["timestamp_ns"] = tick.timestamp_ns;
                return std::move(d);
            }
            return py::none();
        });
}
