#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace quant {

struct TickData {
    std::string code;
    double price = 0.0;
    int64_t volume = 0;
    double amount = 0.0;
    int64_t timestamp_ns = 0;   // nanoseconds since epoch
    char direction = 0;          // 'B'=买, 'S'=卖, 'N'=中性
    char bid_ask = 0;            // 'B'=买盘, 'A'=卖盘
    char trade_type = 0;         // trade type code
};

// Shared memory ring buffer descriptor
struct ShmRingBuffer {
    static constexpr size_t CAPACITY = 65536;
    std::atomic<size_t> write_idx{0};
    std::atomic<size_t> read_idx{0};
    TickData ticks[CAPACITY];
};

class TickReceiver {
public:
    explicit TickReceiver(std::string_view multicast_addr, uint16_t port = 0);
    ~TickReceiver();

    TickData recv();  // blocking
    bool recv_nonblock(TickData& out);
    void start();
    void stop();

    // Expose ring buffer pointer for zero-copy Python access
    const ShmRingBuffer* ring_buffer() const { return shm_; }

private:
    std::string mcast_addr_;
    uint16_t port_;
    bool running_ = false;
    ShmRingBuffer* shm_ = nullptr;

    void decode_fast(const uint8_t* buf, size_t len, TickData& out);
    void decode_binary(const uint8_t* buf, size_t len, TickData& out);
};

} // namespace quant
