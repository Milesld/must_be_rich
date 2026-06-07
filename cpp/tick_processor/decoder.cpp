#include "decoder.h"
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <thread>

namespace quant {

// ── shared memory helpers ──────────────

static ShmRingBuffer* create_shm() {
    const char* name = "/quant_tick_shm";
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        spdlog::error("shm_open failed: {}", strerror(errno));
        return nullptr;
    }
    ftruncate(fd, sizeof(ShmRingBuffer));
    auto* shm = static_cast<ShmRingBuffer*>(
        mmap(nullptr, sizeof(ShmRingBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    );
    close(fd);
    if (shm == MAP_FAILED) {
        spdlog::error("mmap failed: {}", strerror(errno));
        return nullptr;
    }
    new (shm) ShmRingBuffer();
    return shm;
}

// ── TickReceiver ──────────────────────

TickReceiver::TickReceiver(std::string_view addr, uint16_t port)
    : mcast_addr_(addr), port_(port)
{
    shm_ = create_shm();
}

TickReceiver::~TickReceiver() {
    stop();
    if (shm_) {
        munmap(shm_, sizeof(ShmRingBuffer));
        shm_open("/quant_tick_shm", O_RDONLY, 0666); // unlink placeholder
    }
}

void TickReceiver::start() {
    running_ = true;
    // Background thread: listen UDP multicast, decode, push to ring buffer
    std::thread([this]() {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) { spdlog::error("socket failed"); return; }

        int reuse = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(sock, (sockaddr*)&addr, sizeof(addr));

        ip_mreq mreq{};
        inet_pton(AF_INET, mcast_addr_.c_str(), &mreq.imr_multiaddr);
        mreq.imr_interface.s_addr = INADDR_ANY;
        setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));

        uint8_t buf[4096];
        while (running_) {
            ssize_t n = recvfrom(sock, buf, sizeof(buf), 0, nullptr, nullptr);
            if (n <= 0) continue;

            TickData tick;
            decode_binary(buf, static_cast<size_t>(n), tick);

            auto idx = shm_->write_idx.load(std::memory_order_acquire);
            shm_->ticks[idx % ShmRingBuffer::CAPACITY] = tick;
            shm_->write_idx.store(idx + 1, std::memory_order_release);
        }
        close(sock);
    }).detach();
}

void TickReceiver::stop() {
    running_ = false;
}

TickData TickReceiver::recv() {
    TickData out{};
    auto write = shm_->write_idx.load(std::memory_order_acquire);
    auto& read = shm_->read_idx;
    while (read.load(std::memory_order_acquire) >= write) {
        // spin-wait (production: use futex/condition_variable)
    }
    auto idx = read.load(std::memory_order_acquire);
    out = shm_->ticks[idx % ShmRingBuffer::CAPACITY];
    read.store(idx + 1, std::memory_order_release);
    return out;
}

bool TickReceiver::recv_nonblock(TickData& out) {
    auto write = shm_->write_idx.load(std::memory_order_acquire);
    auto& read = shm_->read_idx;
    auto cur = read.load(std::memory_order_acquire);
    if (cur >= write) return false;
    out = shm_->ticks[cur % ShmRingBuffer::CAPACITY];
    read.store(cur + 1, std::memory_order_release);
    return true;
}

// Simplified decoders (full FAST/binary spec requires exchange documentation)
void TickReceiver::decode_fast(const uint8_t* buf, size_t len, TickData& out) {
    (void)len;
    out.code = "000000";
    out.price = 0.0;
    // Production: full FAST (FIX Adapted for Streaming) decoder
}

void TickReceiver::decode_binary(const uint8_t* buf, size_t len, TickData& out) {
    if (len < 32) return;
    // Simplified SZSE binary format decoder
    memcpy(&out.timestamp_ns, buf, 8);
    memcpy(&out.price, buf + 8, 8);
    memcpy(&out.volume, buf + 16, 8);
    memcpy(&out.amount, buf + 24, 8);
    out.code = std::string(reinterpret_cast<const char*>(buf + 32), 6);
    out.direction = static_cast<char>(buf[38]);
}

} // namespace quant
