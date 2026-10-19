#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "thread_pool.h"

namespace mooncake {

class Environ;

struct OffloadRestorePipelineConfig {
    uint64_t sub_batch_bytes = 0;  // Zero preserves the existing read path.
    size_t max_keys = 32;
    size_t workers = 2;

    static OffloadRestorePipelineConfig FromEnvironment(const Environ& env);
};

struct OffloadSubBatch {
    size_t begin;
    size_t end;
};

// Oversized objects keep the original whole-request path. A range always
// contains complete objects, in the order supplied by the caller.
inline std::vector<OffloadSubBatch> PlanOffloadSubBatches(
    const std::vector<int64_t>& sizes, uint64_t target_bytes, size_t max_keys) {
    if (sizes.empty()) return {};
    if (target_bytes == 0 || max_keys == 0 ||
        std::any_of(sizes.begin(), sizes.end(), [target_bytes](int64_t size) {
            return size < 0 || static_cast<uint64_t>(size) > target_bytes;
        })) {
        return {{0, sizes.size()}};
    }
    std::vector<OffloadSubBatch> result;
    size_t begin = 0;
    uint64_t bytes = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto size = static_cast<uint64_t>(sizes[i]);
        if (i != begin &&
            (i - begin == max_keys || size > target_bytes - bytes)) {
            result.push_back({begin, i});
            begin = i;
            bytes = 0;
        }
        bytes += size;
    }
    result.push_back({begin, sizes.size()});
    return result;
}

// Shared reservation state survives the object that issued a reservation.
// Used both for requester admission and the owner's protected staging quota.
class OffloadRestoreBudget {
   public:
    explicit OffloadRestoreBudget(uint64_t limit)
        : limit_(limit), used_(std::make_shared<std::atomic<uint64_t>>(0)) {}

    std::shared_ptr<void> TryAcquire(uint64_t bytes) {
        auto used = used_->load(std::memory_order_relaxed);
        do {
            if (bytes > limit_ || used > limit_ - bytes) return {};
        } while (!used_->compare_exchange_weak(used, used + bytes,
                                               std::memory_order_relaxed));
        return std::shared_ptr<void>(
            used_.get(), [counter = used_, bytes](void*) {
                counter->fetch_sub(bytes, std::memory_order_relaxed);
            });
    }

    uint64_t used() const { return used_->load(std::memory_order_relaxed); }

   private:
    uint64_t limit_;
    std::shared_ptr<std::atomic<uint64_t>> used_;
};

// One look-ahead load per admitted request; destination transfers stay on the
// calling thread, preserving its accelerator context. There are no per-read
// threads and no loader task waits for another task in the same pool.
class OffloadRestorePipeline {
   public:
    explicit OffloadRestorePipeline(size_t workers)
        : admission_(workers), pool_(workers) {}

    std::shared_ptr<void> TryEnter() { return admission_.TryAcquire(1); }

    template <typename Batch, typename Load, typename Consume, typename Result>
    Result Run(const std::vector<Batch>& batches, Load load, Consume consume,
               Result success) {
        if (batches.empty()) return success;
        using Loaded = std::invoke_result_t<Load, const Batch&>;
        auto current = load(batches.front());
        for (size_t i = 0; i < batches.size(); ++i) {
            std::future<Loaded> pending;
            // std::future from a promise does not join on destruction. Drain
            // explicitly on errors/exceptions before the callbacks can die.
            struct Drain {
                std::future<Loaded>& future;
                ~Drain() {
                    if (future.valid()) {
                        try {
                            // Move out and destroy the unused buffer here,
                            // before caller-owned state can be destroyed.
                            (void)future.get();
                        } catch (...) {
                            // Preserve the original failure while draining.
                        }
                    }
                }
            } drain{pending};
            if (current && i + 1 < batches.size()) {
                auto promise = std::make_shared<std::promise<Loaded>>();
                pending = promise->get_future();
                auto task = [&, promise, next = i + 1] {
                    try {
                        promise->set_value(load(batches[next]));
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                };
                try {
                    pool_.enqueue(task);
                } catch (...) {
                    // A stopped/unavailable pool can still make progress.
                    task();
                }
            }
            auto result = consume(batches[i], std::move(current));
            if (!result) return result;
            if (i + 1 == batches.size()) return result;
            current = pending.valid() ? pending.get() : load(batches[i + 1]);
        }
        return success;
    }

   private:
    OffloadRestoreBudget admission_;
    ThreadPool pool_;
};

}  // namespace mooncake
