#include "offload_restore_pipeline.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>

#include "environ.h"

namespace mooncake {
namespace {

using namespace std::chrono_literals;

TEST(OffloadRestorePipelineTest, PlansWholeObjectsByBytesAndKeyCount) {
    auto batches = PlanOffloadSubBatches({3, 4, 2, 1}, 7, 2);
    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0].begin, 0);
    EXPECT_EQ(batches[0].end, 2);
    EXPECT_EQ(batches[1].begin, 2);
    EXPECT_EQ(batches[1].end, 4);
    batches = PlanOffloadSubBatches({0, 0, 0}, 7, 2);
    ASSERT_EQ(batches.size(), 2);
    EXPECT_EQ(batches[0].end, 2);
}

TEST(OffloadRestorePipelineTest, OversizedOrDisabledUsesOriginalBatch) {
    for (const auto bytes : {uint64_t{0}, uint64_t{4}}) {
        auto batches = PlanOffloadSubBatches({1, 8, 1}, bytes, 2);
        ASSERT_EQ(batches.size(), 1);
        EXPECT_EQ(batches[0].end, 3);
    }
    EXPECT_TRUE(PlanOffloadSubBatches({}, 4, 2).empty());
    const auto max = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(PlanOffloadSubBatches({max, max}, max, 32).size(), 2);
    EXPECT_EQ(PlanOffloadSubBatches({-1, 1}, 4, 2).size(), 1);
}

TEST(OffloadRestorePipelineTest, ConfigDefaultsDisabledAndParsesOptIn) {
    MapEnvironSource source;
    const Environ env(source);
    EXPECT_EQ(
        OffloadRestorePipelineConfig::FromEnvironment(env).sub_batch_bytes, 0);
    source.Set("MC_STORE_SSD_RESTORE_SUB_BATCH_BYTES", "1048576");
    source.Set("MC_STORE_SSD_RESTORE_SUB_BATCH_KEYS", "16");
    source.Set("MC_STORE_SSD_RESTORE_WORKERS", "4");
    const auto config = OffloadRestorePipelineConfig::FromEnvironment(env);
    EXPECT_EQ(config.sub_batch_bytes, 1048576);
    EXPECT_EQ(config.max_keys, 16);
    EXPECT_EQ(config.workers, 4);
    for (const auto* invalid : {"-1", "overflow", "18446744073709551616"}) {
        source.Set("MC_STORE_SSD_RESTORE_SUB_BATCH_BYTES", invalid);
        EXPECT_EQ(
            OffloadRestorePipelineConfig::FromEnvironment(env).sub_batch_bytes,
            0);
    }
}

TEST(OffloadRestorePipelineTest, InvalidPoolSizeDisablesFeature) {
    MapEnvironSource source{{"MC_STORE_SSD_RESTORE_SUB_BATCH_BYTES", "4096"}};
    const Environ env(source);
    for (const auto* invalid : {"0", "17", "18446744073709551615"}) {
        source.Set("MC_STORE_SSD_RESTORE_WORKERS", invalid);
        EXPECT_EQ(
            OffloadRestorePipelineConfig::FromEnvironment(env).sub_batch_bytes,
            0);
    }
    source.Set("MC_STORE_SSD_RESTORE_WORKERS", "2");
    source.Set("MC_STORE_SSD_RESTORE_SUB_BATCH_KEYS", "0");
    EXPECT_EQ(
        OffloadRestorePipelineConfig::FromEnvironment(env).sub_batch_bytes, 0);
}

TEST(OffloadRestorePipelineTest, QuotaIsSharedAndReturnedExactlyOnce) {
    OffloadRestoreBudget budget(16);
    auto first = budget.TryAcquire(8);
    ASSERT_TRUE(first);
    auto shared = first;
    EXPECT_FALSE(budget.TryAcquire(9));
    auto second = budget.TryAcquire(8);
    ASSERT_TRUE(second);
    EXPECT_EQ(budget.used(), 16);
    first.reset();
    EXPECT_FALSE(budget.TryAcquire(1));
    shared.reset();
    EXPECT_EQ(budget.used(), 8);
    second.reset();
    EXPECT_EQ(budget.used(), 0);
}

TEST(OffloadRestorePipelineTest, ReservationMayOutliveQuotaOwner) {
    std::shared_ptr<void> reservation;
    {
        OffloadRestoreBudget budget(1);
        reservation = budget.TryAcquire(1);
        ASSERT_TRUE(reservation);
    }
    reservation.reset();  // Also checked by ASan.
}

TEST(OffloadRestorePipelineTest, AdmissionBoundsConcurrentRequests) {
    OffloadRestorePipeline pipeline(2);
    auto first = pipeline.TryEnter();
    auto second = pipeline.TryEnter();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_FALSE(pipeline.TryEnter());
    first.reset();
    EXPECT_TRUE(pipeline.TryEnter());
}

TEST(OffloadRestorePipelineTest, SharedPoolProgressesAtAdmissionLimit) {
    OffloadRestorePipeline pipeline(2);
    std::latch admitted(4);
    std::atomic<int> accepted{0};
    std::atomic<int> completed{0};
    std::vector<std::thread> callers;
    for (int request = 0; request < 4; ++request) {
        callers.emplace_back([&] {
            auto token = pipeline.TryEnter();
            if (token) ++accepted;
            admitted.arrive_and_wait();
            if (!token) return;
            const auto calling_thread = std::this_thread::get_id();
            auto load = [](const int& value) {
                return std::make_unique<int>(value);
            };
            auto consume = [&](const int& value, std::unique_ptr<int> loaded) {
                EXPECT_EQ(*loaded, value);
                EXPECT_EQ(std::this_thread::get_id(), calling_thread);
                return true;
            };
            EXPECT_TRUE(
                pipeline.Run(std::vector<int>{0, 1, 2}, load, consume, true));
            ++completed;
        });
    }
    for (auto& caller : callers) caller.join();
    EXPECT_EQ(accepted, 2);
    EXPECT_EQ(completed, 2);
    EXPECT_TRUE(pipeline.TryEnter());
}

struct LoadedBuffer {
    std::shared_ptr<void> reservation;
    std::string bytes;
};

TEST(OffloadRestorePipelineTest, LoadsNextWhileTransferringOnCallingThread) {
    OffloadRestorePipeline pipeline(2);
    constexpr size_t kBytes = 4096;
    OffloadRestoreBudget budget(2 * kBytes);
    const std::vector<int> batches{0, 1, 2, 3};
    const auto caller = std::this_thread::get_id();
    std::promise<void> second_ready;
    auto ready = second_ready.get_future();
    std::vector<std::string> destination(batches.size());
    std::atomic<int> loaded{0};
    auto load = [&](const int& batch) {
        auto result = std::make_unique<LoadedBuffer>();
        result->reservation = budget.TryAcquire(kBytes);
        EXPECT_TRUE(result->reservation);
        result->bytes = std::string(kBytes, static_cast<char>('a' + batch));
        ++loaded;
        if (batch == 1) second_ready.set_value();
        return result;
    };
    auto consume = [&](const int& batch, std::unique_ptr<LoadedBuffer> source) {
        EXPECT_EQ(std::this_thread::get_id(), caller);
        EXPECT_TRUE(source->reservation);
        if (batch == 0) {
            // This would time out if the two stages were serialized.
            EXPECT_EQ(ready.wait_for(2s), std::future_status::ready);
            EXPECT_EQ(budget.used(), 2 * kBytes);
        }
        destination[batch].resize(kBytes);
        std::memcpy(destination[batch].data(), source->bytes.data(), kBytes);
        return true;
    };
    EXPECT_TRUE(pipeline.Run(batches, load, consume, true));
    EXPECT_EQ(loaded, 4);
    EXPECT_EQ(budget.used(), 0);
    for (size_t i = 0; i < destination.size(); ++i) {
        EXPECT_EQ(destination[i],
                  std::string(kBytes, static_cast<char>('a' + i)));
    }
}

TEST(OffloadRestorePipelineTest, FailureDrainsAndReleasesUnusedLookahead) {
    OffloadRestorePipeline pipeline(1);
    OffloadRestoreBudget budget(2);
    std::promise<void> second_started;
    auto started = second_started.get_future();
    std::promise<void> finish_second;
    auto finish = finish_second.get_future();
    std::atomic<int> loads{0};
    std::atomic<int> transfers{0};
    auto load = [&](const int& batch) {
        ++loads;
        auto buffer = std::make_unique<LoadedBuffer>();
        buffer->reservation = budget.TryAcquire(1);
        if (batch == 1) {
            second_started.set_value();
            finish.wait();
        }
        return buffer;
    };
    auto consume = [&](const int&, std::unique_ptr<LoadedBuffer>) {
        ++transfers;
        EXPECT_EQ(started.wait_for(2s), std::future_status::ready);
        finish_second.set_value();
        return false;
    };
    EXPECT_FALSE(pipeline.Run(std::vector<int>{0, 1, 2}, load, consume, true));
    EXPECT_EQ(loads, 2);
    EXPECT_EQ(transfers, 1);
    EXPECT_EQ(budget.used(), 0);  // Drain releases the completed value too.
}

TEST(OffloadRestorePipelineTest, ConsumerExceptionDrainsBeforeUnwinding) {
    OffloadRestorePipeline pipeline(1);
    OffloadRestoreBudget budget(2);
    auto load = [&](const int&) {
        auto buffer = std::make_unique<LoadedBuffer>();
        buffer->reservation = budget.TryAcquire(1);
        return buffer;
    };
    auto consume = [](const int&, std::unique_ptr<LoadedBuffer>) -> bool {
        throw std::runtime_error("transfer failure");
    };
    EXPECT_THROW(pipeline.Run(std::vector<int>{0, 1}, load, consume, true),
                 std::runtime_error);
    EXPECT_EQ(budget.used(), 0);
}

TEST(OffloadRestorePipelineTest, LoaderExceptionReachesCaller) {
    OffloadRestorePipeline pipeline(1);
    auto load = [](const int& batch) {
        if (batch == 1) throw std::runtime_error("read failure");
        return std::make_unique<int>(batch);
    };
    int transfers = 0;
    auto consume = [&](const int&, std::unique_ptr<int>) {
        ++transfers;
        return true;
    };
    EXPECT_THROW(pipeline.Run(std::vector<int>{0, 1, 2}, load, consume, true),
                 std::runtime_error);
    EXPECT_EQ(transfers, 1);
}

TEST(OffloadRestorePipelineTest, FailedLoadIsConsumedWithoutPrefetchingMore) {
    OffloadRestorePipeline pipeline(1);
    int loads = 0;
    auto load = [&](const int&) {
        ++loads;
        return std::unique_ptr<int>{};
    };
    auto consume = [](const int&, std::unique_ptr<int> value) {
        return value != nullptr;
    };
    EXPECT_FALSE(pipeline.Run(std::vector<int>{0, 1, 2}, load, consume, true));
    EXPECT_EQ(loads, 1);
}

TEST(OffloadRestorePipelineTest, FailedProtectedLoadMayFallBackSynchronously) {
    OffloadRestorePipeline pipeline(1);
    int transfers = 0;
    auto load = [](const int&) { return std::unique_ptr<int>{}; };
    auto consume = [&](const int&, std::unique_ptr<int> value) {
        EXPECT_FALSE(value);
        ++transfers;  // Immediate demand-read fallback.
        return true;
    };
    EXPECT_TRUE(pipeline.Run(std::vector<int>{0, 1, 2}, load, consume, true));
    EXPECT_EQ(transfers, 3);
}

}  // namespace
}  // namespace mooncake
