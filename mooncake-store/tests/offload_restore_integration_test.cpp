#include "real_client.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>

#include "common/network.h"
#include "tenant_id.h"
#include "test_server_helpers.h"

namespace mooncake {
namespace {

class RestoreTestEnv {
   public:
    RestoreTestEnv(const char* name, const char* value) : name_(name) {
        if (const auto* previous = std::getenv(name)) previous_ = previous;
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }
    ~RestoreTestEnv() {
        if (previous_) {
            setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

   private:
    std::string name_;
    std::optional<std::string> previous_;
};

}  // namespace

// Exercise the actual RealClient, owner RPC and FileStorage paths over TCP.
// The backend is seeded directly so the tests need no eviction timing.
class OffloadRestoreIntegrationTest : public ::testing::Test {
   protected:
    static constexpr size_t kObjectBytes = 4096;
    static constexpr size_t kObjects = 4;

    static void SetUpTestSuite() {
        google::InitGoogleLogging("OffloadRestoreIntegrationTest");
        FLAGS_logtostderr = true;
    }
    static void TearDownTestSuite() { google::ShutdownGoogleLogging(); }

    void SetUp() override {
        ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder()
                                      .set_enable_offload(true)
                                      .set_root_fs_dir("")
                                      .build()));
        char pattern[] = "/tmp/mooncake-restore-pipeline-XXXXXX";
        auto* directory = mkdtemp(pattern);
        ASSERT_NE(directory, nullptr);
        path_ = directory;
        owner_ = NewClient();
        ASSERT_NE(owner_, nullptr);

        rpc_ = std::make_unique<coro_rpc::coro_rpc_server>(1, 0, "127.0.0.1");
        rpc_->register_handler<
            &RealClient::batch_get_offload_object,
            &RealClient::batch_get_offload_object_for_transfer,
            &RealClient::release_offload_buffer>(owner_.get());
        rpc_->async_start();
        ASSERT_FALSE(rpc_->get_errc());
        owner_->local_rpc_addr = "127.0.0.1:" + std::to_string(rpc_->port());

        auto config = FileStorageConfig::FromEnvironment();
        config.storage_filepath = path_;
        config.storage_backend_type = StorageBackendType::kFilePerKey;
        config.local_buffer_size = 64 * 1024;
        config.use_uring = false;
        owner_->file_storage_ = std::make_shared<FileStorage>(
            config, owner_->client_, owner_->local_rpc_addr);
        ASSERT_TRUE(owner_->file_storage_->RegisterLocalMemory());
        ASSERT_TRUE(owner_->file_storage_->storage_backend_->Init());

        std::unordered_map<std::string, std::vector<Slice>> writes;
        for (size_t i = 0; i < kObjects; ++i) {
            values_.emplace_back(kObjectBytes, static_cast<char>('a' + i));
        }
        for (size_t i = 0; i < kObjects; ++i) {
            writes.emplace(
                TenantId::Default().MakeScopedKey(Key(i)),
                std::vector<Slice>{{values_[i].data(), kObjectBytes}});
        }
        ASSERT_TRUE(owner_->file_storage_->storage_backend_->BatchOffload(
            writes, nullptr));
    }

    std::shared_ptr<RealClient> NewClient() {
        auto client = RealClient::create();
        const auto endpoint = "127.0.0.1:" + std::to_string(getFreeTcpPort());
        if (client->setup_real(endpoint, "P2PHANDSHAKE", 0, 64 * 1024, "tcp",
                               "", master_.master_address()) != 0) {
            ADD_FAILURE() << "failed to set up TCP restore client";
            return {};
        }
        return client;
    }

    void TearDown() override {
        if (legacy_rpc_) legacy_rpc_->stop();
        if (rpc_) rpc_->stop();
        if (reader_) reader_->tearDownAll();
        reader_.reset();
        if (owner_) owner_->tearDownAll();
        owner_.reset();
        legacy_rpc_.reset();
        rpc_.reset();
        master_.Stop();
        if (!path_.empty()) std::filesystem::remove_all(path_);
    }

    static std::string Key(size_t i) { return "restore-" + std::to_string(i); }

    void CheckRead(RealClient& reader, const std::string& endpoint,
                   int64_t expected_restores) {
        std::vector<char> output(kObjects * kObjectBytes, '\0');
        ASSERT_EQ(reader.register_buffer(output.data(), output.size()), 0);
        std::unordered_map<std::string, std::vector<Slice>> objects;
        for (size_t i = 0; i < kObjects; ++i) {
            objects.emplace(
                Key(i), std::vector<Slice>{
                            {output.data() + i * kObjectBytes, kObjectBytes}});
        }
        const auto before = reader.get_offload_rpc_read_count();
        auto result =
            reader.batch_get_into_offload_object_internal(endpoint, objects);
        EXPECT_TRUE(result) << (result ? "" : toString(result.error()));
        EXPECT_EQ(reader.get_offload_rpc_read_count() - before,
                  expected_restores);
        for (size_t i = 0; i < kObjects; ++i) {
            EXPECT_EQ(
                std::string(output.data() + i * kObjectBytes, kObjectBytes),
                values_[i]);
        }
        EXPECT_EQ(reader.unregister_buffer(output.data()), 0);
    }

    void CheckOwnerReleased() {
        EXPECT_EQ(owner_->file_storage_->protected_restore_budget_.used(), 0u);
        MutexLocker lock(&owner_->file_storage_->client_buffer_mutex_);
        EXPECT_TRUE(
            owner_->file_storage_->client_buffer_allocated_batches_.empty());
    }

    void ReadLocal() {
        ASSERT_NE(owner_->offload_restore_pipeline_, nullptr);
        ASSERT_FALSE(owner_->client_->CanUseLocalMemcpy(
            owner_->client_->GetSegmentEndpoint()));
        CheckRead(*owner_, owner_->local_rpc_addr, kObjects);
        CheckOwnerReleased();
    }

    void ReadRemote(bool legacy) {
        reader_ = NewClient();
        ASSERT_NE(reader_, nullptr);
        auto endpoint = owner_->local_rpc_addr;
        if (legacy) {
            // Emulate a peer which has the old demand-read and release RPCs.
            legacy_rpc_ =
                std::make_unique<coro_rpc::coro_rpc_server>(1, 0, "127.0.0.1");
            legacy_rpc_->register_handler<&RealClient::batch_get_offload_object,
                                          &RealClient::release_offload_buffer>(
                owner_.get());
            legacy_rpc_->async_start();
            ASSERT_FALSE(legacy_rpc_->get_errc());
            endpoint = "127.0.0.1:" + std::to_string(legacy_rpc_->port());
        }
        CheckRead(*reader_, endpoint, legacy ? 2 * kObjects : kObjects);
        CheckOwnerReleased();
    }

    void ReadWithFullProtectedQuota() {
        std::string held_value(32 * 1024, 'h');
        const auto held_key = TenantId::Default().MakeScopedKey("held");
        std::unordered_map<std::string, std::vector<Slice>> writes{
            {held_key, {{held_value.data(), held_value.size()}}}};
        ASSERT_TRUE(owner_->file_storage_->storage_backend_->BatchOffload(
            writes, nullptr));
        auto held = owner_->file_storage_->BatchGetForTransfer(
            {held_key}, {static_cast<int64_t>(held_value.size())});
        ASSERT_TRUE(held);
        CheckRead(*owner_, owner_->local_rpc_addr, 2 * kObjects);
        EXPECT_EQ(owner_->file_storage_->protected_restore_budget_.used(),
                  held_value.size());
        EXPECT_TRUE(owner_->file_storage_->ReleaseBuffer(held->batch_id));
        CheckOwnerReleased();
    }

    void ReadMissingObjects() {
        std::vector<char> output(kObjectBytes * 2);
        ASSERT_EQ(owner_->register_buffer(output.data(), output.size()), 0);
        std::unordered_map<std::string, std::vector<Slice>> objects{
            {"missing-1", {{output.data(), kObjectBytes}}},
            {"missing-2", {{output.data() + kObjectBytes, kObjectBytes}}}};
        EXPECT_FALSE(owner_->batch_get_into_offload_object_internal(
            owner_->local_rpc_addr, objects));
        EXPECT_EQ(owner_->unregister_buffer(output.data()), 0);
        CheckOwnerReleased();
    }

   private:
    RestoreTestEnv bytes_{"MC_STORE_SSD_RESTORE_SUB_BATCH_BYTES", "4096"};
    RestoreTestEnv keys_{"MC_STORE_SSD_RESTORE_SUB_BATCH_KEYS", "1"};
    RestoreTestEnv workers_{"MC_STORE_SSD_RESTORE_WORKERS", "2"};
    RestoreTestEnv memcpy_{"MC_STORE_MEMCPY", "0"};
    RestoreTestEnv hugepage_{"MC_STORE_USE_HUGEPAGE", nullptr};
    testing::InProcMaster master_;
    std::string path_;
    std::vector<std::string> values_;
    std::shared_ptr<RealClient> owner_;
    std::shared_ptr<RealClient> reader_;
    std::unique_ptr<coro_rpc::coro_rpc_server> rpc_;
    std::unique_ptr<coro_rpc::coro_rpc_server> legacy_rpc_;
};

TEST_F(OffloadRestoreIntegrationTest, LocalRestoreWithMemcpyDisabled) {
    ReadLocal();
}

TEST_F(OffloadRestoreIntegrationTest, RemoteProtectedRestore) {
    ReadRemote(false);
}

TEST_F(OffloadRestoreIntegrationTest, OldOwnerFallsBackToDemandReads) {
    ReadRemote(true);
}

TEST_F(OffloadRestoreIntegrationTest, FullOwnerQuotaFallsBackToDemandReads) {
    ReadWithFullProtectedQuota();
}

TEST_F(OffloadRestoreIntegrationTest, StorageErrorReleasesReservation) {
    ReadMissingObjects();
}

}  // namespace mooncake
