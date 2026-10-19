#include "offload_restore_pipeline.h"

#include "environ.h"
#include "environment_variables.h"

namespace mooncake {

OffloadRestorePipelineConfig OffloadRestorePipelineConfig::FromEnvironment(
    const Environ& env) {
    using Vars = OffloadRestoreEnvironmentVariables;
    OffloadRestorePipelineConfig config;
    config.sub_batch_bytes =
        env.GetTypedOr(Vars::MC_STORE_SSD_RESTORE_SUB_BATCH_BYTES, uint64_t{0});
    config.max_keys =
        env.GetTypedOr(Vars::MC_STORE_SSD_RESTORE_SUB_BATCH_KEYS, uint64_t{32});
    config.workers =
        env.GetTypedOr(Vars::MC_STORE_SSD_RESTORE_WORKERS, uint64_t{2});
    if (config.max_keys == 0 || config.workers == 0 || config.workers > 16) {
        // Fail closed rather than constructing a zero-worker pool or
        // unexpectedly creating an unbounded number of loader threads.
        config.sub_batch_bytes = 0;
    }
    return config;
}

}  // namespace mooncake
