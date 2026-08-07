#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t gpnec_create(const char *domain, int32_t steps_hint);
/** Sized subspace lattice (board game); width/height in [2, 64]. Returns 0 on failure. */
uint64_t gpnec_create_subspace(int32_t width, int32_t height);
int32_t gpnec_step(uint64_t handle, int32_t count);
uint64_t gpnec_steps_executed(uint64_t handle);
int32_t gpnec_shape(uint64_t handle, int32_t *batch, int32_t *nodes, int32_t *channels);
int32_t gpnec_read_state(uint64_t handle, float *out, int32_t capacity);
int32_t gpnec_read_channel(uint64_t handle, int32_t channel, float *out, int32_t capacity);
int32_t gpnec_seed_sensor_net(uint64_t handle, const float *occupancy, const float *control, int32_t node_count);
float gpnec_control_l1(uint64_t handle);
void gpnec_destroy(uint64_t handle);
const char *gpnec_version(void);

#ifdef __cplusplus
}
#endif
