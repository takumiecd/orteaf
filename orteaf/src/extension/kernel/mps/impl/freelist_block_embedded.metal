// Freelist that embeds `next` in the head of each free block.
#include <metal_stdlib>
using namespace metal;

struct alignas(4) NodeHeader {
    uint chunk_id;
    atomic_uint next_offset;      // offset in bytes from chunk base; 0xffffffffu = end
    uint next_chunk_id;           // chunk id of the next node (or 0xffffffffu)
};

// Initialize embedded headers across the chunk.
kernel void orteaf_freelist_init_block_embedded(
    device atomic_uint* head_offset [[buffer(0)]],
    device uint* head_chunk_id [[buffer(1)]],
    device NodeHeader* chunk_base [[buffer(2)]],
    constant uint& chunk_size_bytes [[buffer(3)]],
    constant uint& block_size_bytes [[buffer(4)]],
    device uint* out_data [[buffer(5)]],
    constant uint& chunk_id [[buffer(6)]],
    constant uint& old_head_offset [[buffer(7)]],
    constant uint& old_head_chunk_id [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    if (block_size_bytes == 0 || chunk_size_bytes < block_size_bytes) return;

    const uint num_blocks = chunk_size_bytes / block_size_bytes;
    // head points to the first block (offset 0) and chunk id
    atomic_store_explicit(head_offset, 0u, memory_order_relaxed);
    head_chunk_id[0] = chunk_id;
    out_data[0] = 0xffffffffu;
    out_data[1] = 0xffffffffu;

    // Link blocks linearly.
    for (uint i = 0; i < num_blocks; ++i) {
        const uint current_off = i * block_size_bytes;
        const uint next_off = (i + 1u < num_blocks) ? ((i + 1u) * block_size_bytes) : old_head_offset;
        const uint next_chunk = (i + 1u < num_blocks) ? chunk_id : old_head_chunk_id;
        device NodeHeader* node = reinterpret_cast<device NodeHeader*>(reinterpret_cast<device char*>(chunk_base) + current_off);
        node->chunk_id = chunk_id;
        node->next_chunk_id = next_chunk;
        atomic_store_explicit(&node->next_offset, next_off, memory_order_relaxed);
    }
}

// Pop one block: atomically advance head and write the old head offset to out_data[0].
kernel void orteaf_freelist_pop_block_embedded(
    device atomic_uint* head_offset [[buffer(0)]],
    device uint* head_chunk_id [[buffer(1)]],
    device NodeHeader* chunk_base [[buffer(2)]],
    device uint* out_data [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;

    const uint current_chunk = head_chunk_id[0];
    if (current_chunk == 0xffffffffu) {
        out_data[0] = 0xffffffffu;
        out_data[1] = 0xffffffffu;
        return;
    }

    uint observed = atomic_load_explicit(head_offset, memory_order_relaxed);
    while (true) {
        if (observed == 0xffffffffu) {
            out_data[0] = 0xffffffffu;
            out_data[1] = 0xffffffffu;
            return;
        }
        device NodeHeader* node = reinterpret_cast<device NodeHeader*>(reinterpret_cast<device char*>(chunk_base) + observed);
        const uint next = atomic_load_explicit(&node->next_offset, memory_order_relaxed);
        if (atomic_compare_exchange_weak_explicit(
                head_offset, &observed, next, memory_order_relaxed, memory_order_relaxed)) {
            head_chunk_id[0] = node->next_chunk_id;
            out_data[0] = observed;
            out_data[1] = current_chunk;
            return;
        }
        // observed updated by CAS; retry.
    }
}

// Push one block: set its next to current head and swap head to this offset.
kernel void orteaf_freelist_push_block_embedded(
    device atomic_uint* head_offset [[buffer(0)]],
    device uint* head_chunk_id [[buffer(1)]],
    device NodeHeader* chunk_base [[buffer(2)]],
    constant uint& block_offset [[buffer(3)]],
    constant uint& chunk_id [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    if (block_offset == 0xffffffffu) return;

    device NodeHeader* node = reinterpret_cast<device NodeHeader*>(reinterpret_cast<device char*>(chunk_base) + block_offset);
    node->chunk_id = chunk_id;

    uint observed_chunk = head_chunk_id[0];
    uint observed = atomic_load_explicit(head_offset, memory_order_relaxed);
    while (true) {
        node->next_chunk_id = observed_chunk;
        atomic_store_explicit(&node->next_offset, observed, memory_order_relaxed);
        if (atomic_compare_exchange_weak_explicit(
                head_offset, &observed, block_offset, memory_order_relaxed, memory_order_relaxed)) {
            head_chunk_id[0] = chunk_id;
            return;
        }
        // observed updated; loop and try again.
        observed_chunk = head_chunk_id[0];
    }
}
