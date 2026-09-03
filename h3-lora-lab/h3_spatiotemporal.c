#include "h3_spatiotemporal.h"
#include <string.h>

int h3_nax_st_init(h3_nax_st_layout *layout,
                   uint32_t sequence,
                   uint32_t text_rows,
                   uint32_t audio_rows,
                   uint32_t latent_t,
                   uint32_t latent_h,
                   uint32_t latent_w,
                   uint32_t chunk_frames,
                   uint32_t keyframe_stride) {
    if (!layout) return 0;
    memset(layout, 0, sizeof(*layout));
    
    layout->text_rows = text_rows;
    layout->audio_rows = audio_rows;
    layout->latent_t = latent_t;
    layout->spatial_tokens = (latent_h > 0 && latent_w > 0) ? (latent_h * latent_w) : 1536;
    layout->video_rows = latent_t * layout->spatial_tokens;
    layout->total_sequence = sequence;
    
    layout->chunk_frames = (chunk_frames > 0) ? chunk_frames : 4;
    layout->chunk_tokens = layout->chunk_frames * layout->spatial_tokens;
    layout->num_chunks = (latent_t + layout->chunk_frames - 1) / layout->chunk_frames;
    layout->keyframe_stride = (keyframe_stride > 0) ? keyframe_stride : 4;
    
    // Auto-enable when sequence is long enough to benefit from linear scaling (> 4s, i.e. > 30k tokens)
    layout->enabled = (sequence > 30000);
    return 1;
}
