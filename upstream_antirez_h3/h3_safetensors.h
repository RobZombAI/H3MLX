#ifndef H3_SAFETENSORS_H
#define H3_SAFETENSORS_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_DTYPE_UNKNOWN = 0,
    H3_DTYPE_BOOL,
    H3_DTYPE_I8,
    H3_DTYPE_U8,
    H3_DTYPE_I16,
    H3_DTYPE_U16,
    H3_DTYPE_F16,
    H3_DTYPE_BF16,
    H3_DTYPE_I32,
    H3_DTYPE_U32,
    H3_DTYPE_F32,
    H3_DTYPE_I64,
    H3_DTYPE_U64,
    H3_DTYPE_F64
} h3_dtype;

typedef struct {
    char *name;
    h3_dtype dtype;
    int ndim;
    uint64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t file_offset;
} h3_st_tensor;

typedef struct {
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    h3_st_tensor *tensors;
    size_t tensor_count;
} h3_st_header;

int h3_st_read_header(const char *path, h3_st_header *header,
                      char *error, size_t error_size);
void h3_st_free_header(h3_st_header *header);
const h3_st_tensor *h3_st_find(const h3_st_header *header, const char *name);
uint64_t h3_st_tensor_elements(const h3_st_tensor *tensor);
int h3_st_read_data(const h3_st_header *header, const h3_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size);
size_t h3_dtype_size(h3_dtype dtype);
const char *h3_dtype_name(h3_dtype dtype);

int h3_st_inventory_dir(const char *directory, h3_component_info *info,
                        char *error, size_t error_size);

#endif
