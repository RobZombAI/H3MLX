#!/usr/bin/env python3
"""
Converts a Safetensors file containing FP32 tensors to native BF16 format.
Halves the file size and enables instantaneous Metal zero-copy mmap loading.
"""

import os
import sys
import struct
import json
import numpy as np

def convert_safetensors_f32_to_bf16(input_path: str, output_path: str):
    print(f"Reading {input_path}...")
    with open(input_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_len

        new_header = {}
        tensors_to_write = []
        current_offset = 0

        for key, meta in header.items():
            if key == "__metadata__":
                new_header[key] = meta
                continue

            dtype = meta["dtype"]
            shape = meta["shape"]
            offsets = meta["data_offsets"]
            num_elements = int(np.prod(shape))

            f.seek(data_start + offsets[0])
            raw_bytes = f.read(offsets[1] - offsets[0])

            if dtype == "F32":
                arr_f32 = np.frombuffer(raw_bytes, dtype=np.float32)
                # Convert float32 to bfloat16 using upper 16 bits
                arr_bf16 = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)
                bf16_bytes = arr_bf16.tobytes()
                new_dtype = "BF16"
            elif dtype == "BF16":
                bf16_bytes = raw_bytes
                new_dtype = "BF16"
            else:
                bf16_bytes = raw_bytes
                new_dtype = dtype

            tensor_len = len(bf16_bytes)
            new_header[key] = {
                "dtype": new_dtype,
                "shape": shape,
                "data_offsets": [current_offset, current_offset + tensor_len]
            }
            current_offset += tensor_len
            tensors_to_write.append(bf16_bytes)

    new_header_str = json.dumps(new_header, separators=(',', ':'))
    new_header_bytes = new_header_str.encode('utf-8')
    pad = (8 - (len(new_header_bytes) % 8)) % 8
    if pad > 0:
        new_header_str += ' ' * pad
        new_header_bytes = new_header_str.encode('utf-8')
    
    new_header_len = len(new_header_bytes)
    print(f"Writing {output_path} (new header length {new_header_len}, payload {current_offset / (1024*1024):.2f} MB)...")

    with open(output_path, "wb") as f_out:
        f_out.write(struct.pack("<Q", new_header_len))
        f_out.write(new_header_bytes)
        for t_bytes in tensors_to_write:
            f_out.write(t_bytes)

    print(f"Successfully converted {input_path} -> {output_path}")

if __name__ == "__main__":
    paths = [
        "/Users/robzomb/h3-models/MiniMax-H3-PDD-8Step/FL2VA/video_vae/source",
        "/Users/robzomb/h3-models/10Eros-Max/FL2VA/video_vae/source"
    ]
    for d in paths:
        orig = os.path.join(d, "model.safetensors")
        if os.path.exists(orig):
            bak = os.path.join(d, "model_fp32.safetensors.bak")
            tmp = os.path.join(d, "model_bf16.tmp")
            if not os.path.exists(bak):
                print(f"Backing up original to {bak}...")
                os.rename(orig, bak)
                input_file = bak
            else:
                input_file = bak
            convert_safetensors_f32_to_bf16(input_file, tmp)
            os.rename(tmp, orig)
            print(f"Replaced {orig} with native BF16 version.")
