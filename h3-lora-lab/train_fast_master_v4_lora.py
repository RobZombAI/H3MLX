from h3_max_suite.trainer.miles_sglang_trainer import H3MaxTrainConfig, run_miles_training

def main():
    print("🚀 Training / Calibrating Fast Master V4 LoRA Adapter (Rank 128, Alpha 256.0)...")
    config = H3MaxTrainConfig(
        lora_name="minimax_h3_fast_master_v4_rank128_alpha256.safetensors",
        lora_rank=128,
        lora_alpha=256.0,
        learning_rate=2e-5,
        fsdp_flow_shift=12.0,
        num_train_steps=40
    )
    run_miles_training(config)
    print("🎉 Fast Master V4 LoRA Training & Calibration completed successfully!")

if __name__ == "__main__":
    main()
