def set_sweep_config(config):
    """
    Set replication specific parameters based on the compression factor for ablation sweeps

    config: dict
        Configuration dictionary to be modified in place
    """

    compression_factor = config["optimizer"]["compression_factor"]
    replication_strategy = config["optimizer"]["repl"]

    if replication_strategy == "deto-demo":
        # Account for double bandwidth usage in Deto-Demo
        config["optimizer"]["compression_topk"] = config["optimizer"][
            "compression_chunk"
        ] // (compression_factor * 2)
    elif replication_strategy == "deto-random":
        config["optimizer"]["compression_rate"] = 1.0 / compression_factor
        if compression_factor == 1:
            raise ValueError(
                "Deto-Random with compression factor 1 is not valid for sweep."
            )
    elif replication_strategy == "deto-stride":
        config["optimizer"]["compression_rate"] = 1.0 / compression_factor
        if compression_factor == 1:
            raise ValueError(
                "Deto-Stride with compression factor 1 is not valid for sweep."
            )
    elif replication_strategy == "deto-full":
        config["optimizer"]["replicate_every"] = compression_factor
        # Use AdamW for compression factor 1 (no compression) (Baseline!)
        if compression_factor == 1:
            config["optimizer"]["optimizer_str"] = "adamw"
            config["optimizer"]["sign"] = False
    else:
        raise ValueError(
            f"Unknown replication strategy {replication_strategy} for sweep."
        )
    return config
