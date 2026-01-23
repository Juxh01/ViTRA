def set_sweep_config(config):
    """
    Set replication specific parameters based on the compression factor for ablation sweeps

    config: dict
        Configuration dictionary to be modified in place
    """

    compression_factor = config["optimizer"]["compression_factor"]
    replication_strategy = config["optimizer"]["repl"]

    if replication_strategy == "deto-demo":
        config["optimizer"]["compression_topk"] = (
            config["optimizer"]["compression_chunk"] // compression_factor
        )
    elif replication_strategy == "deto-random":
        config["optimizer"]["compression_rate"] = 1.0 / compression_factor
    elif replication_strategy == "deto-stride":
        config["optimizer"]["compression_rate"] = 1.0 / compression_factor
    elif replication_strategy == "deto-full":
        config["optimizer"]["replicate_every"] = compression_factor
    else:
        raise ValueError(
            f"Unknown replication strategy {replication_strategy} for sweep."
        )
    return config
