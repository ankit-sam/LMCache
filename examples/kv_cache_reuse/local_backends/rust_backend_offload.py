# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict
import argparse
import contextlib
import json
import os
import time

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def _build_per_tp_device_mapping(
    per_tp_device_paths: list[str] | None,
) -> dict[str, str]:
    """Build TP-rank to device mapping from positional CLI inputs."""
    if not per_tp_device_paths:
        return {}

    mapping = {}
    seen_paths = set()
    for rank, path in enumerate(per_tp_device_paths):
        if path in seen_paths:
            raise ValueError(
                "Duplicate device path in --per_tp_device_paths: "
                f"{path}. Each TP rank must use a unique device path."
            )
        seen_paths.add(path)
        mapping[str(rank)] = path
    return mapping


def setup_environment_variables(
    raw_block_path: str,
    use_uring: bool = False,
    use_fdp: bool = False,
    max_data_transfer_size: int = 0,
    tp: int = 1,
    per_tp_device_paths: list[str] | None = None,
) -> None:
    """Set up LMCache-related environment variables for the Rust raw block backend.

    Configures environment variables for LMCache including chunk size, storage
    plugins, and Rust raw block backend specific settings.

    Args:
        raw_block_path: Path to the raw block device for storage. This is used
            directly for TP=1 and left empty for TP>1 per-rank device mapping.
        use_uring: Whether to enable io_uring path.
        use_fdp: Whether to enable FDP directive for writes.
        max_data_transfer_size: Maximum transfer size in bytes for each I/O
            request. `0` disables splitting, `-1` auto-detects from the NVMe
            queue limit, and a positive value forces an explicit split size.
        tp: Tensor parallel size.
        per_tp_device_paths: Optional per-TP device paths.

    Returns:
        None
    """
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Disable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "False"

    # Set the maximum size of the local disk size to 5GB
    os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "5"

    os.environ["LMCACHE_STORAGE_PLUGINS"] = "raw_block"
    if tp > 1:
        # Keep Python hash behavior deterministic across TP workers.
        os.environ["PYTHONHASHSEED"] = "0"

    per_tp_device_mapping = _build_per_tp_device_mapping(per_tp_device_paths)

    # Raw block specific extra config
    os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(
        {
            "storage_plugin.raw_block.module_path": "lmcache.v1.storage_backend.plugins.rust_raw_block_backend",  # noqa: E501
            "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
            "rust_raw_block.device_path": raw_block_path,
            # NVMe character devices in FDP mode do not require O_DIRECT.
            "rust_raw_block.use_odirect": not use_fdp,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.use_uring": use_uring or use_fdp,
            "rust_raw_block.use_uring_cmd": use_fdp,
            "rust_raw_block.use_fdp": use_fdp,
            "rust_raw_block.max_data_transfer_size": max_data_transfer_size,
            "rust_raw_block.per_tp_device_paths": (
                per_tp_device_mapping if tp > 1 else {}
            ),
        }
    )


@contextlib.contextmanager
def build_llm_with_lmcache(
    lmcache_connector: str,
    model: str,
    tp: int = 1,
):
    """Build a vLLM LLM instance with LMCache integration.

    Creates a context manager that builds a vLLM LLM instance configured with
    LMCache for KV cache management. The LLM is yielded and cleaned up on exit.

    Args:
        lmcache_connector: The LMCache connector name to use
        model: The model name.
    """
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    # Set GPU memory utilization to 0.5 for an A100 GPU with 40GB
    # memory. Update it accordingly for different GPU.
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tp,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up the LMCache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
) -> None:
    """Generate text using the LLM and print the output with timing information.

    Args:
        llm: The vLLM LLM instance to use for generation.
        prompt: The input prompt(s) as a list of strings.
        sampling_params: Sampling parameters for generation.
        req_str: A string identifier for the request.

    Returns:
        None
    """
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - disk_path: Path to the raw block device for storage, including
              single-namespace FDP runs.
            - per_tp_device_paths: Per-TP raw block device paths.
            - use_uring: Whether to enable io_uring path.
            - use_fdp: Whether to enable FDP directive for writes.
    """
    parser = argparse.ArgumentParser()
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument(
        "--disk_path",
        type=str,
    )
    device_group.add_argument(
        "--per_tp_device_paths",
        nargs="+",
        default=None,
        help=(
            "Optional per-TP device paths in rank order "
            "('/dev/ng0n1 /dev/ng0n2' -> rank 0/1)."
        ),
    )
    parser.add_argument(
        "--use_uring",
        action="store_true",
        help="Enable io_uring path (requires Linux kernel >= 5.1)",
    )
    parser.add_argument(
        "--use_fdp",
        action="store_true",
        help="Enable FDP write directives.",
    )
    parser.add_argument(
        "--max-data-transfer-size",
        type=int,
        default=0,
        help=(
            "Maximum data transfer size for io_uring "
            "(0 = no splitting, -1 = auto from max_hw_sectors_kb, "
            "> 0 = explicit split size)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Rust backend offload example.

    Sets up environment variables, builds an LLM with LMCache integration,
    and runs two requests with a shared prefix to demonstrate KV cache reuse.

    Returns:
        None
    """
    args = parse_args()
    per_tp_device_mapping = _build_per_tp_device_mapping(args.per_tp_device_paths)
    tp = len(per_tp_device_mapping) if per_tp_device_mapping else 1
    if tp == 1 and args.disk_path is None:
        raise ValueError("--disk_path is required when TP=1")
    if tp > 1:
        # In TP>1 mode, the backend selects rank-local paths from
        # rust_raw_block.per_tp_device_paths.
        args.disk_path = ""

    connector = "LMCacheConnectorV1"
    model = "Qwen/Qwen3-8B"

    setup_environment_variables(
        args.disk_path,
        args.use_uring,
        args.use_fdp,
        args.max_data_transfer_size,
        tp,
        args.per_tp_device_paths,
    )

    with build_llm_with_lmcache(connector, model, tp) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        shared_prompt = "Hello, how are you?" * 1000
        first_prompt = [
            shared_prompt + "Hello, my name is",
        ]
        second_prompt = [
            shared_prompt + "Tell me a very long story",
        ]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")


if __name__ == "__main__":
    main()
