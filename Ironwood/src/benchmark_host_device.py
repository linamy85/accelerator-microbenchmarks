"""Benchmarks Host-to-Device and Device-to-Host transfer performance (Simple Baseline)."""

import time
import os
import concurrent.futures
import gc
from typing import Any, Dict, Tuple, List

import jax
import jax.numpy as jnp
from jax import sharding
import numpy as np
import contextlib
from benchmark_utils import MetricsStatistics


libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
# 64 GiB
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"


libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
# 64 GiB
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"

# --- Smart Chunking Implemention Helpers ---

def _run_h2d_chunked(host_shards, target_devices, num_devices, chunks_per_device, data_sharding):
  chk_h2d_start = time.perf_counter()
  total_workers = num_devices * chunks_per_device
  with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
    chunked_futures = []
    for shard, dev in zip(host_shards, target_devices):
      sub_chunks = np.array_split(shard, chunks_per_device, axis=0)
      for chunk in sub_chunks:
        chunked_futures.append(
            executor.submit(jax.device_put, chunk, dev)
        )
    chunked_buffers = [f.result() for f in chunked_futures]

    # Reconstruct the full array on device to ensure usability
    # This involves a D2D copy/concat
    # First, concatenate chunks per device (local concat)
    per_device_arrays = []
    for i in range(num_devices):
      start = i * chunks_per_device
      end = start + chunks_per_device
      per_device_arrays.append(jnp.concatenate(chunked_buffers[start:end], axis=0))

    expected_shape = (sum(s.shape[0] for s in host_shards),) + host_shards[0].shape[1:]

    if data_sharding is not None:
      full_array = jax.make_array_from_single_device_arrays(
          expected_shape, data_sharding, per_device_arrays
      )
    else:
      # Fallback (mostly for single device)
      full_array = jnp.concatenate(per_device_arrays, axis=0)

    full_array.block_until_ready()

    # Verify result is contiguous and correct shape
    assert full_array.shape == expected_shape

  chk_h2d_end = time.perf_counter()
  h2d_ms = (chk_h2d_end - chk_h2d_start) * 1000

  for db in chunked_buffers:
    db.delete()
  full_array.delete()
  return h2d_ms


def _run_d2h_chunked(host_data, data_sharding, num_devices, chunks_per_device):
  data_on_device = jax.device_put(host_data, data_sharding)
  data_on_device.block_until_ready()

  # Pre-allocate host buffer for direct writing
  host_out = np.empty_like(host_data)

  def _fetch_and_store(shard_data, start, end, global_offset):
    chunk_data = jax.device_get(shard_data[start:end])
    host_out[global_offset + start : global_offset + end] = chunk_data

  total_workers = num_devices * chunks_per_device
  chk_d2h_start = time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
    d2h_futures = []
    
    # Sort shards to ensure correct offset calculation
    # Assuming standard mesh/sharding where index determines position
    sorted_shards = sorted(data_on_device.addressable_shards, key=lambda s: s.index)
    
    current_global_offset = 0
    for shard in sorted_shards:
      # Direct slicing on device array to avoid copy
      shard_len = shard.data.shape[0]
      chunk_size = (shard_len + chunks_per_device - 1) // chunks_per_device
      
      for i in range(chunks_per_device):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, shard_len)
        if start < end:
          d2h_futures.append(
              executor.submit(
                  _fetch_and_store, 
                  shard.data, 
                  start, 
                  end, 
                  current_global_offset
              )
          )
      current_global_offset += shard_len

    _ = [f.result() for f in d2h_futures]
  
  chk_d2h_end = time.perf_counter()
  d2h_ms = (chk_d2h_end - chk_d2h_start) * 1000
  
  # Verify result is contiguous and correct shape
  assert host_out.flags.c_contiguous
  assert host_out.shape == host_data.shape

  data_on_device.delete()
  del host_out
  return d2h_ms


def _run_chunked(host_data, data_sharding, host_shards, target_devices, num_devices, chunks_per_device_h2d, chunks_per_device_d2h):
  h2d_ms = _run_h2d_chunked(host_shards, target_devices, num_devices, chunks_per_device_h2d, data_sharding)
  d2h_ms = _run_d2h_chunked(host_data, data_sharding, num_devices, chunks_per_device_d2h)
  return h2d_ms, d2h_ms


def _run_warmup(host_data, data_sharding, data_size_mib):
  # --- ADAPTIVE WARM UP ---
  if data_size_mib <= 128:
    warmup_iters = 50
  elif data_size_mib >= 8192:
    warmup_iters = 3
  else:
    warmup_iters = 10

  for _ in range(warmup_iters):
    data_on_device = jax.device_put(host_data, data_sharding)
    data_on_device.block_until_ready()
    _ = jax.device_get(data_on_device)
    data_on_device.delete()

  gc.collect()


def _find_optimal_chunk_size(
    run_fn,
    num_devices,
    data_size_mib,
    prefix="",
    search_min_size_mib=1,
    max_global_threads=256
):
  """Finds optimal chunk size by iterating over candidates."""
  print(f"  [{prefix}] Searching for optimal chunk size...")

  # Generate size candidates
  candidates_mib = []
  curr = search_min_size_mib
  data_per_device_mib = data_size_mib / num_devices
  
  # Iterate until we cover the full data size per device
  while curr <= data_per_device_mib:
    candidates_mib.append(curr)
    curr *= 2
  # Ensure we test at least one candidate (e.g. if data < min_size)
  if not candidates_mib:
    candidates_mib.append(data_per_device_mib)

  # Map sizes to counts, keeping track of unique counts to test
  candidates_counts = []
  seen_counts = set()
  
  for size_mib in candidates_mib:
    if size_mib > data_per_device_mib:
      count = 1
    else:
      count = int(data_per_device_mib / size_mib)
      if count < 1: count = 1
       
    # Filter by max global threads
    if (count * num_devices) > max_global_threads:
      continue
    
    if count not in seen_counts:
      candidates_counts.append(count)
      seen_counts.add(count)
      
  # Sort candidates (counts) ascending for clean output
  candidates_counts.sort()
  
  if not candidates_counts:
    candidates_counts = [1]

  best_chunk_count = 1
  best_median_bw = -1.0
  
  # 5 search iterations + 3 warmup (before search)
  warmup_iters = 3
  search_iters = 5
  
  try:
    for _ in range(warmup_iters):
      run_fn(1) # Warmup with 1 chunk
  except Exception:
    pass 
    
  for chunk_count in candidates_counts:
    times_ms = []
    try:
      for _ in range(search_iters):
        t_start = time.perf_counter()
        res = run_fn(chunk_count)
        t_end = time.perf_counter()
        
        if isinstance(res, (int, float)):
          times_ms.append(res)
        else:
          times_ms.append((t_end - t_start) * 1000)
      
      median_ms = np.median(times_ms)
      if median_ms > 0:
        if best_median_bw < 0 or median_ms < best_median_bw:
          best_median_bw = median_ms
          best_chunk_count = chunk_count
    except Exception as e:
      continue
      
  print(f"  [{prefix}] Found optimal chunk count: {best_chunk_count} (approx size: {data_per_device_mib/best_chunk_count:.2f} MiB)")
  return best_chunk_count

def benchmark_host_device_smart_chunking(
    data_size_mib: int,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    # --- SMART CHUNKING LOGIC ---
    num_devices = len(jax.devices())
    print(f"  [Smart Chunking Enabled] Using {num_devices} devices")

    num_elements = 1024 * 1024 * data_size_mib // np.dtype(np.float32).itemsize
    host_data = np.random.normal(size=(num_elements // 128, 128)).astype(
        np.float32
    )
    mesh = sharding.Mesh(
        np.array(jax.devices()).reshape((num_devices,)), axis_names=("x",)
    )
    partition_spec = sharding.PartitionSpec("x",)
    data_sharding = sharding.NamedSharding(mesh, partition_spec)

    # --- ADAPTIVE WARM UP ---
    _run_warmup(host_data, data_sharding, data_size_mib)

    # Pre-calculate sharding info
    dummy_put = jax.device_put(host_data[:num_devices], data_sharding)
    target_devices = [s.device for s in dummy_put.addressable_shards]
    dummy_put.delete()

    host_shards = np.split(host_data, num_devices, axis=0)

    # --- SMART CHUNKING CONFIG ---
    def _search_runner_h2d(chunk_count):
        return _run_h2d_chunked(
            host_shards, target_devices, num_devices, chunk_count, data_sharding
        )

    chunks_per_device_h2d = _find_optimal_chunk_size(
        _search_runner_h2d, num_devices, data_size_mib, prefix="H2D"
    )

    def _search_runner_d2h(chunk_count):
        return _run_d2h_chunked(
            host_data, data_sharding, num_devices, chunk_count
        )

    chunks_per_device_d2h = _find_optimal_chunk_size(
        _search_runner_d2h, num_devices, data_size_mib, prefix="D2H"
    )

    if trace_dir:
        trace_dir_smart = os.path.join(trace_dir, "smart_chunking")
        profiler_context = jax.profiler.trace(trace_dir_smart)
    else:
        profiler_context = contextlib.nullcontext()

    h2d_perf, d2h_perf = [], []
    with profiler_context:
        for i in range(num_runs):
            if trace_dir:
                step_context = jax.profiler.StepTraceAnnotation("host_device", step_num=i)
            else:
                step_context = contextlib.nullcontext()

            with step_context:
                h2d_ms, d2h_ms = _run_chunked(
                    host_data, data_sharding, host_shards, target_devices,
                    num_devices, chunks_per_device_h2d, chunks_per_device_d2h
                )
                h2d_perf.append(h2d_ms)
                d2h_perf.append(d2h_ms)

    # Clean up
    del host_shards
    gc.collect()

    results = {
        "H2D_Bandwidth_ms": h2d_perf,
        "D2H_Bandwidth_ms": d2h_perf,
        "Chunk_Count_H2D": chunks_per_device_h2d,
        "Chunk_Count_D2H": chunks_per_device_d2h,
    }
    return results

def _calculate_metrics_impl(
    data_size_mib: int,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
    **kwargs: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for Host-Device transfer."""
    params = locals().items()
    
    # Filter out list params from metadata to avoid explosion
    metadata_keys = {
        "data_size_mib", 
    }
    # Add any extra keys passed via kwargs to metadata
    metadata = {k: v for k, v in params if k in metadata_keys}
    metadata.update(kwargs)
    
    metrics = {}
    
    def add_metric(name, ms_list):
        # Report Bandwidth (GiB/s)
        # Handle division by zero if ms is 0
        bw_list = [
            ((data_size_mib / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in ms_list
        ]
        stats_bw = MetricsStatistics(bw_list, f"{name}_bw (GiB/s)")
        metrics.update(stats_bw.serialize_statistics())

    add_metric("H2D", H2D_Bandwidth_ms)
    add_metric("D2H", D2H_Bandwidth_ms)

    return metadata, metrics


def benchmark_host_device_smart_chunking_calculate_metrics(
    data_size_mib: int,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
    Chunk_Count_H2D: int,
    Chunk_Count_D2H: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _calculate_metrics_impl(
        data_size_mib, 
        H2D_Bandwidth_ms, 
        D2H_Bandwidth_ms, 
        Chunk_Count_H2D=Chunk_Count_H2D, 
        Chunk_Count_D2H=Chunk_Count_D2H
    )


def benchmark_host_device(
    data_size_mib: int,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks H2D/D2H transfer using simple device_put/device_get."""
    
    num_elements = 1024 * 1024 * data_size_mib // np.dtype(np.float32).itemsize
    
    # Allocate Host Source Buffer
    column = 128
    host_data = np.random.normal(size=(num_elements // column, column)).astype(np.float32)
    
    print(
        f"Benchmarking Transfer with Data Size: {data_size_mib} MB for {num_runs} iterations"
    )

    # Performance Lists
    h2d_perf, d2h_perf = [], []

    # Profiling Context
    if trace_dir:
        profiler_context = jax.profiler.trace(trace_dir)
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context:
        try:
             _run_warmup(host_data, None, data_size_mib)
        except Exception:
             pass

        for i in range(num_runs):
            # Step Context
            if trace_dir:
                step_context = jax.profiler.StepTraceAnnotation("host_device", step_num=i)
            else:
                step_context = contextlib.nullcontext()
            
            with step_context:
                 # H2D
                t0 = time.perf_counter()
                
                # Simple device_put
                device_array = jax.device_put(host_data)
                device_array.block_until_ready()
                
                t1 = time.perf_counter()
                h2d_perf.append((t1 - t0) * 1000)
                
                assert device_array.shape == host_data.shape
                
                # D2H
                t2 = time.perf_counter()
                
                # Simple device_get
                # Note: device_get returns a numpy array (copy)
                _ = jax.device_get(device_array)
                
                t3 = time.perf_counter()
                d2h_perf.append((t3 - t2) * 1000)
                
                device_array.delete()

    return {
        "H2D_Bandwidth_ms": h2d_perf,
        "D2H_Bandwidth_ms": d2h_perf,
    }

def benchmark_host_device_calculate_metrics(
    data_size_mib: int,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _calculate_metrics_impl(
        data_size_mib, 
        H2D_Bandwidth_ms, 
        D2H_Bandwidth_ms
    )
