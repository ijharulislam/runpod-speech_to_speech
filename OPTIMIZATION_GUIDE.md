# PlayDiffusion RVC Performance Optimization Guide

## Problem Analysis: High CPU Usage (100%) vs Low GPU Usage (15%)

### Root Causes Identified:

1. **Synchronous I/O Operations**
   - Sequential HTTP downloads of audio files
   - Blocking file I/O operations
   - Synchronous S3 uploads

2. **CPU-Intensive Audio Processing**
   - Audio file decoding with `soundfile` library
   - Audio resampling operations
   - WAV file encoding

3. **Model Loading Overhead**
   - Creating new PlayDiffusion instances for each request
   - No GPU memory optimization

4. **Inefficient Threading**
   - Single-threaded processing pipeline
   - No parallelization of I/O operations

## Optimizations Implemented:

### 1. **Async I/O for File Downloads**
```python
# Before: Sequential downloads
response1 = requests.get(source_audio_url)
response2 = requests.get(target_audio_url)

# After: Parallel async downloads
source_content, target_content = download_files_parallel(source_audio_url, target_audio_url)
```

### 2. **Singleton Model Pattern**
```python
# Before: New model per request
rvc_model = PlayDiffusion()

# After: Reuse model instance
rvc_model = get_rvc_model()  # Singleton pattern
```

### 3. **Thread Pool for CPU-Intensive Tasks**
```python
# Move audio processing to background threads
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future = executor.submit(rvc_model.rvc, rvc_input)
    result = future.result()
```

### 4. **GPU Memory Optimization**
```python
# Configure GPU settings
torch.cuda.set_per_process_memory_fraction(0.8)
torch.set_default_dtype(torch.float16)
```

### 5. **Performance Monitoring**
```python
# Monitor CPU/GPU usage
from monitor_performance import start_monitoring, stop_monitoring
start_monitoring()
# ... perform voice conversion ...
stop_monitoring()
```

## Configuration Files:

### `gpu_optimization_config.py`
- GPU memory management settings
- Threading configuration
- Audio processing parameters

### `monitor_performance.py`
- Real-time CPU/GPU monitoring
- Performance metrics logging
- Usage analysis tools

## Expected Performance Improvements:

1. **CPU Usage**: Reduce from 100% to 30-50%
2. **GPU Usage**: Increase from 15% to 60-80%
3. **Response Time**: 20-40% faster processing
4. **Throughput**: Better concurrent request handling

## Usage Instructions:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure GPU Settings
```python
from gpu_optimization_config import configure_gpu
configure_gpu()
```

### 3. Monitor Performance
```python
from monitor_performance import start_monitoring, stop_monitoring

# Start monitoring before processing
start_monitoring()

# Perform voice conversion
result = voice_conversion(source_path, target_path)

# Stop monitoring and get summary
stop_monitoring()
summary = get_performance_summary()
print(summary)
```

## Additional Recommendations:

### 1. **Batch Processing**
- Process multiple audio files in batches
- Use GPU batching for better utilization

### 2. **Model Quantization**
- Consider using INT8 quantization for faster inference
- Implement model pruning for reduced memory usage

### 3. **Caching Strategy**
- Cache frequently used voice embeddings
- Implement audio file caching

### 4. **Load Balancing**
- Distribute requests across multiple GPU instances
- Use round-robin scheduling for better resource utilization

### 5. **Memory Management**
- Implement automatic GPU memory cleanup
- Monitor memory fragmentation

## Troubleshooting:

### High CPU Usage Persists:
1. Check if async I/O is working properly
2. Verify thread pool configuration
3. Monitor for blocking operations

### Low GPU Usage:
1. Ensure model is loaded on GPU
2. Check batch size configuration
3. Verify GPU memory allocation

### Memory Issues:
1. Reduce `GPU_MEMORY_FRACTION` in config
2. Implement memory cleanup
3. Monitor for memory leaks

## Performance Metrics:

Use the monitoring script to track:
- CPU utilization percentage
- GPU memory usage
- GPU compute utilization
- Response time
- Throughput (requests/second)

## Environment Variables:

```bash
# GPU optimization
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
``` 