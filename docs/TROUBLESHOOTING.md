# Neural Operator Cryptanalysis Lab - Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide addresses common issues encountered when using the Neural Operator Cryptanalysis Lab. The guide is organized by component and includes diagnostic steps, solutions, and prevention strategies.

**Responsible Use Notice**: This framework is for defensive security research only. Ensure proper authorization before testing on any systems.

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Installation Issues](#installation-issues)
3. [Neural Operator Training Issues](#neural-operator-training-issues)
4. [Hardware Integration Issues](#hardware-integration-issues)
5. [Performance Issues](#performance-issues)
6. [Memory and Resource Issues](#memory-and-resource-issues)
7. [Data Issues](#data-issues)
8. [API and Network Issues](#api-and-network-issues)
9. [Security and Authentication Issues](#security-and-authentication-issues)
10. [Deployment Issues](#deployment-issues)
11. [FAQ](#frequently-asked-questions)

---

## Quick Diagnostic Tools

### System Health Check

```python
from neural_cryptanalysis.utils.diagnostics import SystemDiagnostics

# Run comprehensive system check
diagnostics = SystemDiagnostics()
health_report = diagnostics.run_full_check()

print("=== System Health Report ===")
for component, status in health_report.items():
    icon = "✅" if status['healthy'] else "❌"
    print(f"{icon} {component}: {status['message']}")
    
    if not status['healthy'] and 'suggestions' in status:
        print(f"   Suggestions: {status['suggestions']}")
```

### Quick Environment Check

```bash
#!/bin/bash
# quick_check.sh - Run this script to diagnose common issues

echo "=== Neural SCA Environment Check ==="

# Python version
echo "Python version:"
python --version

# PyTorch installation
echo "PyTorch status:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Neural Cryptanalysis installation
echo "Neural Cryptanalysis status:"
python -c "import neural_cryptanalysis; print(f'Version: {neural_cryptanalysis.__version__}')"

# GPU status
echo "GPU information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPUs found"

# Hardware devices
echo "USB devices:"
lsusb | grep -E "(Pico|ChipWhisperer|Oscilloscope)" || echo "No supported hardware devices found"

# Disk space
echo "Disk space:"
df -h / | tail -1

# Memory usage
echo "Memory usage:"
free -h
```

### Debug Mode Activation

```python
# Enable debug mode for detailed logging
import logging
from neural_cryptanalysis.utils.logging_utils import configure_debug_logging

configure_debug_logging(
    level=logging.DEBUG,
    include_traceback=True,
    log_to_file=True,
    log_file='debug.log'
)

# Enable PyTorch debugging
import torch
torch.autograd.set_detect_anomaly(True)
```

---

## Installation Issues

### Issue: Package Installation Fails

**Symptoms**:
- `pip install neural-operator-cryptanalysis` fails
- Missing dependencies errors
- Compilation errors during installation

**Diagnosis**:
```bash
# Check pip version
pip --version

# Check available space
df -h

# Check internet connectivity
ping pypi.org

# Check for conflicting packages
pip list | grep -E "(torch|numpy|scipy)"
```

**Solutions**:

1. **Update pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Clean installation**:
   ```bash
   pip cache purge
   pip install --no-cache-dir neural-operator-cryptanalysis
   ```

3. **Install with specific PyTorch version**:
   ```bash
   pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   pip install neural-operator-cryptanalysis
   ```

4. **Use conda for complex dependencies**:
   ```bash
   conda create -n neural_sca python=3.9
   conda activate neural_sca
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   pip install neural-operator-cryptanalysis
   ```

### Issue: CUDA Installation Problems

**Symptoms**:
- `torch.cuda.is_available()` returns `False`
- CUDA version mismatches
- GPU not detected

**Diagnosis**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA compatibility
python -c "import torch; print(torch.version.cuda)"
```

**Solutions**:

1. **Install NVIDIA drivers**:
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-515
   sudo reboot
   ```

2. **Install CUDA toolkit**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

3. **Install compatible PyTorch**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Hardware Dependencies Missing

**Symptoms**:
- Cannot import hardware modules
- USB device access denied
- Driver not found errors

**Solutions**:

1. **Install hardware dependencies**:
   ```bash
   # For Picoscope
   sudo apt-get install libps6000
   
   # For ChipWhisperer
   pip install chipwhisperer
   
   # For general USB access
   sudo apt-get install libusb-1.0-0-dev libudev-dev
   ```

2. **Set up device permissions**:
   ```bash
   # Add user to dialout group
   sudo usermod -a -G dialout $USER
   
   # Create udev rules for ChipWhisperer
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="ace2", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-chipwhisperer.rules
   
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

---

## Neural Operator Training Issues

### Issue: Training Fails to Converge

**Symptoms**:
- Loss plateaus at high values
- Accuracy remains low
- Training oscillates without improvement

**Diagnosis**:
```python
from neural_cryptanalysis.utils.training_diagnostics import TrainingDiagnostics

diagnostics = TrainingDiagnostics()

# Analyze training data
data_analysis = diagnostics.analyze_training_data(traces, labels)
print(f"Data quality score: {data_analysis.quality_score}")
print(f"Label distribution: {data_analysis.label_distribution}")

# Check model initialization
model_analysis = diagnostics.analyze_model_initialization(model)
print(f"Weight initialization: {model_analysis.weight_stats}")
print(f"Gradient flow: {model_analysis.gradient_flow}")
```

**Solutions**:

1. **Improve data quality**:
   ```python
   # Better preprocessing
   from neural_cryptanalysis.preprocessing import AdvancedPreprocessor
   
   preprocessor = AdvancedPreprocessor()
   processed_traces = preprocessor.process(
       traces,
       steps=['alignment', 'denoising', 'normalization', 'poi_selection'],
       alignment_method='cross_correlation',
       poi_count=200
   )
   ```

2. **Adjust learning rate schedule**:
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
   
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
   scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
   ```

3. **Use better model initialization**:
   ```python
   def init_weights(module):
       if isinstance(module, torch.nn.Linear):
           torch.nn.init.xavier_normal_(module.weight)
           torch.nn.init.zeros_(module.bias)
       elif isinstance(module, torch.nn.Conv1d):
           torch.nn.init.kaiming_normal_(module.weight, mode='fan_out')
   
   model.apply(init_weights)
   ```

### Issue: Overfitting

**Symptoms**:
- Training accuracy high, validation accuracy low
- Large gap between training and validation loss
- Model performs poorly on new data

**Solutions**:

1. **Add regularization**:
   ```python
   model = FourierNeuralOperator(
       modes=16,
       width=64,
       dropout=0.3,  # Increased dropout
       weight_decay=1e-3  # L2 regularization
   )
   ```

2. **Data augmentation**:
   ```python
   from neural_cryptanalysis.augmentation import TraceAugmentation
   
   augmentation = TraceAugmentation([
       'temporal_jitter',
       'noise_addition',
       'amplitude_scaling',
       'mixup'
   ])
   
   augmented_traces = augmentation.apply(traces, augmentation_factor=2)
   ```

3. **Early stopping**:
   ```python
   from neural_cryptanalysis.utils.early_stopping import EarlyStopping
   
   early_stopping = EarlyStopping(
       patience=10,
       min_delta=1e-4,
       restore_best_weights=True
   )
   ```

### Issue: GPU Memory Overflow

**Symptoms**:
- "CUDA out of memory" errors
- Training crashes during forward/backward pass
- Cannot load model or data

**Solutions**:

1. **Reduce batch size**:
   ```python
   # Use gradient accumulation for effective large batch
   effective_batch_size = 64
   mini_batch_size = 16
   accumulation_steps = effective_batch_size // mini_batch_size
   
   optimizer.zero_grad()
   for i, (batch_traces, batch_labels) in enumerate(dataloader):
       outputs = model(batch_traces)
       loss = criterion(outputs, batch_labels) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Use gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   class MemoryEfficientFNO(FourierNeuralOperator):
       def forward(self, x):
           for layer in self.layers:
               x = checkpoint(layer, x, use_reentrant=False)
           return x
   ```

3. **Mixed precision training**:
   ```python
   from torch.cuda.amp import GradScaler, autocast
   
   scaler = GradScaler()
   
   for batch_traces, batch_labels in dataloader:
       optimizer.zero_grad()
       
       with autocast():
           outputs = model(batch_traces)
           loss = criterion(outputs, batch_labels)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

---

## Hardware Integration Issues

### Issue: Cannot Connect to Oscilloscope

**Symptoms**:
- Device not found errors
- USB connection timeouts
- Oscilloscope not responding

**Diagnosis**:
```bash
# Check USB connection
lsusb | grep -i pico

# Check device permissions
ls -la /dev/ttyUSB*

# Test basic connection
python -c "from neural_cryptanalysis.hardware import list_oscilloscopes; print(list_oscilloscopes())"
```

**Solutions**:

1. **Check physical connections**:
   - Ensure USB cable is properly connected
   - Try different USB port
   - Check cable integrity
   - Verify oscilloscope is powered on

2. **Fix permissions**:
   ```bash
   sudo usermod -a -G dialout $USER
   newgrp dialout
   ```

3. **Install drivers**:
   ```bash
   # For Picoscope
   wget https://www.picotech.com/download/software/libps6000-2.1.117-4r1693.x86_64.rpm
   sudo rpm -i libps6000-2.1.117-4r1693.x86_64.rpm
   ```

4. **Reset USB device**:
   ```bash
   # Find device
   lsusb
   
   # Reset specific device
   sudo usb_reset /dev/bus/usb/001/002
   ```

### Issue: Target Board Communication Problems

**Symptoms**:
- Cannot program target board
- Firmware upload fails
- Board not responding to commands

**Solutions**:

1. **Check programming setup**:
   ```python
   from neural_cryptanalysis.hardware import ChipWhispererInterface
   
   # Enable debug mode
   cw = ChipWhispererInterface(debug=True)
   
   # Check connection
   if not cw.connect():
       print("Connection failed - check cables and power")
   
   # Verify target detection
   target_info = cw.get_target_info()
   print(f"Target detected: {target_info}")
   ```

2. **Programming troubleshooting**:
   ```bash
   # Check if target is in bootloader mode
   dmesg | tail
   
   # Manual programming with OpenOCD
   openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program firmware.hex verify reset exit"
   ```

3. **Power and clock issues**:
   ```python
   # Check power supply
   target_voltage = cw.get_target_voltage()
   if target_voltage < 3.0:
       print("Low target voltage - check power supply")
   
   # Check clock signal
   clock_freq = cw.get_clock_frequency()
   print(f"Clock frequency: {clock_freq} Hz")
   ```

### Issue: Synchronization Problems

**Symptoms**:
- Traces not aligned properly
- Trigger not working
- Multiple devices out of sync

**Solutions**:

1. **Improve trigger setup**:
   ```python
   from neural_cryptanalysis.hardware import SynchronizationManager
   
   sync_manager = SynchronizationManager()
   
   # Configure trigger
   trigger_config = {
       'channel': 'external',
       'threshold': 2.5,  # V
       'edge': 'rising',
       'delay': 0,  # samples
       'timeout': 1000  # ms
   }
   
   sync_manager.configure_trigger(trigger_config)
   ```

2. **Hardware clock synchronization**:
   ```python
   # Use external clock source
   scope.set_clock_source('external')
   target.set_clock_source('scope')
   
   # Verify synchronization
   sync_quality = sync_manager.test_synchronization()
   print(f"Sync quality: {sync_quality}")
   ```

3. **Software alignment**:
   ```python
   from neural_cryptanalysis.preprocessing import TraceAlignment
   
   aligner = TraceAlignment(method='cross_correlation')
   aligned_traces = aligner.align(traces, reference_trace)
   ```

---

## Performance Issues

### Issue: Slow Training Performance

**Symptoms**:
- Training takes much longer than expected
- Low GPU utilization
- High CPU usage during training

**Diagnosis**:
```python
from neural_cryptanalysis.utils.performance import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile('training'):
    for epoch in range(10):
        for batch in dataloader:
            # Training code
            pass

# Analyze bottlenecks
bottlenecks = profiler.identify_bottlenecks()
print("Performance bottlenecks:")
for component, time_pct in bottlenecks.items():
    print(f"  {component}: {time_pct:.1f}%")
```

**Solutions**:

1. **Optimize data loading**:
   ```python
   from torch.utils.data import DataLoader
   
   # Optimized DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=64,
       num_workers=8,  # Increase workers
       pin_memory=True,  # Speed up GPU transfer
       persistent_workers=True,  # Keep workers alive
       prefetch_factor=4  # Prefetch more batches
   )
   ```

2. **Model compilation** (PyTorch 2.0+):
   ```python
   import torch
   
   # Compile model for optimization
   if hasattr(torch, 'compile'):
       model = torch.compile(model, mode='max-autotune')
   ```

3. **Optimize neural operator architecture**:
   ```python
   # Use efficient attention mechanisms
   model = FourierNeuralOperator(
       modes=16,  # Reduce for speed
       width=64,  # Smaller hidden dimension
       efficient_attention=True,
       flash_attention=True  # If available
   )
   ```

### Issue: High Memory Usage

**Symptoms**:
- System runs out of RAM
- Slow performance due to swapping
- Cannot process large datasets

**Solutions**:

1. **Streaming data processing**:
   ```python
   from neural_cryptanalysis.data import StreamingDataset
   
   # Process data in chunks
   streaming_dataset = StreamingDataset(
       data_path='large_dataset.h5',
       chunk_size=1000,
       preload_chunks=2
   )
   ```

2. **Memory-mapped files**:
   ```python
   import numpy as np
   
   # Use memory-mapped arrays for large datasets
   traces = np.memmap(
       'traces.dat',
       dtype=np.float32,
       mode='r',
       shape=(1000000, 5000)
   )
   ```

3. **Optimize preprocessing**:
   ```python
   # In-place operations to save memory
   from neural_cryptanalysis.preprocessing import InPlacePreprocessor
   
   preprocessor = InPlacePreprocessor()
   preprocessor.standardize_inplace(traces)
   preprocessor.filter_inplace(traces, cutoff=100000)
   ```

### Issue: Slow Inference

**Symptoms**:
- Predictions take too long
- Real-time analysis not possible
- High latency for single traces

**Solutions**:

1. **Model optimization**:
   ```python
   # Use TensorRT for optimization (NVIDIA GPUs)
   import torch_tensorrt
   
   optimized_model = torch_tensorrt.compile(
       model,
       inputs=[torch.randn(1, 5000).cuda()],
       enabled_precisions={torch.float, torch.half}
   )
   ```

2. **Batch inference**:
   ```python
   # Process multiple traces together
   def batch_inference(model, traces, batch_size=32):
       results = []
       for i in range(0, len(traces), batch_size):
           batch = traces[i:i+batch_size]
           with torch.no_grad():
               batch_results = model(batch)
           results.extend(batch_results)
       return results
   ```

3. **Model quantization**:
   ```python
   # Quantize model for faster inference
   quantized_model = torch.quantization.quantize_dynamic(
       model,
       {torch.nn.Linear, torch.nn.Conv1d},
       dtype=torch.qint8
   )
   ```

---

## Memory and Resource Issues

### Issue: System Runs Out of Disk Space

**Symptoms**:
- "No space left on device" errors
- Cannot save models or traces
- System becomes unresponsive

**Diagnosis and Solutions**:

1. **Check disk usage**:
   ```bash
   # Check overall disk usage
   df -h
   
   # Find large files
   du -h /data --max-depth=2 | sort -hr | head -20
   
   # Find large log files
   find /var/log -type f -size +100M -exec ls -lh {} +
   ```

2. **Clean up space**:
   ```bash
   # Clean Docker images
   docker system prune -af
   
   # Clean old model checkpoints
   find /data/models -name "*.pth" -mtime +7 -not -name "*best*" -delete
   
   # Compress old traces
   find /data/traces -name "*.npz" -mtime +30 -exec gzip {} \;
   
   # Clean pip cache
   pip cache purge
   
   # Clean conda cache
   conda clean --all
   ```

3. **Implement automated cleanup**:
   ```python
   # automated_cleanup.py
   import os
   import time
   from pathlib import Path
   
   class StorageManager:
       def __init__(self, data_dir, max_usage_percent=85):
           self.data_dir = Path(data_dir)
           self.max_usage_percent = max_usage_percent
       
       def check_disk_usage(self):
           usage = os.statvfs(self.data_dir)
           used_percent = (1 - usage.f_bavail / usage.f_blocks) * 100
           return used_percent
       
       def cleanup_old_files(self):
           current_usage = self.check_disk_usage()
           
           if current_usage > self.max_usage_percent:
               # Remove old checkpoints first
               self.cleanup_checkpoints()
               
               # Then compress old traces
               self.compress_old_traces()
               
               # Finally remove very old files
               self.remove_old_files()
   ```

### Issue: CPU/GPU Resource Contention

**Symptoms**:
- Multiple processes competing for resources
- System becomes unresponsive
- Training crashes due to resource limits

**Solutions**:

1. **Resource monitoring and limiting**:
   ```python
   import psutil
   import torch
   
   # Monitor system resources
   def monitor_resources():
       cpu_percent = psutil.cpu_percent(interval=1)
       memory = psutil.virtual_memory()
       
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.memory_reserved() / torch.cuda.max_memory_reserved()
           print(f"GPU Memory: {gpu_memory:.1%}")
       
       print(f"CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%")
   
   # Set resource limits
   torch.set_num_threads(4)  # Limit CPU threads
   torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory
   ```

2. **Process scheduling**:
   ```bash
   # Run with lower priority
   nice -n 10 python train_model.py
   
   # Use cgroups for resource limiting
   cgcreate -g memory,cpu:neural_sca
   echo "8G" > /sys/fs/cgroup/memory/neural_sca/memory.limit_in_bytes
   cgexec -g memory,cpu:neural_sca python train_model.py
   ```

---

## Data Issues

### Issue: Corrupted or Invalid Data

**Symptoms**:
- Training fails with data errors
- Unexpected NaN values
- Shape mismatches

**Diagnosis**:
```python
from neural_cryptanalysis.utils.data_validation import DataValidator

validator = DataValidator()

# Check data integrity
validation_report = validator.validate_traces(traces, labels)

print("Data validation report:")
print(f"  Shape consistency: {validation_report.shape_consistent}")
print(f"  NaN values: {validation_report.nan_count}")
print(f"  Infinite values: {validation_report.inf_count}")
print(f"  Label distribution: {validation_report.label_distribution}")
print(f"  Data type: {validation_report.dtype}")
```

**Solutions**:

1. **Data cleaning**:
   ```python
   import numpy as np
   
   # Remove NaN and infinite values
   def clean_traces(traces, labels):
       # Check for NaN or infinite values
       valid_mask = np.isfinite(traces).all(axis=1)
       
       if not valid_mask.all():
           print(f"Removing {(~valid_mask).sum()} corrupted traces")
           traces = traces[valid_mask]
           labels = labels[valid_mask]
       
       return traces, labels
   
   traces, labels = clean_traces(traces, labels)
   ```

2. **Data validation during loading**:
   ```python
   class ValidatingDataLoader:
       def __init__(self, dataset, **kwargs):
           self.dataset = dataset
           self.dataloader = DataLoader(dataset, **kwargs)
       
       def __iter__(self):
           for batch_traces, batch_labels in self.dataloader:
               # Validate each batch
               if torch.isnan(batch_traces).any():
                   print("Warning: NaN values in batch, skipping...")
                   continue
               
               if batch_traces.shape[0] != batch_labels.shape[0]:
                   print("Warning: Shape mismatch in batch, skipping...")
                   continue
               
               yield batch_traces, batch_labels
   ```

### Issue: Poor Data Quality

**Symptoms**:
- Low signal-to-noise ratio
- Poor alignment between traces
- Inconsistent labeling

**Solutions**:

1. **Improve data collection**:
   ```python
   # Better measurement parameters
   measurement_config = {
       'sampling_rate': 5e9,  # Higher sampling rate
       'trigger_delay': 100,  # Stable trigger
       'averaging': 4,       # Reduce noise
       'bandwidth_limit': 1e6  # Filter high-frequency noise
   }
   ```

2. **Advanced preprocessing**:
   ```python
   from neural_cryptanalysis.preprocessing import AdvancedPreprocessor
   
   preprocessor = AdvancedPreprocessor()
   
   # Multi-stage preprocessing
   processed_traces = preprocessor.process_pipeline(
       traces,
       steps=[
           ('resample', {'target_rate': 1e6}),
           ('denoise', {'method': 'wavelet', 'threshold': 0.1}),
           ('align', {'method': 'dtw', 'reference': 'mean'}),
           ('normalize', {'method': 'robust_zscore'}),
           ('segment', {'window_size': 2000, 'stride': 1000})
       ]
   )
   ```

3. **Quality assessment**:
   ```python
   from neural_cryptanalysis.analysis import QualityAssessment
   
   qa = QualityAssessment()
   
   # Assess trace quality
   quality_metrics = qa.assess_traces(traces, labels)
   
   print(f"SNR: {quality_metrics.snr:.2f} dB")
   print(f"Alignment quality: {quality_metrics.alignment_score:.3f}")
   print(f"Noise level: {quality_metrics.noise_level:.3f}")
   
   # Filter low-quality traces
   good_traces = qa.filter_by_quality(traces, min_snr=5.0)
   ```

---

## API and Network Issues

### Issue: API Connection Problems

**Symptoms**:
- Cannot connect to API server
- Timeouts during requests
- Authentication failures

**Diagnosis**:
```bash
# Test API connectivity
curl -v http://localhost:8000/health

# Check if port is open
netstat -tlnp | grep 8000

# Test DNS resolution
nslookup neural-sca.example.com

# Check firewall rules
sudo iptables -L
```

**Solutions**:

1. **Network configuration**:
   ```bash
   # Check server status
   systemctl status neural-cryptanalysis
   
   # Check logs
   journalctl -u neural-cryptanalysis -f
   
   # Restart service
   sudo systemctl restart neural-cryptanalysis
   ```

2. **Authentication troubleshooting**:
   ```python
   import requests
   
   # Test authentication
   auth_response = requests.post(
       'http://localhost:8000/auth/login',
       json={'username': 'test', 'password': 'test'},
       timeout=10
   )
   
   if auth_response.status_code == 200:
       token = auth_response.json()['token']
       print(f"Authentication successful: {token[:20]}...")
   else:
       print(f"Authentication failed: {auth_response.status_code}")
   ```

### Issue: Slow API Responses

**Symptoms**:
- Long response times
- Timeouts on large requests
- Poor user experience

**Solutions**:

1. **Implement caching**:
   ```python
   from functools import lru_cache
   import redis
   
   # Redis caching
   redis_client = redis.Redis(host='localhost', port=6379, db=0)
   
   def cache_result(key, value, expiration=3600):
       redis_client.setex(key, expiration, value)
   
   def get_cached_result(key):
       return redis_client.get(key)
   ```

2. **Async processing**:
   ```python
   from fastapi import FastAPI, BackgroundTasks
   import asyncio
   
   app = FastAPI()
   
   @app.post("/analyze/async")
   async def async_analysis(
       traces: List[float],
       background_tasks: BackgroundTasks
   ):
       # Start background task
       task_id = str(uuid.uuid4())
       background_tasks.add_task(process_traces, task_id, traces)
       
       return {"task_id": task_id, "status": "processing"}
   
   @app.get("/analyze/status/{task_id}")
   async def get_analysis_status(task_id: str):
       # Check task status
       result = get_task_result(task_id)
       return {"task_id": task_id, "result": result}
   ```

---

## Security and Authentication Issues

### Issue: Authentication Failures

**Symptoms**:
- Cannot log in to system
- Token validation errors
- Access denied messages

**Solutions**:

1. **Check authentication configuration**:
   ```yaml
   # config/security.yaml
   authentication:
     enabled: true
     method: jwt
     secret_key: your-secret-key
     token_expiry: 3600
     
   authorization:
     enabled: true
     default_role: user
   ```

2. **Debug authentication**:
   ```python
   from neural_cryptanalysis.security import AuthenticationManager
   
   auth_manager = AuthenticationManager()
   
   # Test token validation
   token = "your-jwt-token"
   is_valid = auth_manager.validate_token(token)
   print(f"Token valid: {is_valid}")
   
   if not is_valid:
       # Check token expiration
       payload = auth_manager.decode_token(token, verify=False)
       print(f"Token payload: {payload}")
   ```

### Issue: Permission Denied Errors

**Symptoms**:
- Cannot access certain endpoints
- File permission errors
- Docker container access issues

**Solutions**:

1. **Fix file permissions**:
   ```bash
   # Fix ownership
   sudo chown -R neural_sca:neural_sca /app/data
   
   # Fix permissions
   chmod -R 755 /app/data
   chmod -R 644 /app/config/*.yaml
   ```

2. **Docker permission fixes**:
   ```dockerfile
   # Create user with matching UID/GID
   RUN groupadd -r neural_sca -g 1000 && \
       useradd -r -u 1000 -g neural_sca neural_sca
   
   # Set proper ownership
   COPY --chown=neural_sca:neural_sca . /app
   ```

---

## Deployment Issues

### Issue: Kubernetes Pod Failures

**Symptoms**:
- Pods crash or fail to start
- ImagePullBackOff errors
- Resource limit issues

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n neural-cryptanalysis

# Describe failing pod
kubectl describe pod <pod-name> -n neural-cryptanalysis

# Check logs
kubectl logs <pod-name> -n neural-cryptanalysis --previous

# Check events
kubectl get events -n neural-cryptanalysis --sort-by='.firstTimestamp'
```

**Solutions**:

1. **Fix image issues**:
   ```bash
   # Check if image exists
   docker pull neural-cryptanalysis:latest
   
   # Push to correct registry
   docker tag neural-cryptanalysis:latest your-registry/neural-cryptanalysis:latest
   docker push your-registry/neural-cryptanalysis:latest
   
   # Update deployment
   kubectl set image deployment/neural-cryptanalysis neural-sca=your-registry/neural-cryptanalysis:latest -n neural-cryptanalysis
   ```

2. **Adjust resource limits**:
   ```yaml
   resources:
     requests:
       memory: "4Gi"
       cpu: "2"
     limits:
       memory: "8Gi"
       cpu: "4"
   ```

### Issue: Docker Container Problems

**Symptoms**:
- Container exits immediately
- Cannot bind to ports
- Volume mount issues

**Solutions**:

1. **Debug container startup**:
   ```bash
   # Run interactively
   docker run -it --entrypoint /bin/bash neural-cryptanalysis:latest
   
   # Check container logs
   docker logs neural-sca --follow
   
   # Inspect container
   docker inspect neural-sca
   ```

2. **Fix port binding issues**:
   ```bash
   # Check what's using the port
   sudo netstat -tlnp | grep 8000
   
   # Use different port
   docker run -p 8001:8000 neural-cryptanalysis:latest
   ```

---

## Frequently Asked Questions

### General Questions

**Q: What Python versions are supported?**
A: Python 3.9, 3.10, and 3.11 are officially supported. Python 3.12 support is experimental.

**Q: Can I run this without a GPU?**
A: Yes, but performance will be significantly slower. CPU-only mode is suitable for small datasets and testing.

**Q: Is this compatible with Apple Silicon (M1/M2) Macs?**
A: Yes, but with limitations. Metal Performance Shaders (MPS) backend is supported for PyTorch operations, but some hardware integration features may not work.

### Neural Operator Questions

**Q: Which neural operator architecture should I use?**
A: 
- **FNO**: Best for most side-channel analysis tasks
- **DeepONet**: Good for operator learning between different domains
- **GraphNO**: Best for multi-modal sensor fusion
- **PhysicsNO**: When you have physical constraints

**Q: My model isn't learning anything. What should I check?**
A:
1. Data quality (SNR, alignment)
2. Label correctness
3. Learning rate (try 1e-4 to 1e-2)
4. Model architecture size
5. Preprocessing steps

**Q: How much training data do I need?**
A: Typical requirements:
- Simple attacks: 1K-10K traces
- Complex implementations: 50K-500K traces
- Masked implementations: 100K-1M+ traces

### Hardware Questions

**Q: What oscilloscopes are supported?**
A: Officially supported:
- Picoscope 6000 series
- Keysight InfiniiVision series
- ChipWhisperer scopes
- Generic SCPI-compatible instruments

**Q: Can I use my own target board?**
A: Yes, you can implement custom target interfaces by extending the `TargetBoard` base class.

**Q: Do I need expensive equipment?**
A: No, you can start with:
- ChipWhisperer Lite ($300)
- Arduino/STM32 development boards ($20-50)
- Basic USB oscilloscope ($200-500)

### Performance Questions

**Q: Training is very slow. How can I speed it up?**
A:
1. Use GPU acceleration
2. Increase batch size
3. Enable mixed precision training
4. Use model compilation (PyTorch 2.0+)
5. Optimize data loading pipeline

**Q: How can I reduce memory usage?**
A:
1. Reduce batch size
2. Use gradient checkpointing
3. Enable memory-efficient attention
4. Process data in chunks
5. Use data streaming

### Security Questions

**Q: Is this tool safe to use?**
A: Yes, when used responsibly. The framework is designed for defensive security research only. Always:
- Obtain proper authorization
- Follow responsible disclosure practices
- Use only on your own systems or with permission

**Q: How is data protected?**
A: The framework includes:
- Encryption at rest and in transit
- Access controls and authentication
- Audit logging
- Data anonymization features

### Deployment Questions

**Q: Can I deploy this in the cloud?**
A: Yes, the framework supports:
- AWS, Azure, GCP deployments
- Kubernetes orchestration
- Docker containerization
- Auto-scaling capabilities

**Q: What about compliance requirements?**
A: Built-in support for:
- GDPR compliance
- SOC 2 frameworks
- NIST cybersecurity standards
- Academic research ethics

### Research Questions

**Q: How do I cite this work?**
A: See the README.md file for current citation information and published papers.

**Q: Can I contribute to the project?**
A: Yes! Please see CONTRIBUTING.md for guidelines on:
- Code contributions
- Bug reports
- Feature requests
- Documentation improvements

**Q: Is commercial use allowed?**
A: Check the LICENSE file for current licensing terms. Generally, defensive security research is encouraged.

---

## Getting Help

### Community Support

- **GitHub Discussions**: https://github.com/neural-cryptanalysis/discussions
- **Documentation**: https://neural-cryptanalysis.readthedocs.io
- **Issue Tracker**: https://github.com/neural-cryptanalysis/issues

### Professional Support

For professional support, training, or consulting:
- Email: support@neural-cryptanalysis.org
- Enterprise support plans available

### Emergency Issues

For security vulnerabilities or urgent issues:
- Security email: security@neural-cryptanalysis.org
- Follow responsible disclosure guidelines

---

## Diagnostic Checklist

Before reporting an issue, please run through this checklist:

- [ ] Ran system health check
- [ ] Checked logs for error messages
- [ ] Verified all dependencies are installed
- [ ] Tested with minimal example
- [ ] Checked hardware connections (if applicable)
- [ ] Reviewed configuration files
- [ ] Searched existing issues and documentation
- [ ] Prepared minimal reproducible example

This troubleshooting guide should help resolve most common issues. For problems not covered here, please consult the community resources or contact support.