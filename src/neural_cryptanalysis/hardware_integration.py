"""
Real-Time Hardware-in-the-Loop Integration System.

This module provides comprehensive integration with oscilloscopes, signal generators,
target boards, and automated measurement orchestration for live side-channel analysis.
"""

import asyncio
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from .core import TraceData, NeuralSCA
from .multi_modal_fusion import MultiModalData, SensorConfig
from .utils.logging_utils import setup_logger
from .utils.performance import PerformanceMonitor

logger = setup_logger(__name__)

@dataclass
class HardwareConfig:
    """Configuration for hardware devices."""
    device_type: str  # 'oscilloscope', 'signal_generator', 'target_board', 'probe'
    model: str
    connection_type: str  # 'usb', 'ethernet', 'serial', 'pcie'
    connection_params: Dict[str, Any]
    capabilities: Dict[str, Any]
    calibration_data: Optional[Dict[str, float]] = None
    last_calibration: Optional[datetime] = None

@dataclass
class MeasurementConfig:
    """Configuration for measurement acquisition."""
    channels: List[str]
    sample_rate: float
    memory_depth: int
    trigger_config: Dict[str, Any]
    preprocessing: List[str] = field(default_factory=list)
    real_time_processing: bool = True
    buffer_size: int = 10000
    timeout_ms: int = 5000

class HardwareDevice(ABC):
    """Abstract base class for hardware devices."""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.is_connected = False
        self.last_error = None
        self.performance_monitor = PerformanceMonitor()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to hardware device."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from hardware device."""
        pass
    
    @abstractmethod
    async def configure(self, settings: Dict[str, Any]) -> bool:
        """Configure device settings."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current device status."""
        pass
    
    @abstractmethod
    async def calibrate(self) -> bool:
        """Perform device calibration."""
        pass

class OscilloscopeDevice(HardwareDevice):
    """Oscilloscope hardware interface."""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self.acquisition_running = False
        self.trace_buffer = queue.Queue(maxsize=1000)
        
    async def connect(self) -> bool:
        """Connect to oscilloscope."""
        try:
            logger.info(f"Connecting to oscilloscope {self.config.model}")
            
            # Simulate connection based on type
            if self.config.connection_type == 'usb':
                await self._connect_usb()
            elif self.config.connection_type == 'ethernet':
                await self._connect_ethernet()
            else:
                raise ValueError(f"Unsupported connection type: {self.config.connection_type}")
            
            self.is_connected = True
            logger.info("Oscilloscope connected successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to connect to oscilloscope: {e}")
            return False
    
    async def _connect_usb(self):
        """Connect via USB interface."""
        # Simulated USB connection
        await asyncio.sleep(0.1)
        if 'usb_port' not in self.config.connection_params:
            raise ValueError("USB port not specified")
        
    async def _connect_ethernet(self):
        """Connect via Ethernet interface."""
        # Simulated Ethernet connection
        await asyncio.sleep(0.1)
        if 'ip_address' not in self.config.connection_params:
            raise ValueError("IP address not specified")
        
    async def disconnect(self) -> bool:
        """Disconnect from oscilloscope."""
        try:
            if self.acquisition_running:
                await self.stop_acquisition()
            
            self.is_connected = False
            logger.info("Oscilloscope disconnected")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to disconnect oscilloscope: {e}")
            return False
    
    async def configure(self, settings: Dict[str, Any]) -> bool:
        """Configure oscilloscope settings."""
        try:
            if not self.is_connected:
                raise RuntimeError("Device not connected")
            
            # Configure channels
            if 'channels' in settings:
                for channel, config in settings['channels'].items():
                    await self._configure_channel(channel, config)
            
            # Configure timebase
            if 'sample_rate' in settings:
                await self._set_sample_rate(settings['sample_rate'])
            
            # Configure memory depth
            if 'memory_depth' in settings:
                await self._set_memory_depth(settings['memory_depth'])
            
            # Configure trigger
            if 'trigger' in settings:
                await self._configure_trigger(settings['trigger'])
            
            logger.info("Oscilloscope configured successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to configure oscilloscope: {e}")
            return False
    
    async def _configure_channel(self, channel: str, config: Dict[str, Any]):
        """Configure individual channel."""
        # Simulate channel configuration
        await asyncio.sleep(0.01)
        logger.debug(f"Configured channel {channel}: {config}")
    
    async def _set_sample_rate(self, rate: float):
        """Set oscilloscope sample rate."""
        await asyncio.sleep(0.01)
        logger.debug(f"Set sample rate: {rate} Hz")
    
    async def _set_memory_depth(self, depth: int):
        """Set memory depth."""
        await asyncio.sleep(0.01)
        logger.debug(f"Set memory depth: {depth}")
    
    async def _configure_trigger(self, trigger_config: Dict[str, Any]):
        """Configure trigger settings."""
        await asyncio.sleep(0.01)
        logger.debug(f"Configured trigger: {trigger_config}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get oscilloscope status."""
        return {
            'connected': self.is_connected,
            'acquisition_running': self.acquisition_running,
            'buffer_size': self.trace_buffer.qsize(),
            'last_error': self.last_error
        }
    
    async def calibrate(self) -> bool:
        """Perform oscilloscope calibration."""
        try:
            logger.info("Starting oscilloscope calibration")
            
            # Simulate calibration procedure
            await asyncio.sleep(2.0)
            
            self.config.last_calibration = datetime.now()
            logger.info("Oscilloscope calibration completed")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Calibration failed: {e}")
            return False
    
    async def start_acquisition(self, config: MeasurementConfig) -> bool:
        """Start continuous trace acquisition."""
        try:
            if not self.is_connected:
                raise RuntimeError("Device not connected")
            
            self.acquisition_running = True
            
            # Start acquisition in background task
            asyncio.create_task(self._acquisition_loop(config))
            
            logger.info("Acquisition started")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to start acquisition: {e}")
            return False
    
    async def stop_acquisition(self) -> bool:
        """Stop trace acquisition."""
        try:
            self.acquisition_running = False
            logger.info("Acquisition stopped")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to stop acquisition: {e}")
            return False
    
    async def _acquisition_loop(self, config: MeasurementConfig):
        """Continuous acquisition loop."""
        while self.acquisition_running:
            try:
                # Simulate trace acquisition
                trace_data = await self._acquire_single_trace(config)
                
                # Add to buffer (non-blocking)
                try:
                    self.trace_buffer.put_nowait(trace_data)
                except queue.Full:
                    # Remove oldest trace if buffer full
                    try:
                        self.trace_buffer.get_nowait()
                        self.trace_buffer.put_nowait(trace_data)
                    except queue.Empty:
                        pass
                
                # Maintain acquisition rate
                await asyncio.sleep(1.0 / 1000.0)  # 1kHz acquisition rate
                
            except Exception as e:
                logger.error(f"Error in acquisition loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _acquire_single_trace(self, config: MeasurementConfig) -> Dict[str, np.ndarray]:
        """Acquire single multi-channel trace."""
        # Simulate realistic trace acquisition
        trace_length = config.memory_depth
        traces = {}
        
        for channel in config.channels:
            # Generate synthetic trace with realistic characteristics
            base_signal = np.random.randn(trace_length) * 0.001
            
            if channel.startswith('power'):
                # Power channel: more structured signal
                t = np.linspace(0, trace_length / config.sample_rate, trace_length)
                signal = base_signal + 0.01 * np.sin(2 * np.pi * 1000 * t)
                noise = np.random.normal(0, 0.005, trace_length)
            elif channel.startswith('em'):
                # EM channel: correlated but noisier
                signal = base_signal * 0.7 + np.random.randn(trace_length) * 0.002
                noise = np.random.normal(0, 0.01, trace_length)
            else:
                # Generic channel
                signal = base_signal
                noise = np.random.normal(0, 0.005, trace_length)
            
            traces[channel] = signal + noise
        
        return traces
    
    async def get_traces(self, n_traces: int = 1) -> List[Dict[str, np.ndarray]]:
        """Get traces from buffer."""
        traces = []
        
        for _ in range(min(n_traces, self.trace_buffer.qsize())):
            try:
                trace = self.trace_buffer.get_nowait()
                traces.append(trace)
            except queue.Empty:
                break
        
        return traces

class TargetBoard(HardwareDevice):
    """Target board/device interface."""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self.is_programmed = False
        self.current_firmware = None
    
    async def connect(self) -> bool:
        """Connect to target board."""
        try:
            logger.info(f"Connecting to target board {self.config.model}")
            
            # Simulate connection
            await asyncio.sleep(0.1)
            
            self.is_connected = True
            logger.info("Target board connected")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to connect to target board: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from target board."""
        self.is_connected = False
        return True
    
    async def configure(self, settings: Dict[str, Any]) -> bool:
        """Configure target board."""
        try:
            if 'clock_frequency' in settings:
                await self._set_clock_frequency(settings['clock_frequency'])
            
            if 'voltage' in settings:
                await self._set_voltage(settings['voltage'])
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to configure target board: {e}")
            return False
    
    async def _set_clock_frequency(self, frequency: float):
        """Set target clock frequency."""
        await asyncio.sleep(0.01)
        logger.debug(f"Set target clock frequency: {frequency} Hz")
    
    async def _set_voltage(self, voltage: float):
        """Set target supply voltage."""
        await asyncio.sleep(0.01)
        logger.debug(f"Set target voltage: {voltage} V")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get target board status."""
        return {
            'connected': self.is_connected,
            'programmed': self.is_programmed,
            'firmware': self.current_firmware
        }
    
    async def calibrate(self) -> bool:
        """Calibrate target board."""
        await asyncio.sleep(1.0)
        return True
    
    async def program_firmware(self, firmware_path: str) -> bool:
        """Program firmware to target board."""
        try:
            logger.info(f"Programming firmware: {firmware_path}")
            
            # Simulate programming
            await asyncio.sleep(3.0)
            
            self.is_programmed = True
            self.current_firmware = firmware_path
            
            logger.info("Firmware programmed successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to program firmware: {e}")
            return False
    
    async def trigger_operation(self, operation: str, data: bytes = None) -> bool:
        """Trigger cryptographic operation on target."""
        try:
            logger.debug(f"Triggering operation: {operation}")
            
            # Simulate operation trigger
            await asyncio.sleep(0.001)  # 1ms operation time
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to trigger operation: {e}")
            return False

class RealTimeAnalysisEngine:
    """Real-time side-channel analysis engine."""
    
    def __init__(self, 
                 neural_sca: NeuralSCA,
                 buffer_size: int = 1000,
                 analysis_interval: float = 1.0):
        self.neural_sca = neural_sca
        self.buffer_size = buffer_size
        self.analysis_interval = analysis_interval
        
        # Trace buffers
        self.trace_buffer = queue.Queue(maxsize=buffer_size)
        self.analysis_results = queue.Queue(maxsize=100)
        
        # Control flags
        self.analysis_running = False
        self.analysis_task = None
        
        # Statistics
        self.traces_processed = 0
        self.successful_attacks = 0
        
        logger.info("Real-time analysis engine initialized")
    
    async def start_analysis(self) -> bool:
        """Start real-time analysis."""
        try:
            if self.analysis_running:
                return True
            
            self.analysis_running = True
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            logger.info("Real-time analysis started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start analysis: {e}")
            return False
    
    async def stop_analysis(self) -> bool:
        """Stop real-time analysis."""
        try:
            self.analysis_running = False
            
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Real-time analysis stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop analysis: {e}")
            return False
    
    async def add_trace(self, trace_data: Dict[str, np.ndarray]) -> bool:
        """Add new trace for analysis."""
        try:
            # Convert to TraceData format
            traces = [trace_data['power']] if 'power' in trace_data else [list(trace_data.values())[0]]
            trace_obj = TraceData(traces=np.array(traces))
            
            self.trace_buffer.put_nowait(trace_obj)
            return True
            
        except queue.Full:
            # Remove oldest trace
            try:
                self.trace_buffer.get_nowait()
                self.trace_buffer.put_nowait(trace_obj)
                return True
            except queue.Empty:
                return False
        except Exception as e:
            logger.error(f"Failed to add trace: {e}")
            return False
    
    async def _analysis_loop(self):
        """Main analysis loop."""
        while self.analysis_running:
            try:
                # Collect traces for batch analysis
                batch_traces = []
                
                # Get available traces (up to batch size)
                for _ in range(min(32, self.trace_buffer.qsize())):
                    try:
                        trace = self.trace_buffer.get_nowait()
                        batch_traces.append(trace)
                    except queue.Empty:
                        break
                
                if batch_traces:
                    # Perform analysis
                    await self._analyze_batch(batch_traces)
                
                # Wait for next analysis interval
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _analyze_batch(self, traces: List[TraceData]):
        """Analyze batch of traces."""
        try:
            # Combine traces
            all_trace_data = []
            for trace_obj in traces:
                if hasattr(trace_obj, 'traces') and len(trace_obj.traces) > 0:
                    all_trace_data.extend(trace_obj.traces)
            
            if not all_trace_data:
                return
            
            combined_traces = TraceData(traces=np.array(all_trace_data))
            
            # Perform attack
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.neural_sca.attack, combined_traces
            )
            
            # Update statistics
            self.traces_processed += len(all_trace_data)
            if results.get('success', 0) > 0.8:
                self.successful_attacks += 1
            
            # Store results
            analysis_result = {
                'timestamp': datetime.now(),
                'traces_analyzed': len(all_trace_data),
                'success_rate': results.get('success', 0),
                'confidence': results.get('avg_confidence', 0),
                'total_traces_processed': self.traces_processed
            }
            
            try:
                self.analysis_results.put_nowait(analysis_result)
            except queue.Full:
                # Remove oldest result
                try:
                    self.analysis_results.get_nowait()
                    self.analysis_results.put_nowait(analysis_result)
                except queue.Empty:
                    pass
            
            logger.debug(f"Analyzed batch: success={results.get('success', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {e}")
    
    def get_latest_results(self, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get latest analysis results."""
        results = []
        
        for _ in range(min(n_results, self.analysis_results.qsize())):
            try:
                result = self.analysis_results.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results[::-1]  # Return in chronological order
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'traces_processed': self.traces_processed,
            'successful_attacks': self.successful_attacks,
            'success_rate': self.successful_attacks / max(1, self.traces_processed / 100),
            'buffer_size': self.trace_buffer.qsize(),
            'results_available': self.analysis_results.qsize()
        }

class HardwareInTheLoopSystem:
    """Complete hardware-in-the-loop side-channel analysis system."""
    
    def __init__(self, neural_sca: NeuralSCA):
        self.neural_sca = neural_sca
        self.devices = {}
        self.real_time_engine = RealTimeAnalysisEngine(neural_sca)
        
        # Measurement orchestration
        self.measurement_active = False
        self.measurement_task = None
        
        # Configuration
        self.system_config = {}
        
        logger.info("Hardware-in-the-loop system initialized")
    
    async def add_device(self, device_id: str, device: HardwareDevice) -> bool:
        """Add hardware device to system."""
        try:
            # Connect device
            if await device.connect():
                self.devices[device_id] = device
                logger.info(f"Added device: {device_id}")
                return True
            else:
                logger.error(f"Failed to connect device: {device_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding device {device_id}: {e}")
            return False
    
    async def remove_device(self, device_id: str) -> bool:
        """Remove device from system."""
        try:
            if device_id in self.devices:
                await self.devices[device_id].disconnect()
                del self.devices[device_id]
                logger.info(f"Removed device: {device_id}")
                return True
            else:
                logger.warning(f"Device not found: {device_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing device {device_id}: {e}")
            return False
    
    async def configure_system(self, config: Dict[str, Any]) -> bool:
        """Configure entire system."""
        try:
            self.system_config = config
            
            # Configure devices
            for device_id, device_config in config.get('devices', {}).items():
                if device_id in self.devices:
                    await self.devices[device_id].configure(device_config)
            
            logger.info("System configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure system: {e}")
            return False
    
    async def calibrate_all_devices(self) -> bool:
        """Calibrate all devices in system."""
        try:
            logger.info("Starting system calibration")
            
            for device_id, device in self.devices.items():
                logger.info(f"Calibrating {device_id}")
                if not await device.calibrate():
                    logger.error(f"Calibration failed for {device_id}")
                    return False
            
            logger.info("System calibration completed")
            return True
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    async def start_automated_measurement(self, measurement_config: MeasurementConfig) -> bool:
        """Start automated measurement and analysis."""
        try:
            if self.measurement_active:
                return True
            
            # Start real-time analysis engine
            await self.real_time_engine.start_analysis()
            
            # Start measurement orchestration
            self.measurement_active = True
            self.measurement_task = asyncio.create_task(
                self._measurement_orchestration_loop(measurement_config)
            )
            
            logger.info("Automated measurement started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start measurement: {e}")
            return False
    
    async def stop_automated_measurement(self) -> bool:
        """Stop automated measurement."""
        try:
            self.measurement_active = False
            
            # Stop measurement task
            if self.measurement_task:
                self.measurement_task.cancel()
                try:
                    await self.measurement_task
                except asyncio.CancelledError:
                    pass
            
            # Stop real-time analysis
            await self.real_time_engine.stop_analysis()
            
            # Stop device acquisitions
            for device in self.devices.values():
                if isinstance(device, OscilloscopeDevice):
                    await device.stop_acquisition()
            
            logger.info("Automated measurement stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop measurement: {e}")
            return False
    
    async def _measurement_orchestration_loop(self, config: MeasurementConfig):
        """Orchestrate synchronized measurements."""
        while self.measurement_active:
            try:
                # Trigger target operation
                await self._trigger_crypto_operation()
                
                # Collect traces from oscilloscope
                await self._collect_and_process_traces(config)
                
                # Brief pause between measurements
                await asyncio.sleep(0.01)  # 100 Hz measurement rate
                
            except Exception as e:
                logger.error(f"Error in measurement orchestration: {e}")
                await asyncio.sleep(0.1)
    
    async def _trigger_crypto_operation(self):
        """Trigger cryptographic operation on target."""
        for device in self.devices.values():
            if isinstance(device, TargetBoard):
                # Generate random data for operation
                random_data = np.random.randint(0, 256, 16, dtype=np.uint8).tobytes()
                await device.trigger_operation('aes_encrypt', random_data)
                break
    
    async def _collect_and_process_traces(self, config: MeasurementConfig):
        """Collect traces from oscilloscope and send for analysis."""
        for device in self.devices.values():
            if isinstance(device, OscilloscopeDevice):
                traces = await device.get_traces(1)
                
                for trace_data in traces:
                    await self.real_time_engine.add_trace(trace_data)
                break
    
    async def perform_campaign(self, 
                             campaign_config: Dict[str, Any],
                             n_traces: int = 10000) -> Dict[str, Any]:
        """Perform complete measurement campaign."""
        logger.info(f"Starting measurement campaign: {n_traces} traces")
        
        # Configure system
        await self.configure_system(campaign_config)
        
        # Start automated measurement
        measurement_config = MeasurementConfig(
            channels=campaign_config.get('channels', ['power']),
            sample_rate=campaign_config.get('sample_rate', 1e6),
            memory_depth=campaign_config.get('memory_depth', 10000),
            trigger_config=campaign_config.get('trigger', {})
        )
        
        await self.start_automated_measurement(measurement_config)
        
        # Wait for campaign completion
        start_time = time.time()
        target_traces = n_traces
        
        while self.real_time_engine.traces_processed < target_traces:
            await asyncio.sleep(1.0)
            
            # Progress update
            progress = self.real_time_engine.traces_processed / target_traces
            elapsed = time.time() - start_time
            
            if elapsed > 0:
                rate = self.real_time_engine.traces_processed / elapsed
                eta = (target_traces - self.real_time_engine.traces_processed) / rate
                
                logger.info(f"Campaign progress: {progress:.1%}, "
                           f"Rate: {rate:.0f} traces/sec, ETA: {eta:.0f}s")
            
            # Timeout protection
            if elapsed > 3600:  # 1 hour max
                logger.warning("Campaign timeout reached")
                break
        
        # Stop measurement
        await self.stop_automated_measurement()
        
        # Compile results
        final_results = self.real_time_engine.get_latest_results(10)
        statistics = self.real_time_engine.get_statistics()
        
        campaign_results = {
            'traces_collected': self.real_time_engine.traces_processed,
            'successful_attacks': self.real_time_engine.successful_attacks,
            'campaign_duration': time.time() - start_time,
            'final_results': final_results,
            'statistics': statistics
        }
        
        logger.info(f"Campaign completed: {campaign_results['traces_collected']} traces, "
                   f"{campaign_results['successful_attacks']} successful attacks")
        
        return campaign_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        device_status = {}
        for device_id, device in self.devices.items():
            device_status[device_id] = {
                'connected': device.is_connected,
                'last_error': device.last_error
            }
        
        return {
            'devices': device_status,
            'measurement_active': self.measurement_active,
            'analysis_running': self.real_time_engine.analysis_running,
            'traces_processed': self.real_time_engine.traces_processed,
            'successful_attacks': self.real_time_engine.successful_attacks
        }

# Factory functions for creating hardware devices
def create_oscilloscope(model: str, connection_params: Dict[str, Any]) -> OscilloscopeDevice:
    """Create oscilloscope device."""
    config = HardwareConfig(
        device_type='oscilloscope',
        model=model,
        connection_type=connection_params.get('type', 'usb'),
        connection_params=connection_params,
        capabilities={
            'max_sample_rate': connection_params.get('max_sample_rate', 1e9),
            'channels': connection_params.get('channels', 4),
            'memory_depth': connection_params.get('max_memory', 1e6)
        }
    )
    
    return OscilloscopeDevice(config)

def create_target_board(model: str, connection_params: Dict[str, Any]) -> TargetBoard:
    """Create target board device."""
    config = HardwareConfig(
        device_type='target_board',
        model=model,
        connection_type=connection_params.get('type', 'usb'),
        connection_params=connection_params,
        capabilities={
            'programmable': connection_params.get('programmable', True),
            'max_clock_freq': connection_params.get('max_clock_freq', 168e6),
            'voltage_range': connection_params.get('voltage_range', (1.8, 3.6))
        }
    )
    
    return TargetBoard(config)