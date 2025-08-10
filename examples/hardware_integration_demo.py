#!/usr/bin/env python3
"""
Hardware-in-the-Loop Integration System - Demonstration.

This example shows how to use the real-time hardware integration system
for automated side-channel analysis with oscilloscopes and target boards.
"""

import sys
import os
import asyncio
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cryptanalysis.hardware_integration import (
    HardwareInTheLoopSystem, OscilloscopeDevice, TargetBoard,
    create_oscilloscope, create_target_board,
    MeasurementConfig, HardwareConfig
)
from neural_cryptanalysis.core import NeuralSCA
from neural_cryptanalysis.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

async def demonstrate_device_creation_and_connection():
    """Demonstrate creation and connection of hardware devices."""
    print("\n" + "="*60)
    print("HARDWARE DEVICE CREATION AND CONNECTION")
    print("="*60)
    
    # Create oscilloscope device
    scope_params = {
        'type': 'usb',
        'usb_port': 'USB0::0x0699::0x0408::C000001::INSTR',
        'max_sample_rate': 5e9,
        'channels': 4,
        'max_memory': 1e6
    }
    
    oscilloscope = create_oscilloscope('Picoscope_6404D', scope_params)
    
    print(f"Created oscilloscope:")
    print(f"  Model: {oscilloscope.config.model}")
    print(f"  Connection: {oscilloscope.config.connection_type}")
    print(f"  Capabilities: {oscilloscope.config.capabilities}")
    
    # Connect to oscilloscope
    print(f"\nConnecting to oscilloscope...")
    connected = await oscilloscope.connect()
    
    if connected:
        print(f"✓ Oscilloscope connected successfully")
        
        # Get status
        status = await oscilloscope.get_status()
        print(f"  Status: {status}")
        
        # Configure oscilloscope
        scope_config = {
            'channels': {
                'A': {'range': '100mV', 'coupling': 'DC', 'enabled': True},
                'B': {'range': '50mV', 'coupling': 'AC', 'enabled': True}
            },
            'sample_rate': 1e6,
            'memory_depth': 10000,
            'trigger': {
                'channel': 'C',
                'level': 2.5,
                'edge': 'rising'
            }
        }
        
        config_success = await oscilloscope.configure(scope_config)
        print(f"  Configuration: {'Success' if config_success else 'Failed'}")
        
    else:
        print(f"✗ Failed to connect to oscilloscope: {oscilloscope.last_error}")
    
    # Create target board
    target_params = {
        'type': 'usb',
        'usb_port': '/dev/ttyUSB0',
        'programmable': True,
        'max_clock_freq': 168e6,
        'voltage_range': (1.8, 3.6)
    }
    
    target_board = create_target_board('CW308_STM32F4', target_params)
    
    print(f"\nCreated target board:")
    print(f"  Model: {target_board.config.model}")
    print(f"  Connection: {target_board.config.connection_type}")
    print(f"  Capabilities: {target_board.config.capabilities}")
    
    # Connect to target board
    print(f"\nConnecting to target board...")
    target_connected = await target_board.connect()
    
    if target_connected:
        print(f"✓ Target board connected successfully")
        
        # Configure target
        target_config = {
            'clock_frequency': 24e6,  # 24 MHz
            'voltage': 3.3           # 3.3V
        }
        
        target_config_success = await target_board.configure(target_config)
        print(f"  Configuration: {'Success' if target_config_success else 'Failed'}")
        
        # Program firmware (simulated)
        firmware_path = "aes_masked_implementation.hex"
        programming_success = await target_board.program_firmware(firmware_path)
        print(f"  Firmware programming: {'Success' if programming_success else 'Failed'}")
        
        if programming_success:
            # Test operation trigger
            trigger_success = await target_board.trigger_operation('aes_encrypt', b'\x00' * 16)
            print(f"  Operation trigger: {'Success' if trigger_success else 'Failed'}")
        
    else:
        print(f"✗ Failed to connect to target board: {target_board.last_error}")
    
    # Cleanup
    if connected:
        await oscilloscope.disconnect()
    if target_connected:
        await target_board.disconnect()
    
    return oscilloscope, target_board

async def demonstrate_real_time_analysis():
    """Demonstrate real-time analysis engine."""
    print("\n" + "="*60)
    print("REAL-TIME ANALYSIS ENGINE")
    print("="*60)
    
    # Initialize neural SCA
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    
    # Create hardware-in-the-loop system
    hitl_system = HardwareInTheLoopSystem(neural_sca)
    
    print(f"Hardware-in-the-loop system initialized")
    
    # Add simulated devices
    print(f"\nAdding hardware devices...")
    
    # Oscilloscope
    scope_params = {
        'type': 'ethernet',
        'ip_address': '192.168.1.100',
        'max_sample_rate': 1e9,
        'channels': 2,
        'max_memory': 100000
    }
    oscilloscope = create_oscilloscope('Keysight_DSOX3034A', scope_params)
    scope_added = await hitl_system.add_device('main_scope', oscilloscope)
    print(f"  Oscilloscope: {'Added' if scope_added else 'Failed'}")
    
    # Target board
    target_params = {
        'type': 'serial',
        'serial_port': '/dev/ttyACM0',
        'baud_rate': 115200
    }
    target_board = create_target_board('ChipWhisperer_Lite', target_params)
    target_added = await hitl_system.add_device('cw_lite', target_board)
    print(f"  Target board: {'Added' if target_added else 'Failed'}")
    
    if scope_added and target_added:
        # Configure system
        system_config = {
            'devices': {
                'main_scope': {
                    'channels': {
                        'power': {'range': '50mV', 'coupling': 'DC'},
                        'trigger': {'range': '5V', 'coupling': 'DC'}
                    },
                    'sample_rate': 5e5,  # 500 kS/s
                    'memory_depth': 5000
                },
                'cw_lite': {
                    'clock_frequency': 7.37e6,  # 7.37 MHz
                    'voltage': 3.3
                }
            }
        }
        
        config_success = await hitl_system.configure_system(system_config)
        print(f"  System configuration: {'Success' if config_success else 'Failed'}")
        
        if config_success:
            # Start real-time measurement and analysis
            measurement_config = MeasurementConfig(
                channels=['power', 'trigger'],
                sample_rate=5e5,
                memory_depth=5000,
                trigger_config={
                    'channel': 'trigger',
                    'level': 2.5,
                    'edge': 'rising'
                },
                real_time_processing=True
            )
            
            print(f"\nStarting automated measurement...")
            measurement_started = await hitl_system.start_automated_measurement(measurement_config)
            
            if measurement_started:
                print(f"✓ Automated measurement started")
                
                # Let it run for a few seconds to collect data
                print(f"  Collecting traces for 10 seconds...")
                
                for i in range(10):
                    await asyncio.sleep(1)
                    
                    # Get real-time results
                    latest_results = hitl_system.real_time_engine.get_latest_results(1)
                    if latest_results:
                        result = latest_results[0]
                        print(f"    t={i+1:2d}s: {result['traces_analyzed']:3d} traces, "
                              f"success={result['success_rate']:.3f}, "
                              f"confidence={result['confidence']:.3f}")
                    else:
                        print(f"    t={i+1:2d}s: No results yet")
                
                # Get final statistics
                stats = hitl_system.real_time_engine.get_statistics()
                print(f"\nReal-time analysis statistics:")
                print(f"  Total traces processed: {stats['traces_processed']}")
                print(f"  Successful attacks: {stats['successful_attacks']}")
                print(f"  Overall success rate: {stats['success_rate']:.3f}")
                
                # Stop measurement
                await hitl_system.stop_automated_measurement()
                print(f"✓ Automated measurement stopped")
            
            else:
                print(f"✗ Failed to start automated measurement")
    
    # Get system status
    system_status = hitl_system.get_system_status()
    print(f"\nFinal system status:")
    for device_id, status in system_status['devices'].items():
        print(f"  {device_id}: Connected={status['connected']}")
    
    return hitl_system

async def demonstrate_measurement_campaign():
    """Demonstrate automated measurement campaign."""
    print("\n" + "="*60)
    print("AUTOMATED MEASUREMENT CAMPAIGN")
    print("="*60)
    
    # Initialize system
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    hitl_system = HardwareInTheLoopSystem(neural_sca)
    
    # Add devices (simulated for demo)
    scope_params = {'type': 'usb', 'usb_port': 'USB0::INSTR'}
    target_params = {'type': 'serial', 'serial_port': '/dev/ttyUSB0'}
    
    oscilloscope = create_oscilloscope('Demo_Scope', scope_params)
    target_board = create_target_board('Demo_Target', target_params)
    
    await hitl_system.add_device('scope', oscilloscope)
    await hitl_system.add_device('target', target_board)
    
    # Define campaign configuration
    campaign_config = {
        'channels': ['power', 'em_near'],
        'sample_rate': 1e6,  # 1 MS/s
        'memory_depth': 8000,
        'trigger': {
            'channel': 'trigger',
            'level': 2.5,
            'edge': 'rising',
            'timeout_ms': 1000
        },
        'devices': {
            'scope': {
                'channels': {
                    'power': {'range': '20mV', 'coupling': 'DC'},
                    'em_near': {'range': '10mV', 'coupling': 'AC'},
                    'trigger': {'range': '5V', 'coupling': 'DC'}
                }
            },
            'target': {
                'clock_frequency': 16e6,  # 16 MHz
                'voltage': 3.0
            }
        }
    }
    
    print(f"Campaign Configuration:")
    print(f"  Channels: {campaign_config['channels']}")
    print(f"  Sample Rate: {campaign_config['sample_rate']:,.0f} S/s")
    print(f"  Memory Depth: {campaign_config['memory_depth']:,}")
    print(f"  Target Clock: {campaign_config['devices']['target']['clock_frequency']:,.0f} Hz")
    
    # Run campaign
    n_traces = 5000  # Reduced for demo
    print(f"\nStarting measurement campaign ({n_traces:,} traces)...")
    
    start_time = time.time()
    
    try:
        campaign_results = await hitl_system.perform_campaign(
            campaign_config=campaign_config,
            n_traces=n_traces
        )
        
        campaign_duration = time.time() - start_time
        
        print(f"\nCampaign Results:")
        print(f"  Duration: {campaign_duration:.1f} seconds")
        print(f"  Traces collected: {campaign_results['traces_collected']:,}")
        print(f"  Successful attacks: {campaign_results['successful_attacks']}")
        print(f"  Collection rate: {campaign_results['traces_collected'] / campaign_duration:.0f} traces/sec")
        
        # Show final results
        if campaign_results['final_results']:
            final_result = campaign_results['final_results'][0]
            print(f"  Final success rate: {final_result['success_rate']:.3f}")
            print(f"  Final confidence: {final_result['confidence']:.3f}")
        
        # Performance statistics
        statistics = campaign_results['statistics']
        print(f"\nPerformance Statistics:")
        print(f"  Buffer utilization: {statistics['buffer_size']}/{statistics['buffer_size']} (full)")
        print(f"  Results available: {statistics['results_available']}")
        
        return campaign_results
        
    except Exception as e:
        print(f"Campaign failed: {e}")
        return None

async def demonstrate_multi_device_coordination():
    """Demonstrate coordination of multiple devices."""
    print("\n" + "="*60)
    print("MULTI-DEVICE COORDINATION")
    print("="*60)
    
    # Initialize system with multiple devices
    neural_sca = NeuralSCA(architecture='fourier_neural_operator')
    hitl_system = HardwareInTheLoopSystem(neural_sca)
    
    # Add multiple oscilloscopes
    devices_config = [
        ('scope_1', 'Picoscope_6404D', {'type': 'usb', 'usb_port': 'USB0::INSTR'}),
        ('scope_2', 'Keysight_DSOX3034A', {'type': 'ethernet', 'ip_address': '192.168.1.101'}),
        ('target_1', 'CW308_STM32F4', {'type': 'serial', 'serial_port': '/dev/ttyUSB0'}),
        ('target_2', 'CW308_STM32F4', {'type': 'serial', 'serial_port': '/dev/ttyUSB1'})
    ]
    
    print(f"Adding {len(devices_config)} devices...")
    
    connected_devices = []
    for device_id, model, params in devices_config:
        if 'scope' in device_id:
            device = create_oscilloscope(model, params)
        else:
            device = create_target_board(model, params)
        
        success = await hitl_system.add_device(device_id, device)
        if success:
            connected_devices.append(device_id)
            print(f"  ✓ {device_id} ({model})")
        else:
            print(f"  ✗ {device_id} ({model}) - Connection failed")
    
    print(f"\nConnected devices: {len(connected_devices)}")
    
    if len(connected_devices) >= 2:
        # Configure multi-device system
        multi_config = {
            'synchronization': {
                'master_clock': 'external_10mhz',
                'trigger_distribution': 'star_topology'
            },
            'devices': {}
        }
        
        # Configure each connected device
        for device_id in connected_devices:
            if 'scope' in device_id:
                multi_config['devices'][device_id] = {
                    'channels': {
                        'power': {'range': '50mV', 'coupling': 'DC'},
                        'trigger': {'range': '5V', 'coupling': 'DC'}
                    },
                    'sample_rate': 1e6,
                    'sync_source': 'external'
                }
            else:  # target
                multi_config['devices'][device_id] = {
                    'clock_frequency': 24e6,
                    'voltage': 3.3,
                    'sync_output': True
                }
        
        print(f"\nConfiguring synchronized measurement...")
        config_success = await hitl_system.configure_system(multi_config)
        
        if config_success:
            print(f"✓ Multi-device configuration successful")
            
            # Calibrate all devices
            print(f"Calibrating all devices...")
            calibration_success = await hitl_system.calibrate_all_devices()
            
            if calibration_success:
                print(f"✓ All devices calibrated")
                
                # Demonstrate synchronized operation
                print(f"\nDemonstrating synchronized operation...")
                
                measurement_config = MeasurementConfig(
                    channels=['power'],
                    sample_rate=1e6,
                    memory_depth=5000,
                    trigger_config={
                        'channel': 'trigger',
                        'level': 2.5,
                        'sync_mode': 'multi_device'
                    }
                )
                
                # Run short synchronized measurement
                await hitl_system.start_automated_measurement(measurement_config)
                
                print(f"  Running synchronized measurement for 5 seconds...")
                await asyncio.sleep(5)
                
                # Get results
                stats = hitl_system.real_time_engine.get_statistics()
                print(f"  Synchronized traces: {stats['traces_processed']}")
                
                await hitl_system.stop_automated_measurement()
                
            else:
                print(f"✗ Device calibration failed")
        else:
            print(f"✗ Multi-device configuration failed")
    
    # Final system status
    system_status = hitl_system.get_system_status()
    print(f"\nFinal System Status:")
    print(f"  Total devices: {len(system_status['devices'])}")
    print(f"  Measurement active: {system_status['measurement_active']}")
    print(f"  Analysis running: {system_status['analysis_running']}")
    
    return hitl_system

def main():
    """Run all hardware integration demonstrations."""
    print("NEURAL OPERATOR CRYPTANALYSIS - HARDWARE INTEGRATION DEMONSTRATIONS")
    print("=" * 70)
    
    async def run_demos():
        try:
            # Device creation and connection
            devices = await demonstrate_device_creation_and_connection()
            
            # Real-time analysis
            hitl_system = await demonstrate_real_time_analysis()
            
            # Measurement campaign
            campaign_results = await demonstrate_measurement_campaign()
            
            # Multi-device coordination
            multi_system = await demonstrate_multi_device_coordination()
            
            print("\n" + "="*70)
            print("ALL HARDWARE INTEGRATION DEMONSTRATIONS COMPLETED")
            print("="*70)
            
            print(f"\nSummary:")
            print(f"  Device creation: 2 devices (oscilloscope, target board)")
            print(f"  Real-time analysis: System operational")
            
            if campaign_results:
                print(f"  Measurement campaign: {campaign_results['traces_collected']:,} traces collected")
                print(f"  Campaign success rate: {campaign_results.get('final_results', [{}])[0].get('success_rate', 0):.3f}")
            else:
                print(f"  Measurement campaign: Failed")
            
            print(f"  Multi-device coordination: {len(multi_system.devices)} devices coordinated")
            
        except Exception as e:
            print(f"Error during demonstrations: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    return asyncio.run(run_demos())

if __name__ == "__main__":
    exit(main())