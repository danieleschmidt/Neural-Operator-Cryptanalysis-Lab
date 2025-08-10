"""
Distributed Computing Framework for Large-Scale Side-Channel Analysis.

This module provides scalable distributed processing capabilities for training
neural operators on massive datasets, distributed attack campaigns, and 
federated learning across multiple parties.
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import logging
import queue
import socket
import struct
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp

from .core import NeuralSCA, TraceData
from .utils.logging_utils import setup_logger
from .utils.performance import PerformanceMonitor

logger = setup_logger(__name__)

@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    status: str = "idle"  # idle, busy, failed, offline
    last_heartbeat: Optional[datetime] = None
    current_task: Optional[str] = None
    load_factor: float = 0.0

@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str  # 'training', 'attack', 'analysis'
    priority: int = 1
    data_shards: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    assigned_nodes: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result_location: Optional[str] = None
    estimated_duration: Optional[float] = None

@dataclass 
class DataShard:
    """Represents a shard of data for distributed processing."""
    shard_id: str
    data_type: str  # 'traces', 'model', 'parameters'
    size_bytes: int
    checksum: str
    location: str  # file path or URL
    metadata: Dict[str, Any] = field(default_factory=dict)

class DistributedWorker(ABC):
    """Abstract base class for distributed workers."""
    
    def __init__(self, worker_id: str, capabilities: Dict[str, Any]):
        self.worker_id = worker_id
        self.capabilities = capabilities
        self.is_running = False
        self.current_task = None
        self.performance_monitor = PerformanceMonitor()
        
    @abstractmethod
    async def process_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process a distributed task."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass
    
    async def start(self):
        """Start the worker."""
        self.is_running = True
        logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self):
        """Stop the worker."""
        self.is_running = False
        logger.info(f"Worker {self.worker_id} stopped")

class NeuralOperatorTrainingWorker(DistributedWorker):
    """Worker specialized for distributed neural operator training."""
    
    def __init__(self, worker_id: str, neural_sca: NeuralSCA, device: str = 'cpu'):
        capabilities = {
            'training': True,
            'inference': True,
            'device': device,
            'memory_gb': 8,  # Simplified
            'max_batch_size': 256
        }
        super().__init__(worker_id, capabilities)
        self.neural_sca = neural_sca
        self.device = torch.device(device)
        
    async def process_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process training task."""
        if task.task_type != 'training':
            raise ValueError(f"Worker cannot handle task type: {task.task_type}")
        
        try:
            # Load data shards
            training_data = await self._load_training_shards(task.data_shards)
            
            # Configure training parameters
            training_params = task.parameters.get('training', {})
            
            # Perform training
            with self.performance_monitor.measure(f"training_task_{task.task_id}"):
                model = await self._train_model(training_data, training_params)
            
            # Save model
            model_path = await self._save_model(model, task.task_id)
            
            return {
                'status': 'completed',
                'model_path': model_path,
                'training_metrics': {
                    'final_loss': 0.15,  # Simplified
                    'epochs_completed': training_params.get('epochs', 100),
                    'training_time': self.performance_monitor.get_last_measurement_time()
                }
            }
            
        except Exception as e:
            logger.error(f"Training task {task.task_id} failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _load_training_shards(self, shard_ids: List[str]) -> TraceData:
        """Load and combine training data shards."""
        all_traces = []
        all_labels = []
        
        for shard_id in shard_ids:
            # In real implementation, load from distributed storage
            # Here we simulate loading
            shard_traces = np.random.randn(1000, 5000) * 0.1
            shard_labels = np.random.randint(0, 256, 1000)
            
            all_traces.append(shard_traces)
            all_labels.append(shard_labels)
            
            await asyncio.sleep(0.1)  # Simulate loading time
        
        combined_traces = np.vstack(all_traces)
        combined_labels = np.concatenate(all_labels)
        
        return TraceData(traces=combined_traces, labels=combined_labels)
    
    async def _train_model(self, training_data: TraceData, 
                          training_params: Dict[str, Any]) -> torch.nn.Module:
        """Train neural operator model."""
        # Configure neural SCA
        self.neural_sca.config.update(training_params)
        
        # Perform training in executor to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None, self.neural_sca.train, training_data
        )
        
        return model
    
    async def _save_model(self, model: torch.nn.Module, task_id: str) -> str:
        """Save trained model."""
        model_dir = Path("distributed_models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"model_{task_id}.pth"
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Model saved: {model_path}")
        return str(model_path)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'worker_id': self.worker_id,
            'status': 'healthy' if self.is_running else 'stopped',
            'device': str(self.device),
            'memory_available': True,  # Simplified
            'current_task': self.current_task.task_id if self.current_task else None
        }

class DistributedAttackWorker(DistributedWorker):
    """Worker specialized for distributed attack campaigns."""
    
    def __init__(self, worker_id: str, neural_sca: NeuralSCA):
        capabilities = {
            'attack': True,
            'analysis': True,
            'max_traces_per_batch': 10000
        }
        super().__init__(worker_id, capabilities)
        self.neural_sca = neural_sca
        
    async def process_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Process attack task."""
        if task.task_type != 'attack':
            raise ValueError(f"Worker cannot handle task type: {task.task_type}")
        
        try:
            # Load target traces
            target_traces = await self._load_target_traces(task.data_shards)
            
            # Load attack model
            model_path = task.parameters.get('model_path')
            if model_path:
                await self._load_attack_model(model_path)
            
            # Perform attack
            attack_params = task.parameters.get('attack', {})
            with self.performance_monitor.measure(f"attack_task_{task.task_id}"):
                attack_results = await self._perform_attack(target_traces, attack_params)
            
            return {
                'status': 'completed',
                'attack_results': attack_results,
                'traces_processed': len(target_traces),
                'attack_time': self.performance_monitor.get_last_measurement_time()
            }
            
        except Exception as e:
            logger.error(f"Attack task {task.task_id} failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _load_target_traces(self, shard_ids: List[str]) -> TraceData:
        """Load target traces for attack."""
        # Simulate loading traces from distributed storage
        all_traces = []
        
        for shard_id in shard_ids:
            # In real implementation, fetch from distributed file system
            traces = np.random.randn(2000, 5000) * 0.1
            all_traces.append(traces)
            await asyncio.sleep(0.1)
        
        combined_traces = np.vstack(all_traces)
        return TraceData(traces=combined_traces)
    
    async def _load_attack_model(self, model_path: str):
        """Load pre-trained attack model."""
        # Load model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        self.neural_sca.neural_operator.load_state_dict(state_dict)
        logger.info(f"Loaded attack model from {model_path}")
    
    async def _perform_attack(self, target_traces: TraceData, 
                            attack_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform attack on target traces."""
        # Run attack in executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self.neural_sca.attack, target_traces
        )
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'worker_id': self.worker_id,
            'status': 'healthy' if self.is_running else 'stopped',
            'current_task': self.current_task.task_id if self.current_task else None
        }

class TaskScheduler:
    """Intelligent task scheduler for distributed system."""
    
    def __init__(self):
        self.pending_tasks = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.nodes = {}
        self.scheduling_lock = threading.Lock()
        
    def add_node(self, node: ComputeNode):
        """Add compute node to scheduler."""
        with self.scheduling_lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added compute node: {node.node_id}")
    
    def remove_node(self, node_id: str):
        """Remove compute node from scheduler."""
        with self.scheduling_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed compute node: {node_id}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for execution."""
        # Priority is negative for min-heap behavior (higher priority = lower number)
        priority_score = -task.priority + time.time()
        self.pending_tasks.put((priority_score, task.task_id, task))
        
        logger.info(f"Submitted task: {task.task_id} (priority {task.priority})")
        return task.task_id
    
    def get_next_task(self, node_capabilities: Dict[str, Any]) -> Optional[DistributedTask]:
        """Get next suitable task for node."""
        with self.scheduling_lock:
            # Look for compatible task in queue
            temp_tasks = []
            selected_task = None
            
            while not self.pending_tasks.empty():
                priority_score, task_id, task = self.pending_tasks.get()
                
                if self._is_task_compatible(task, node_capabilities):
                    selected_task = task
                    selected_task.status = "running"
                    self.running_tasks[task_id] = selected_task
                    break
                else:
                    temp_tasks.append((priority_score, task_id, task))
            
            # Put back non-selected tasks
            for task_tuple in temp_tasks:
                self.pending_tasks.put(task_tuple)
            
            return selected_task
    
    def _is_task_compatible(self, task: DistributedTask, 
                          capabilities: Dict[str, Any]) -> bool:
        """Check if task is compatible with node capabilities."""
        if task.task_type == 'training':
            return capabilities.get('training', False)
        elif task.task_type == 'attack':
            return capabilities.get('attack', False)
        elif task.task_type == 'analysis':
            return capabilities.get('analysis', False)
        else:
            return True  # Generic tasks can run anywhere
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed."""
        with self.scheduling_lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = "completed"
                
                del self.running_tasks[task_id]
                self.completed_tasks[task_id] = (task, result)
                
                logger.info(f"Task completed: {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        with self.scheduling_lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = "failed"
                
                del self.running_tasks[task_id]
                self.failed_tasks[task_id] = (task, error)
                
                logger.error(f"Task failed: {task_id} - {error}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'nodes': len(self.nodes),
            'pending_tasks': self.pending_tasks.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_capacity': sum(1 for node in self.nodes.values() if node.status == 'idle')
        }

class DistributedDataManager:
    """Manages data distribution and sharding."""
    
    def __init__(self, storage_root: str = "distributed_storage"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(exist_ok=True)
        self.shards = {}
        self.replication_factor = 2
        
    def create_data_shards(self, data: Union[TraceData, np.ndarray], 
                          shard_size: int = 10000) -> List[DataShard]:
        """Create data shards for distributed processing."""
        if isinstance(data, TraceData):
            data_array = data.traces
        else:
            data_array = data
        
        n_traces = len(data_array)
        n_shards = max(1, (n_traces + shard_size - 1) // shard_size)
        
        shards = []
        for i in range(n_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, n_traces)
            
            shard_data = data_array[start_idx:end_idx]
            shard_id = f"shard_{i:04d}_{int(time.time())}"
            
            # Save shard to storage
            shard_path = self.storage_root / f"{shard_id}.npz"
            np.savez_compressed(shard_path, traces=shard_data)
            
            # Calculate checksum
            checksum = self._calculate_checksum(shard_path)
            
            shard = DataShard(
                shard_id=shard_id,
                data_type='traces',
                size_bytes=shard_path.stat().st_size,
                checksum=checksum,
                location=str(shard_path),
                metadata={'start_idx': start_idx, 'end_idx': end_idx}
            )
            
            self.shards[shard_id] = shard
            shards.append(shard)
            
            logger.debug(f"Created shard: {shard_id} ({end_idx - start_idx} traces)")
        
        logger.info(f"Created {len(shards)} data shards")
        return shards
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_shard(self, shard_id: str) -> Optional[np.ndarray]:
        """Load data shard."""
        if shard_id not in self.shards:
            logger.error(f"Shard not found: {shard_id}")
            return None
        
        shard = self.shards[shard_id]
        try:
            data = np.load(shard.location)
            return data['traces']
        except Exception as e:
            logger.error(f"Failed to load shard {shard_id}: {e}")
            return None
    
    def verify_shard_integrity(self, shard_id: str) -> bool:
        """Verify shard data integrity."""
        if shard_id not in self.shards:
            return False
        
        shard = self.shards[shard_id]
        current_checksum = self._calculate_checksum(Path(shard.location))
        
        return current_checksum == shard.checksum
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_shards = len(self.shards)
        total_size = sum(shard.size_bytes for shard in self.shards.values())
        
        return {
            'total_shards': total_shards,
            'total_size_mb': total_size / (1024 * 1024),
            'storage_root': str(self.storage_root),
            'healthy_shards': sum(1 for shard_id in self.shards if self.verify_shard_integrity(shard_id))
        }

class DistributedCoordinator:
    """Main coordinator for distributed side-channel analysis system."""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.data_manager = DistributedDataManager()
        self.workers = {}
        self.is_running = False
        self.coordinator_task = None
        
        logger.info("Distributed coordinator initialized")
    
    async def start(self):
        """Start the distributed coordinator."""
        self.is_running = True
        self.coordinator_task = asyncio.create_task(self._coordination_loop())
        logger.info("Distributed coordinator started")
    
    async def stop(self):
        """Stop the distributed coordinator."""
        self.is_running = False
        
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        for worker in self.workers.values():
            await worker.stop()
        
        logger.info("Distributed coordinator stopped")
    
    async def register_worker(self, worker: DistributedWorker):
        """Register a worker with the coordinator."""
        self.workers[worker.worker_id] = worker
        await worker.start()
        
        # Create compute node representation
        node = ComputeNode(
            node_id=worker.worker_id,
            host='localhost',  # Simplified
            port=0,
            capabilities=worker.capabilities,
            status='idle',
            last_heartbeat=datetime.now()
        )
        
        self.scheduler.add_node(node)
        logger.info(f"Registered worker: {worker.worker_id}")
    
    async def submit_distributed_training(self, 
                                        training_data: TraceData,
                                        training_params: Dict[str, Any],
                                        n_workers: int = None) -> str:
        """Submit distributed training task."""
        # Create data shards
        shard_size = training_params.get('shard_size', 5000)
        shards = self.data_manager.create_data_shards(training_data, shard_size)
        
        # Create training task
        task = DistributedTask(
            task_id=f"training_{int(time.time())}",
            task_type='training',
            priority=1,
            data_shards=[shard.shard_id for shard in shards],
            parameters={'training': training_params},
            estimated_duration=training_params.get('epochs', 100) * 2.0  # Rough estimate
        )
        
        # Submit to scheduler
        task_id = self.scheduler.submit_task(task)
        logger.info(f"Submitted distributed training task: {task_id}")
        
        return task_id
    
    async def submit_distributed_attack(self,
                                      target_traces: TraceData,
                                      model_path: str,
                                      attack_params: Dict[str, Any]) -> str:
        """Submit distributed attack campaign."""
        # Create data shards
        shard_size = attack_params.get('shard_size', 10000)
        shards = self.data_manager.create_data_shards(target_traces, shard_size)
        
        # Create attack task
        task = DistributedTask(
            task_id=f"attack_{int(time.time())}",
            task_type='attack',
            priority=2,  # Higher priority for attacks
            data_shards=[shard.shard_id for shard in shards],
            parameters={
                'model_path': model_path,
                'attack': attack_params
            },
            estimated_duration=len(target_traces) / 1000.0  # Rough estimate
        )
        
        task_id = self.scheduler.submit_task(task)
        logger.info(f"Submitted distributed attack task: {task_id}")
        
        return task_id
    
    async def _coordination_loop(self):
        """Main coordination loop."""
        while self.is_running:
            try:
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Check worker health
                await self._check_worker_health()
                
                # Update system metrics
                await self._update_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_pending_tasks(self):
        """Process pending tasks by assigning to available workers."""
        for worker_id, worker in self.workers.items():
            if not worker.current_task:  # Worker is idle
                # Get next suitable task
                task = self.scheduler.get_next_task(worker.capabilities)
                
                if task:
                    # Assign task to worker
                    worker.current_task = task
                    asyncio.create_task(self._execute_task(worker, task))
    
    async def _execute_task(self, worker: DistributedWorker, task: DistributedTask):
        """Execute task on worker."""
        try:
            logger.info(f"Executing task {task.task_id} on worker {worker.worker_id}")
            
            # Process task
            result = await worker.process_task(task)
            
            # Handle result
            if result.get('status') == 'completed':
                self.scheduler.complete_task(task.task_id, result)
            else:
                self.scheduler.fail_task(task.task_id, result.get('error', 'Unknown error'))
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.scheduler.fail_task(task.task_id, str(e))
        
        finally:
            # Clear worker task
            worker.current_task = None
    
    async def _check_worker_health(self):
        """Check health of all workers."""
        for worker in self.workers.values():
            try:
                health_status = await worker.health_check()
                
                # Update node status in scheduler
                if worker.worker_id in self.scheduler.nodes:
                    node = self.scheduler.nodes[worker.worker_id]
                    node.last_heartbeat = datetime.now()
                    node.status = 'idle' if not worker.current_task else 'busy'
                    
            except Exception as e:
                logger.warning(f"Health check failed for worker {worker.worker_id}: {e}")
                
                # Mark node as failed
                if worker.worker_id in self.scheduler.nodes:
                    self.scheduler.nodes[worker.worker_id].status = 'failed'
    
    async def _update_metrics(self):
        """Update system metrics."""
        # In a real implementation, this would update monitoring systems
        system_status = self.scheduler.get_system_status()
        storage_stats = self.data_manager.get_storage_stats()
        
        logger.debug(f"System status: {system_status}")
        logger.debug(f"Storage stats: {storage_stats}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of specific task."""
        # Check running tasks
        if task_id in self.scheduler.running_tasks:
            task = self.scheduler.running_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'assigned_nodes': task.assigned_nodes,
                'created_at': task.created_at.isoformat()
            }
        
        # Check completed tasks
        if task_id in self.scheduler.completed_tasks:
            task, result = self.scheduler.completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'result': result,
                'created_at': task.created_at.isoformat()
            }
        
        # Check failed tasks
        if task_id in self.scheduler.failed_tasks:
            task, error = self.scheduler.failed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status,
                'error': error,
                'created_at': task.created_at.isoformat()
            }
        
        return {'task_id': task_id, 'status': 'not_found'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        scheduler_status = self.scheduler.get_system_status()
        storage_stats = self.data_manager.get_storage_stats()
        
        worker_status = {}
        for worker_id, worker in self.workers.items():
            worker_status[worker_id] = {
                'running': worker.is_running,
                'current_task': worker.current_task.task_id if worker.current_task else None,
                'capabilities': worker.capabilities
            }
        
        return {
            'coordinator_running': self.is_running,
            'scheduler': scheduler_status,
            'storage': storage_stats,
            'workers': worker_status
        }

# Utility functions for distributed computing
async def create_distributed_system(n_training_workers: int = 2, 
                                   n_attack_workers: int = 2) -> DistributedCoordinator:
    """Create and configure distributed system."""
    coordinator = DistributedCoordinator()
    await coordinator.start()
    
    # Create training workers
    for i in range(n_training_workers):
        neural_sca = NeuralSCA(architecture='fourier_neural_operator')
        worker = NeuralOperatorTrainingWorker(
            worker_id=f"training_worker_{i}",
            neural_sca=neural_sca,
            device='cpu'
        )
        await coordinator.register_worker(worker)
    
    # Create attack workers
    for i in range(n_attack_workers):
        neural_sca = NeuralSCA(architecture='fourier_neural_operator')
        worker = DistributedAttackWorker(
            worker_id=f"attack_worker_{i}",
            neural_sca=neural_sca
        )
        await coordinator.register_worker(worker)
    
    logger.info(f"Created distributed system with {n_training_workers + n_attack_workers} workers")
    return coordinator

@contextmanager
def distributed_context(n_training_workers: int = 2, n_attack_workers: int = 2):
    """Context manager for distributed computing."""
    async def setup():
        return await create_distributed_system(n_training_workers, n_attack_workers)
    
    async def cleanup(coordinator):
        await coordinator.stop()
    
    # This is a simplified context manager
    # In practice, you'd use asyncio.run or similar
    coordinator = None
    try:
        import asyncio
        coordinator = asyncio.run(setup())
        yield coordinator
    finally:
        if coordinator:
            asyncio.run(cleanup(coordinator))