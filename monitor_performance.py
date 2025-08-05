#!/usr/bin/env python3
"""
Performance Monitoring Script for PlayDiffusion RVC
Monitors CPU and GPU usage during voice conversion operations.
"""

import psutil
import torch
import time
import threading
import json
from datetime import datetime
import os


class PerformanceMonitor:
    def __init__(self, log_file="performance_log.json"):
        self.log_file = log_file
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start monitoring CPU and GPU usage"""
        if self.monitoring:
            return

        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Performance monitoring started...")

    def stop_monitoring(self):
        """Stop monitoring and save results"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self._save_metrics()
        print("Performance monitoring stopped.")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                # GPU metrics
                gpu_metrics = self._get_gpu_metrics()

                # Record metrics
                metric = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'gpu_metrics': gpu_metrics
                }

                self.metrics.append(metric)

                # Print current status
                print(
                    f"CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}% | GPU: {gpu_metrics}")

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)

    def _get_gpu_metrics(self):
        """Get GPU usage metrics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        try:
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(
                device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(
                device) / 1024**3    # GB
            memory_total = torch.cuda.get_device_properties(
                device).total_memory / 1024**3  # GB

            # Try to get GPU utilization (requires nvidia-ml-py)
            gpu_util = "N/A"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except ImportError:
                pass

            return {
                "device": device,
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_util_percent": round((memory_allocated / memory_total) * 100, 1),
                "gpu_util_percent": gpu_util
            }
        except Exception as e:
            return {"error": str(e)}

    def _save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'monitoring_session': {
                        'start_time': self.metrics[0]['timestamp'] if self.metrics else None,
                        'end_time': self.metrics[-1]['timestamp'] if self.metrics else None,
                        'total_samples': len(self.metrics)
                    },
                    'metrics': self.metrics
                }, f, indent=2)
            print(f"Performance metrics saved to {self.log_file}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")

    def get_summary(self):
        """Get performance summary"""
        if not self.metrics:
            return "No metrics available"

        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]

        summary = {
            'cpu': {
                'avg': round(sum(cpu_values) / len(cpu_values), 1),
                'max': round(max(cpu_values), 1),
                'min': round(min(cpu_values), 1)
            },
            'memory': {
                'avg': round(sum(memory_values) / len(memory_values), 1),
                'max': round(max(memory_values), 1),
                'min': round(min(memory_values), 1)
            },
            'duration_seconds': len(self.metrics)
        }

        return summary


# Global monitor instance
monitor = PerformanceMonitor()


def start_monitoring():
    """Start performance monitoring"""
    monitor.start_monitoring()


def stop_monitoring():
    """Stop performance monitoring"""
    monitor.stop_monitoring()


def get_performance_summary():
    """Get performance summary"""
    return monitor.get_summary()


if __name__ == "__main__":
    # Example usage
    print("Starting performance monitoring...")
    start_monitoring()

    try:
        # Simulate some work
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        stop_monitoring()
        print("Performance Summary:")
        print(json.dumps(get_performance_summary(), indent=2))
