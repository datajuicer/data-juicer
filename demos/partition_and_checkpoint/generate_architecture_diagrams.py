#!/usr/bin/env python3
"""
Architecture Diagram Generator for Data-Juicer Partitioning/Checkpointing/Event-Logging System

This script generates visual diagrams showing the architecture and data flow
of the partitioning, checkpointing, and event logging system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path
import os

# Set up matplotlib for better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_system_architecture_diagram():
    """Create the high-level system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'output': '#E8FDF5',
        'core': '#F0F8FF',
        'component': '#FFF8E1',
        'event': '#F3E5F5',
        'storage': '#E8F5E8'
    }
    
    # Title
    ax.text(9, 13.2, 'Data-Juicer: Partitioning, Checkpointing & Event Logging System', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input Section
    input_box = FancyBboxPatch((1, 11), 4, 2, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(3, 12.2, 'Input Dataset', fontsize=13, fontweight='bold', ha='center')
    ax.text(3, 11.8, '• JSONL/Parquet Files', fontsize=10, ha='center')
    ax.text(3, 11.5, '• Large Datasets', fontsize=10, ha='center')
    ax.text(3, 11.2, '• Remote URLs', fontsize=10, ha='center')
    
    # Configuration Section
    config_box = FancyBboxPatch((7, 11), 4, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(config_box)
    ax.text(9, 12.2, 'Configuration', fontsize=13, fontweight='bold', ha='center')
    ax.text(9, 11.8, '• YAML Config Files', fontsize=10, ha='center')
    ax.text(9, 11.5, '• Pipeline Operations', fontsize=10, ha='center')
    ax.text(9, 11.2, '• System Settings', fontsize=10, ha='center')
    
    # Work Directory Section
    work_box = FancyBboxPatch((13, 11), 4, 2, boxstyle="round,pad=0.1", 
                             facecolor=colors['storage'], edgecolor='black', linewidth=2)
    ax.add_patch(work_box)
    ax.text(15, 12.2, 'Work Directory', fontsize=13, fontweight='bold', ha='center')
    ax.text(15, 11.8, '• Partitions', fontsize=10, ha='center')
    ax.text(15, 11.5, '• Checkpoints', fontsize=10, ha='center')
    ax.text(15, 11.2, '• Event Logs', fontsize=10, ha='center')
    
    # Main Executor
    executor_box = FancyBboxPatch((2, 6.5), 14, 4, boxstyle="round,pad=0.1", 
                                 facecolor=colors['core'], edgecolor='black', linewidth=2)
    ax.add_patch(executor_box)
    ax.text(9, 10.2, 'EnhancedPartitionedRayExecutor', fontsize=16, fontweight='bold', ha='center')
    
    # Components within executor - Top row
    top_components = [
        ('DatasetBuilder', 4, 9.2, '• Load Dataset\n• Format Detection\n• Schema Inference'),
        ('Partitioning\nEngine', 9, 9.2, '• Split Dataset\n• Size Control\n• Metadata Generation'),
        ('EventLogger', 14, 9.2, '• Track All Events\n• Real-time Monitoring\n• Performance Metrics')
    ]
    
    for name, x, y, desc in top_components:
        comp_box = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1, boxstyle="round,pad=0.05", 
                                 facecolor=colors['component'], edgecolor='black', linewidth=1)
        ax.add_patch(comp_box)
        ax.text(x, y+0.2, name, fontsize=9, fontweight='bold', ha='center')
        lines = desc.split('\n')
        for i, line in enumerate(lines):
            ax.text(x, y-0.1-i*0.15, line, fontsize=7, ha='center')
    
    # Components within executor - Bottom row
    bottom_components = [
        ('Checkpoint\nManager', 4, 7.2, '• Save States\n• Load States\n• Cleanup'),
        ('Ray Cluster', 9, 7.2, '• Distribute\n• Parallel Execution\n• Fault Handling'),
        ('Result Merger', 14, 7.2, '• Combine Partitions\n• Validate Results\n• Export Dataset')
    ]
    
    for name, x, y, desc in bottom_components:
        comp_box = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1, boxstyle="round,pad=0.05", 
                                 facecolor=colors['component'], edgecolor='black', linewidth=1)
        ax.add_patch(comp_box)
        ax.text(x, y+0.2, name, fontsize=9, fontweight='bold', ha='center')
        lines = desc.split('\n')
        for i, line in enumerate(lines):
            ax.text(x, y-0.1-i*0.15, line, fontsize=7, ha='center')
    
    # Output Section
    output_boxes = [
        ('Output Dataset', 3, 4, '• Processed Data\n• Validated Results\n• Optimized Format'),
        ('Event Logs', 9, 4, '• JSONL Format\n• Rotated Logs\n• Compressed Storage'),
        ('Performance\nReport', 15, 4, '• Timing Analysis\n• Resource Usage\n• Bottleneck Detection')
    ]
    
    for name, x, y, desc in output_boxes:
        out_box = FancyBboxPatch((x-1.8, y-0.8), 3.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
        ax.add_patch(out_box)
        ax.text(x, y+0.3, name, fontsize=12, fontweight='bold', ha='center')
        lines = desc.split('\n')
        for i, line in enumerate(lines):
            ax.text(x, y-0.1-i*0.2, line, fontsize=9, ha='center')
    
    # Arrows - Input to Executor (centered connections)
    input_arrows = [
        ((3, 11), (4, 10.5)),    # Input to DatasetBuilder
        ((9, 11), (9, 10.5)),    # Config to Partitioning Engine
        ((15, 11), (14, 10.5))   # Work Dir to EventLogger
    ]
    
    for start, end in input_arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Arrows - Executor to Output (centered connections)
    output_arrows = [
        ((4, 6.5), (3, 4.8)),    # CheckpointManager to Output Dataset
        ((9, 6.5), (9, 4.8)),    # Ray Cluster to Event Logs
        ((14, 6.5), (15, 4.8))   # Result Merger to Performance Report
    ]
    
    for start, end in output_arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """Create the data flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 16)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'process': '#F3E5F5',
        'partition': '#E8F5E8',
        'worker': '#FFF3E0',
        'output': '#FCE4EC'
    }
    
    # Title
    ax.text(8, 11.2, 'Data Flow: Partitioning, Processing & Merging', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input Dataset
    input_box = FancyBboxPatch((6, 10), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(8, 10.4, 'Input Dataset (Large File)', fontsize=12, fontweight='bold', ha='center')
    
    # Dataset Load & Analysis
    load_box = FancyBboxPatch((6, 8.8), 4, 0.8, boxstyle="round,pad=0.1", 
                             facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(load_box)
    ax.text(8, 9.2, 'Dataset Load & Analysis', fontsize=12, fontweight='bold', ha='center')
    
    # Partitioning Engine
    partition_box = FancyBboxPatch((6, 7.6), 4, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(partition_box)
    ax.text(8, 8, 'Partitioning Engine', fontsize=12, fontweight='bold', ha='center')
    
    # Partitions
    partitions = [
        ('Partition 1\n(10K samples)', 2, 6.2),
        ('Partition 2\n(10K samples)', 8, 6.2),
        ('Partition N\n(10K samples)', 14, 6.2)
    ]
    
    for name, x, y in partitions:
        part_box = FancyBboxPatch((x-1.8, y-0.5), 3.6, 1, boxstyle="round,pad=0.1", 
                                 facecolor=colors['partition'], edgecolor='black', linewidth=2)
        ax.add_patch(part_box)
        ax.text(x, y+0.2, name.split('\n')[0], fontsize=10, fontweight='bold', ha='center')
        ax.text(x, y-0.1, name.split('\n')[1], fontsize=9, ha='center')
    
    # Ray Workers
    workers = [
        ('Ray Worker 1', 2, 4.2),
        ('Ray Worker 2', 8, 4.2),
        ('Ray Worker N', 14, 4.2)
    ]
    
    for name, x, y in workers:
        worker_box = FancyBboxPatch((x-1.8, y-0.5), 3.6, 1, boxstyle="round,pad=0.1", 
                                   facecolor=colors['worker'], edgecolor='black', linewidth=2)
        ax.add_patch(worker_box)
        ax.text(x, y+0.2, name, fontsize=10, fontweight='bold', ha='center')
        
        # Worker details
        details = ['• Load Partition', '• Apply Ops', '• Save Checkpt', '• Log Events']
        for i, detail in enumerate(details):
            ax.text(x, y-0.1-i*0.15, detail, fontsize=7, ha='center')
    
    # Processed Partitions
    processed = [
        ('Processed P1\n+ Checkpoints', 2, 2.2),
        ('Processed P2\n+ Checkpoints', 8, 2.2),
        ('Processed PN\n+ Checkpoints', 14, 2.2)
    ]
    
    for name, x, y in processed:
        proc_box = FancyBboxPatch((x-1.8, y-0.5), 3.6, 1, boxstyle="round,pad=0.1", 
                                 facecolor=colors['output'], edgecolor='black', linewidth=2)
        ax.add_patch(proc_box)
        ax.text(x, y+0.2, name.split('\n')[0], fontsize=10, fontweight='bold', ha='center')
        ax.text(x, y-0.1, name.split('\n')[1], fontsize=9, ha='center')
    
    # Result Merger - Moved further down for better spacing
    merger_box = FancyBboxPatch((6, 0.4), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(merger_box)
    ax.text(8, 0.8, 'Result Merger', fontsize=12, fontweight='bold', ha='center')
    
    # Final Dataset - Adjusted for new merger position
    final_box = FancyBboxPatch((6, -0.8), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(8, -0.4, 'Final Dataset + Event Logs + Performance', fontsize=11, fontweight='bold', ha='center')
    
    # Arrows - Vertical flow (perfectly centered)
    vertical_arrows = [
        ((8, 10), (8, 9.6)),     # Input to Load
        ((8, 8.8), (8, 8.4)),    # Load to Partition
        ((8, 7.6), (8, 6.7)),    # Partition to Partitions
        ((2, 6.2), (2, 4.7)),    # P1 to Worker 1
        ((8, 6.2), (8, 4.7)),    # P2 to Worker 2
        ((14, 6.2), (14, 4.7)),  # PN to Worker N
        ((2, 4.2), (2, 2.7)),    # Worker 1 to Processed P1
        ((8, 4.2), (8, 2.7)),    # Worker 2 to Processed P2
        ((14, 4.2), (14, 2.7)),  # Worker N to Processed PN
        ((8, 2.2), (8, 1.2)),    # Processed P2 to Merger (center)
        ((8, 0.4), (8, 0.0))     # Merger to Final (center)
    ]
    
    for start, end in vertical_arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=15, fc="black", linewidth=1.5)
        ax.add_patch(arrow)
    
    # Arrows - Diagonal to merger (precise connections to box edges)
    diagonal_arrows = [
        ((2, 2.2), (6.0, 0.8)),  # Processed P1 to Merger (left edge)
        ((14, 2.2), (10.0, 0.8)) # Processed PN to Merger (right edge)
    ]
    
    for start, end in diagonal_arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=15, fc="black", linewidth=1.5,
                              connectionstyle="arc3,rad=0.15")
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig

def create_event_logging_diagram():
    """Create the event logging system diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'source': '#E8F5E8',
        'logger': '#E3F2FD',
        'storage': '#FFF3E0',
        'analysis': '#F3E5F5'
    }
    
    # Title
    ax.text(9, 11.2, 'Event Logging System Architecture', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Event Sources
    sources_box = FancyBboxPatch((1, 8.5), 3.5, 2, boxstyle="round,pad=0.1", 
                                facecolor=colors['source'], edgecolor='black', linewidth=2)
    ax.add_patch(sources_box)
    ax.text(2.75, 10.2, 'Event Sources', fontsize=13, fontweight='bold', ha='center')
    
    sources = ['• Operations', '• Partitions', '• Checkpoints', '• System']
    for i, source in enumerate(sources):
        ax.text(2.75, 9.7-i*0.3, source, fontsize=11, ha='center')
    
    # Event Logger
    logger_box = FancyBboxPatch((6, 8.5), 3.5, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['logger'], edgecolor='black', linewidth=2)
    ax.add_patch(logger_box)
    ax.text(7.75, 10.2, 'Event Logger', fontsize=13, fontweight='bold', ha='center')
    
    logger_features = ['• Event Queue', '• Timestamp', '• Metadata', '• Filtering']
    for i, feature in enumerate(logger_features):
        ax.text(7.75, 9.7-i*0.3, feature, fontsize=11, ha='center')
    
    # Event Storage
    storage_box = FancyBboxPatch((11, 8.5), 3.5, 2, boxstyle="round,pad=0.1", 
                                facecolor=colors['storage'], edgecolor='black', linewidth=2)
    ax.add_patch(storage_box)
    ax.text(12.75, 10.2, 'Event Storage', fontsize=13, fontweight='bold', ha='center')
    
    storage_features = ['• Memory Buffer', '• File System', '• Compression', '• Rotation']
    for i, feature in enumerate(storage_features):
        ax.text(12.75, 9.7-i*0.3, feature, fontsize=11, ha='center')
    
    # Event Types
    ax.text(9, 7.5, 'Event Types', fontsize=13, fontweight='bold', ha='center')
    
    event_categories = [
        ('Processing Events', 3, 6.2, ['• START', '• COMPLETE', '• ERROR']),
        ('Partition Events', 9, 6.2, ['• START', '• COMPLETE', '• ERROR']),
        ('Operation Events', 15, 6.2, ['• START', '• COMPLETE', '• ERROR']),
        ('Checkpoint Events', 3, 4.5, ['• SAVE', '• LOAD', '• CLEANUP']),
        ('Performance Events', 9, 4.5, ['• METRIC', '• THROUGHPUT', '• RESOURCE']),
        ('System Events', 15, 4.5, ['• WARNING', '• INFO', '• DEBUG'])
    ]
    
    for name, x, y, events in event_categories:
        cat_box = FancyBboxPatch((x-1.5, y-0.7), 3, 1.4, boxstyle="round,pad=0.1", 
                                facecolor=colors['analysis'], edgecolor='black', linewidth=1)
        ax.add_patch(cat_box)
        ax.text(x, y+0.3, name, fontsize=10, fontweight='bold', ha='center')
        for i, event in enumerate(events):
            ax.text(x, y-0.1-i*0.2, event, fontsize=9, ha='center')
    
    # Event Analysis
    analysis_box = FancyBboxPatch((1, 1.5), 16, 2, boxstyle="round,pad=0.1", 
                                 facecolor=colors['analysis'], edgecolor='black', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(9, 3.2, 'Event Analysis & Monitoring', fontsize=13, fontweight='bold', ha='center')
    
    analysis_features = [
        ('Real-time Monitoring', 4, 2.5, ['• Live Events', '• Alerts', '• Dashboards']),
        ('Filtering & Querying', 9, 2.5, ['• By Type', '• By Time', '• By Partition']),
        ('Reporting', 14, 2.5, ['• Status Reports', '• Performance Analysis', '• Error Analysis'])
    ]
    
    for name, x, y, features in analysis_features:
        ax.text(x, y+0.4, name, fontsize=11, fontweight='bold', ha='center')
        for i, feature in enumerate(features):
            ax.text(x, y-0.1-i*0.2, feature, fontsize=9, ha='center')
    
    # Arrows - Horizontal connections (centered)
    arrows = [
        ((4.5, 9.5), (6, 9.5)),     # Sources to Logger
        ((9.5, 9.5), (11, 9.5)),    # Logger to Storage
        ((2.75, 8.5), (3, 6.9)),    # Sources to Event Types
        ((7.75, 8.5), (9, 6.9)),    # Logger to Event Types
        ((12.75, 8.5), (15, 6.9)),  # Storage to Event Types
        ((3, 4.5), (4, 3.5)),       # Event Types to Analysis
        ((9, 4.5), (9, 3.5)),       # Event Types to Analysis
        ((15, 4.5), (14, 3.5))      # Event Types to Analysis
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=15, fc="black", linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig

def create_fault_tolerance_diagram():
    """Create the fault tolerance and recovery diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'normal': '#E8F5E8',
        'failure': '#FFEBEE',
        'recovery': '#E3F2FD',
        'strategy': '#FFF3E0'
    }
    
    # Title
    ax.text(9, 11.2, 'Fault Tolerance & Recovery System', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Normal Processing
    normal_box = FancyBboxPatch((1, 8.5), 4, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['normal'], edgecolor='black', linewidth=2)
    ax.add_patch(normal_box)
    ax.text(3, 9.7, 'Normal Processing', fontsize=13, fontweight='bold', ha='center')
    
    normal_features = ['• Process Data', '• Save Checkpoints', '• Log Progress']
    for i, feature in enumerate(normal_features):
        ax.text(3, 9.2-i*0.3, feature, fontsize=11, ha='center')
    
    # Failure Detection
    failure_box = FancyBboxPatch((7, 8.5), 4, 2, boxstyle="round,pad=0.1", 
                                facecolor=colors['failure'], edgecolor='black', linewidth=2)
    ax.add_patch(failure_box)
    ax.text(9, 9.7, 'Failure Detection', fontsize=13, fontweight='bold', ha='center')
    
    failure_features = ['• Error Event', '• Log Error', '• Alert System']
    for i, feature in enumerate(failure_features):
        ax.text(9, 9.2-i*0.3, feature, fontsize=11, ha='center')
    
    # Recovery Process
    recovery_box = FancyBboxPatch((13, 8.5), 4, 2, boxstyle="round,pad=0.1", 
                                 facecolor=colors['recovery'], edgecolor='black', linewidth=2)
    ax.add_patch(recovery_box)
    ax.text(15, 9.7, 'Recovery Process', fontsize=13, fontweight='bold', ha='center')
    
    recovery_features = ['• Load Checkpoint', '• Retry Operation', '• Resume Processing']
    for i, feature in enumerate(recovery_features):
        ax.text(15, 9.2-i*0.3, feature, fontsize=11, ha='center')
    
    # Recovery Strategies
    ax.text(9, 7.2, 'Recovery Strategies', fontsize=13, fontweight='bold', ha='center')
    
    strategies = [
        ('Checkpoint\nRecovery', 4, 5.5, ['• Load Last Checkpoint', '• Resume from Last Op', '• Continue Processing']),
        ('Retry with\nBackoff', 9, 5.5, ['• Exponential Backoff', '• Max Retries', '• Error Logging']),
        ('Graceful\nDegradation', 14, 5.5, ['• Skip Failed Partition', '• Continue with Success', '• Report Partial Results'])
    ]
    
    for name, x, y, features in strategies:
        strat_box = FancyBboxPatch((x-1.8, y-0.9), 3.6, 1.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['strategy'], edgecolor='black', linewidth=2)
        ax.add_patch(strat_box)
        ax.text(x, y+0.4, name, fontsize=11, fontweight='bold', ha='center')
        for i, feature in enumerate(features):
            ax.text(x, y-0.1-i*0.25, feature, fontsize=9, ha='center')
    
    # Error Handling
    error_box = FancyBboxPatch((1, 1.5), 16, 2, boxstyle="round,pad=0.1", 
                              facecolor=colors['failure'], edgecolor='black', linewidth=2)
    ax.add_patch(error_box)
    ax.text(9, 3.2, 'Error Handling & Reporting', fontsize=13, fontweight='bold', ha='center')
    
    error_categories = [
        ('Error Types', 4, 2.5, ['• Network Errors', '• Memory Errors', '• Processing Errors', '• System Errors']),
        ('Error Logging', 9, 2.5, ['• Stack Trace', '• Context Info', '• Timestamp', '• Metadata']),
        ('Error Reporting', 14, 2.5, ['• Error Summary', '• Failed Partitions', '• Recovery Actions', '• Recommendations'])
    ]
    
    for name, x, y, features in error_categories:
        ax.text(x, y+0.4, name, fontsize=11, fontweight='bold', ha='center')
        for i, feature in enumerate(features):
            ax.text(x, y-0.1-i*0.25, feature, fontsize=9, ha='center')
    
    # Arrows - Horizontal connections (centered)
    arrows = [
        ((5, 9.5), (7, 9.5)),      # Normal to Failure
        ((11, 9.5), (13, 9.5)),    # Failure to Recovery
        ((3, 8.5), (4, 6.4)),      # Normal to Checkpoint Recovery
        ((9, 8.5), (9, 6.4)),      # Failure to Retry Backoff
        ((15, 8.5), (14, 6.4)),    # Recovery to Graceful Degradation
        ((4, 5.5), (4, 3.5)),      # Strategies to Error Types
        ((9, 5.5), (9, 3.5)),      # Strategies to Error Logging
        ((14, 5.5), (14, 3.5))     # Strategies to Error Reporting
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=15, fc="black", linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all architecture diagrams."""
    print("Generating Data-Juicer Architecture Diagrams...")
    
    # Create output directory
    output_dir = Path("architecture")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate diagrams
    diagrams = [
        ("system_architecture", create_system_architecture_diagram),
        ("data_flow", create_data_flow_diagram),
        ("event_logging", create_event_logging_diagram),
        ("fault_tolerance", create_fault_tolerance_diagram)
    ]
    
    for name, create_func in diagrams:
        print(f"Creating {name} diagram...")
        fig = create_func()
        
        # Save as PNG
        png_path = output_dir / f"{name}.png"
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {png_path}")
        
        # Save as PDF
        pdf_path = output_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
        print(f"  Saved: {pdf_path}")
        
        plt.close(fig)
    
    print(f"\nAll diagrams generated in: {output_dir}")
    print("\nGenerated files:")
    for name, _ in diagrams:
        print(f"  - {name}.png (high-resolution PNG)")
        print(f"  - {name}.pdf (vector PDF)")

if __name__ == "__main__":
    main() 