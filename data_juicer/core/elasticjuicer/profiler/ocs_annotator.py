"""
Operator Cost Signature (OCS) Annotator

Provides semantic annotations for operators based on Alpa's operator modeling:
- Memory Locality: Device preference (CPU-Strong, GPU-Strong, Balanced)
- Transfer Cost: Data movement overhead (Low, Medium, High)
- Failure Cost: Recovery cost from OOM (Low, Medium, High)
- State-free: Whether operator can be safely retried

Inspired by:
- Alpa's operator cost modeling
- ExoFlow's failure cost analysis
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json


class MemoryLocality(Enum):
    """Device preference for operator execution"""

    CPU_STRONG = "cpu_strong"  # Strongly prefers CPU (e.g., regex, text filters)
    GPU_STRONG = "gpu_strong"  # Strongly prefers GPU (e.g., VLM, video decoding)
    BALANCED = "balanced"  # Can run efficiently on either
    MIXED = "mixed"  # Benefits from CPU-GPU cooperation


class TransferCost(Enum):
    """Data movement overhead"""

    LOW = "low"  # < 1MB per sample (text, metadata)
    MEDIUM = "medium"  # 1-100MB per sample (images)
    HIGH = "high"  # > 100MB per sample (videos, large models)


class FailureCost(Enum):
    """Recovery cost from failure"""

    LOW = "low"  # Fast retry, no state loss
    MEDIUM = "medium"  # Moderate retry cost
    HIGH = "high"  # Expensive recomputation (e.g., long video processing)


@dataclass
class OpCostSignature:
    """
    Cost signature for an operator.

    This is the core of OCS profiling - semantic annotations that guide scheduling.
    """

    op_name: str
    op_type: str  # filter, mapper, deduplicator, etc.

    # Core OCS attributes (based on Alpa)
    memory_locality: MemoryLocality = MemoryLocality.BALANCED
    transfer_cost: TransferCost = TransferCost.MEDIUM
    failure_cost: FailureCost = FailureCost.MEDIUM

    # State properties (based on ExoFlow)
    state_free: bool = True  # Can be safely retried without side effects
    deterministic: bool = True  # Same input always produces same output

    # Resource preferences
    preferred_batch_size: Optional[int] = None
    min_memory_mb: Optional[float] = None
    max_memory_mb: Optional[float] = None

    # Modality tags
    handles_text: bool = False
    handles_image: bool = False
    handles_video: bool = False
    handles_audio: bool = False

    # Additional metadata
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "op_name": self.op_name,
            "op_type": self.op_type,
            "memory_locality": self.memory_locality.value,
            "transfer_cost": self.transfer_cost.value,
            "failure_cost": self.failure_cost.value,
            "state_free": self.state_free,
            "deterministic": self.deterministic,
            "preferred_batch_size": self.preferred_batch_size,
            "min_memory_mb": self.min_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "handles_text": self.handles_text,
            "handles_image": self.handles_image,
            "handles_video": self.handles_video,
            "handles_audio": self.handles_audio,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OpCostSignature":
        """Create from dictionary"""
        return cls(
            op_name=data["op_name"],
            op_type=data["op_type"],
            memory_locality=MemoryLocality(data.get("memory_locality", "balanced")),
            transfer_cost=TransferCost(data.get("transfer_cost", "medium")),
            failure_cost=FailureCost(data.get("failure_cost", "medium")),
            state_free=data.get("state_free", True),
            deterministic=data.get("deterministic", True),
            preferred_batch_size=data.get("preferred_batch_size"),
            min_memory_mb=data.get("min_memory_mb"),
            max_memory_mb=data.get("max_memory_mb"),
            handles_text=data.get("handles_text", False),
            handles_image=data.get("handles_image", False),
            handles_video=data.get("handles_video", False),
            handles_audio=data.get("handles_audio", False),
            notes=data.get("notes", ""),
        )


class OCSAnnotator:
    """
    Annotates operators with cost signatures.

    Provides pre-defined annotations for common Data-Juicer operators
    and supports custom annotations.
    """

    def __init__(self):
        self.signatures: Dict[str, OpCostSignature] = {}
        self._load_default_signatures()

    def _load_default_signatures(self):
        """Load default OCS signatures for common Data-Juicer operators"""

        # Text Filters - CPU Strong, Low Transfer, Low Failure
        text_filter_ops = [
            "TextLengthFilter",
            "AlphanumericFilter",
            "CharacterRepetitionFilter",
            "WordRepetitionFilter",
            "SpecialCharactersFilter",
        ]
        for op in text_filter_ops:
            self.signatures[op] = OpCostSignature(
                op_name=op,
                op_type="filter",
                memory_locality=MemoryLocality.CPU_STRONG,
                transfer_cost=TransferCost.LOW,
                failure_cost=FailureCost.LOW,
                state_free=True,
                deterministic=True,
                handles_text=True,
                notes="Lightweight text filter, CPU-bound",
            )

        # Image Operations - GPU Preferred, Medium Transfer
        image_ops = [
            "ImageFaceRatioFilter",
            "ImageAestheticFilter",
            "ImageNSFWFilter",
        ]
        for op in image_ops:
            self.signatures[op] = OpCostSignature(
                op_name=op,
                op_type="filter",
                memory_locality=MemoryLocality.GPU_STRONG,
                transfer_cost=TransferCost.MEDIUM,
                failure_cost=FailureCost.MEDIUM,
                state_free=True,
                deterministic=True,
                handles_image=True,
                notes="Image model inference, GPU-accelerated",
            )

        # Video Operations - GPU Strong, High Transfer, High Failure
        video_ops = [
            "VideoDecoder",
            "VideoCaptioning",
            "VideoActionRecognition",
        ]
        for op in video_ops:
            self.signatures[op] = OpCostSignature(
                op_name=op,
                op_type="mapper",
                memory_locality=MemoryLocality.GPU_STRONG,
                transfer_cost=TransferCost.HIGH,
                failure_cost=FailureCost.HIGH,
                state_free=True,
                deterministic=True,
                handles_video=True,
                notes="Heavy video processing, high memory requirement",
            )

        # Deduplicators - Mixed locality, variable cost
        self.signatures["DocumentDeduplicator"] = OpCostSignature(
            op_name="DocumentDeduplicator",
            op_type="deduplicator",
            memory_locality=MemoryLocality.CPU_STRONG,
            transfer_cost=TransferCost.LOW,
            failure_cost=FailureCost.HIGH,
            state_free=False,  # Maintains hash index
            deterministic=True,
            handles_text=True,
            notes="Hash-based dedup, stateful index",
        )

        self.signatures["ImageDeduplicator"] = OpCostSignature(
            op_name="ImageDeduplicator",
            op_type="deduplicator",
            memory_locality=MemoryLocality.MIXED,
            transfer_cost=TransferCost.MEDIUM,
            failure_cost=FailureCost.HIGH,
            state_free=False,
            deterministic=True,
            handles_image=True,
            notes="Image hash computation, benefits from GPU",
        )

    def annotate(self, op_name: str, signature: OpCostSignature):
        """Add or update OCS signature for an operator"""
        self.signatures[op_name] = signature

    def get_signature(self, op_name: str) -> Optional[OpCostSignature]:
        """Get OCS signature for an operator"""
        return self.signatures.get(op_name)

    def get_all_signatures(self) -> Dict[str, OpCostSignature]:
        """Get all registered signatures"""
        return dict(self.signatures)

    def export_to_file(self, filepath: str):
        """Export signatures to JSON file"""
        data = {name: sig.to_dict() for name, sig in self.signatures.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_file(self, filepath: str):
        """Import signatures from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        for name, sig_dict in data.items():
            self.signatures[name] = OpCostSignature.from_dict(sig_dict)

    def infer_signature(self, op_name: str, op_type: str, **hints) -> OpCostSignature:
        """
        Infer OCS signature from operator name and hints.

        This provides a best-effort annotation for unknown operators.
        """
        # Default values
        locality = MemoryLocality.BALANCED
        transfer = TransferCost.MEDIUM
        failure = FailureCost.MEDIUM

        # Infer from name patterns
        op_lower = op_name.lower()

        if "video" in op_lower:
            locality = MemoryLocality.GPU_STRONG
            transfer = TransferCost.HIGH
            failure = FailureCost.HIGH
        elif "image" in op_lower:
            locality = MemoryLocality.GPU_STRONG
            transfer = TransferCost.MEDIUM
        elif "text" in op_lower or "word" in op_lower or "character" in op_lower:
            locality = MemoryLocality.CPU_STRONG
            transfer = TransferCost.LOW
            failure = FailureCost.LOW

        # Apply hints if provided
        if "accelerator" in hints and hints["accelerator"] == "cuda":
            locality = MemoryLocality.GPU_STRONG

        return OpCostSignature(
            op_name=op_name,
            op_type=op_type,
            memory_locality=locality,
            transfer_cost=transfer,
            failure_cost=failure,
            handles_text="text" in op_lower,
            handles_image="image" in op_lower,
            handles_video="video" in op_lower,
            handles_audio="audio" in op_lower,
            notes="Auto-inferred signature",
        )
