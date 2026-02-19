from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BaseConfig:
    project_root: Path = field(default_factory=lambda: Path("."))
    data_root: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    features_dir: Path = field(default_factory=lambda: Path("./outputs/features"))
    logs_dir: Path = field(default_factory=lambda: Path("./outputs/logs"))
    models_dir: Path = field(default_factory=lambda: Path("./outputs/models"))
    results_dir: Path = field(default_factory=lambda: Path("./outputs/results"))
    device: str = "cuda"
    seed: int = 42
    
    disease_labels: Dict[str, str] = field(default_factory=lambda: {
        "bing_du_xing_fei_yan": "病毒性肺炎",
        "正常胸部CT图像": "正常",
        "PJP": "PJP肺炎",
        "细菌性": "细菌性肺炎",
    })
    
    label_dir_map: Dict[str, str] = field(default_factory=lambda: {
        "病毒性肺炎": "bing_du_xing_fei_yan",
        "正常": "正常胸部CT图像",
        "PJP肺炎": "PJP",
        "细菌性肺炎": "细菌性",
    })
