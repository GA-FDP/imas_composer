from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Dict
from enum import Enum
import yaml
from pathlib import Path

class RequirementStage(Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    COMPUTED = "computed"

@dataclass
class Requirement:
    mds_path: str
    shot: int
    treename: str = "ELECTRONS"
    
    def __hash__(self):
        return hash((self.mds_path, self.shot, self.treename))
    
    def as_key(self):
        return (self.mds_path, self.shot, self.treename)

@dataclass
class IDSEntrySpec:
    stage: RequirementStage
    static_requirements: list[Requirement] = field(default_factory=list)
    derive_requirements: Optional[Callable[[int, dict], list[Requirement]]] = None
    compose: Optional[Callable[[int, dict], Any]] = None
    depends_on: list[str] = field(default_factory=list)
    ids_path: Optional[str] = None
    docs_file: Optional[str] = None
    _docs_cache: Dict[str, Dict] = field(default_factory=dict, repr=False, compare=False)
    
    @property
    def documentation(self) -> Dict[str, Any]:
        """
        Lazy load documentation from YAML file.
        
        Returns documentation dict for this IDS entry, or empty dict if not found.
        """
        if not self.docs_file or not self.ids_path:
            return {}
        
        # Check cache first
        cache_key = self.docs_file
        if cache_key not in self._docs_cache:
            # Load YAML file
            docs_path = Path(__file__).parent / 'ids' / self.docs_file
            if docs_path.exists():
                with open(docs_path, 'r') as f:
                    self._docs_cache[cache_key] = yaml.safe_load(f)
            else:
                self._docs_cache[cache_key] = {}
        
        # Return entry-specific documentation
        all_docs = self._docs_cache[cache_key]
        return all_docs.get('entries', {}).get(self.ids_path, {})
    
    def get_summary(self) -> str:
        """Get short summary of this IDS entry"""
        return self.documentation.get('summary', 'No documentation available')
    
    def get_description(self) -> str:
        """Get detailed description of this IDS entry"""
        return self.documentation.get('description', '')
    
    def get_mds_paths(self) -> list[str]:
        """Get list of MDSplus paths used by this entry"""
        doc = self.documentation
        if 'mds_path' in doc:
            return [doc['mds_path']]
        elif 'mds_paths' in doc:
            return doc['mds_paths']
        return []
