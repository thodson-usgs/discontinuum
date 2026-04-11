"""Base provider types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


@dataclass
class MetaData:
    id: str
    name: str
    latitude: float
    longitude: float


@dataclass
class USGSParameter:
    """USGS parameter definition with unit conversion."""
    pcode: str
    standard_name: str
    long_name: Optional[str] = ""
    units: Optional[str] = ""
    conversion: Optional[float] = 1.0

    @property
    def name(self):
        """Alias for standard_name."""
        return self.standard_name
