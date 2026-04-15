"""Base provider types."""

from __future__ import annotations

from dataclasses import dataclass


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
    long_name: str | None = ""
    units: str | None = ""
    conversion: float | None = 1.0

    @property
    def name(self):
        """Alias for standard_name."""
        return self.standard_name
