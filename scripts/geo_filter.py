from typing import Any, Dict, List, Tuple
import numpy as np


class BoundingBox:
    def __init__(self, name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
        self.name = name
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)

        self._validate()

    def _validate(self):
        if not (-90.0 <= self.lat_min <= self.lat_max <= 90.0):
            raise ValueError(f"Invalid latitude range for {self.name}: [{self.lat_min}, {self.lat_max}]")
        if not (-180.0 <= self.lon_min <= self.lon_max <= 180.0):
            raise ValueError(f"Invalid longitude range for {self.name}: [{self.lon_min}, {self.lon_max}]")

    def contains_point(self, lat: float, lon: float) -> bool:
        """
        Check if a single coordinate is within the bounding box.
        """
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)

    def mask_arrays(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Produce a boolean numpy mask for coordinate arrays.
        """
        # Validate coordinate bounds
        valid_coords = (lats >= -90.0) & (lats <= 90.0) & (lons >= -180.0) & (lons <= 180.0)
        in_box = (
            (lats >= self.lat_min)
            & (lats <= self.lat_max)
            & (lons >= self.lon_min)
            & (lons <= self.lon_max)
        )
        return valid_coords & in_box

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


def get_bounding_boxes_from_config(config: Dict[str, Any]) -> Tuple[BoundingBox, BoundingBox]:
    """
    Parse study_region (Chandipur) and optional regional_box from config.
    """
    sr = config.get("study_region", {})
    chandipur_box = BoundingBox(
        name=sr.get("name", "chandipur"),
        lat_min=sr.get("lat_min", 21.25),
        lat_max=sr.get("lat_max", 21.50),
        lon_min=sr.get("lon_min", 87.00),
        lon_max=sr.get("lon_max", 87.25),
    )

    rb = config.get("regional_box", {})
    regional_box = BoundingBox(
        name=rb.get("name", "odisha_regional"),
        lat_min=rb.get("lat_min", 17.50),
        lat_max=rb.get("lat_max", 23.50),
        lon_min=rb.get("lon_min", 81.00),
        lon_max=rb.get("lon_max", 89.00),
    )

    return chandipur_box, regional_box
