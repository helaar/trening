from typing import Literal
from datetime import datetime
from pydantic import BaseModel, PrivateAttr, computed_field, field_validator


class Sensor(BaseModel):
    name: str
    manufacturer: str

class Device(BaseModel):
    _product: str | None = PrivateAttr(None)
    _manufacturer: str | None = PrivateAttr(None)
    _sensors: list[Sensor] = PrivateAttr(default_factory=list)

    @computed_field
    def device(self) -> str:
        prod = self._product if self._product != self._manufacturer else ""
        sensor_str = "med "+self.sensors if self.sensors is not None else ""
        return  f"{self._manufacturer} {prod} {sensor_str}"


    @property
    def sensors(self) -> str | None:
        if not self._sensors:
            return None
        else:
            ret = []
            for s in self._sensors:
                if s.manufacturer is None or s.manufacturer == self._manufacturer:
                    ret.append(s.name)
                else:
                    ret.append(f"{s.manufacturer} {s.name}")
            return ", ".join(ret)


class Workout(BaseModel):
    name: str | None = None
    start_time: datetime | None = None
    sport: str
    sub_sport: str | None = None
    
    device: Device
    _distance: float = PrivateAttr(0.0)
    
    @field_validator('start_time', mode='before')
    @classmethod
    def parse_start_time(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try to parse various datetime string formats
            try:
                # Handle ISO format with Z timezone
                if v.endswith('Z'):
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                # Handle standard ISO format
                return datetime.fromisoformat(v)
            except ValueError:
                try:
                    # Try common format
                    return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Try another common format
                        return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S')
                    except ValueError:
                        # If all parsing fails, return None
                        return None
        # For any other type, return None
        return None
    

    @computed_field
    def distance(self) -> str:
        return f"{self.distance_km:.2f} km"
    
    @property
    def distance_km(self) -> float:
        return self._distance / 1000.0

    @property
    def category(self) -> Literal["cycling","running","strength","other"] :
        match self.sport:
            case "cycling" | "running":
                return self.sport
            
            case "training":
                if self.sub_sport and "strength" in self.sub_sport:
                    return "strength"
                else:
                    return "other"
            case _:
                return "other"


