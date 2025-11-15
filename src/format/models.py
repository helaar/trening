from typing import Literal
from pydantic import BaseModel, PrivateAttr, computed_field


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
    start_time: str | None = None
    sport: str
    sub_sport: str | None = None
    
    device: Device
    _distance: float = PrivateAttr(0.0)

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
                if "strength" in self.sub_sport:
                    return "strength"
                else:
                    return "other"
            case _:
                return "other"


