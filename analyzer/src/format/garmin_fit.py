from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from fitparse import FitFile
from format.models import Workout, Device, Sensor

class FitFileParser:

    data_frame: pd.DataFrame
    laps : list[dict[str,pd.Timestamp]]
    sets : list[dict[str,pd.Timestamp]]
    workout: Workout

    def __init__(self,file_path:Path):
        self.data_frame, self.laps, self.sets, workout = self._parse_fitfile(fit_path=str(file_path))
        self.workout = self._extract_workout(workout)

    
    def _semicircles_to_degrees(self,value):
        return value * 180.0 / (2**31)

    def _sanitize_record_value(self, name: str, value):
        """Eventuell konvertering av FIT-felt til mer menneskevennlig format."""
        if value is None:
            return None

        # Eksempel: posisjoner kommer i 'semicircles'; konverter til grader hvis du trenger det.
        if name in ("position_lat", "position_long", "enhanced_position_lat", "enhanced_position_long"):
            return self._semicircles_to_degrees(value)

        # Tidspunkt i FIT er datetime – det er greit å beholde slik:
        if isinstance(value, datetime):
            return value

        # FIT har gjerne Decimal-objekter, ints, floats, bools, str...
        # La dem være i fred – pandas håndterer dette fint.
        return value

    def _parse_and_merge_device(self, device: dict[str,object] | None, message):
        manufacturer = message.get_value("manufacturer")
        if manufacturer is None:
            pass
        elif manufacturer == "garmin" or  manufacturer == "4iiiis":
            device = device or {"sensors":set(),"creator":None}
            index = message.get_value("device_index") 
            product = message.get_value("garmin_product") 
            sensor = message.get_value("local_device_type")  or message.get_value("garmin_product") or message.get_value("antplus_device_type")
            if index == "creator":
                device[index] = product
                device["manufacturer"]=manufacturer
            elif sensor:
                device["sensors"].add((manufacturer,sensor))
        elif manufacturer == "zwift":
            device = device or {"sensors":set(),"creator":None}
            index = message.get_value("device_index") 
            if index == "creator":
                device[index] = "zwift"
                device["manufacturer"]=manufacturer

        else: # print debug device info
            data = {}
            for field in message:
                data[field.name] = field.value
            print(data)
        return device


    def _parse_fitfile(self, fit_path: str) -> tuple[pd.DataFrame, list[dict[str, pd.Timestamp]], list[dict[str, pd.Timestamp]], list[dict[str, object]]]:
        """
        Leser kraft/puls/kadens (resamplet til 1 Hz) og lap-informasjon fra FIT-filen.
        Returnerer en DataFrame og en liste med lap-intervaller {"start": ts, "end": ts}.
        """
        fitfile = FitFile(fit_path)

        record_rows: list[dict[str, object]] = []
        sets: list[dict[str, object]] = []
        laps: list[dict[str, pd.Timestamp]] = []
        workout_info : list[dict[str, object]] = []
        device : dict[str, object] = None


        for message in fitfile.get_messages():
            if message.name == "record":
                data = message.get_values()  # dict med ALLE standardfelt + de fleste developer fields
                timestamp = data.pop("timestamp", None)
                if timestamp is None:
                    continue

                row = {"timestamp": pd.to_datetime(timestamp)}
                for name, value in data.items():
                    row[name] = self._sanitize_record_value(name, value)
                record_rows.append(row)
            elif message.name == "set":
                data = message.get_values()
                timestamp = data.pop("timestamp", None)
                if timestamp is None:
                    continue

                row = {"timestamp": pd.to_datetime(timestamp)}
                for name, value in data.items():
                    row[name] = self._sanitize_record_value(name, value)
                sets.append(row)
            elif message.name == "lap":
                data = {d.name: d.value for d in message}
                start = data.get("start_time")
                duration = data.get("total_elapsed_time")
                end = data.get("timestamp") or data.get("end_time")
                if not start:
                    continue
                start_ts = pd.to_datetime(start)
                if duration is not None:
                    end_ts = start_ts + pd.to_timedelta(duration, unit="s")
                elif end is not None:
                    end_ts = pd.to_datetime(end)
                else:
                    continue
                lap_name = data.get("wkt_step_name") or data.get("lap_name") or data.get("swim_name")
                intensity = data.get("intensity")

                laps.append(
                    {
                        "start": start_ts,
                        "end": end_ts,
                        "label": lap_name,
                        "intensity": intensity,
                    }
                )
            elif message.name == "session":
                workout_info.append({
                    "sport": message.get_value("sport"),
                    "sub_sport": message.get_value("sub_sport"),
                    "virtual_activity": message.get_value("virtual_activity"),
                    "indoor": message.get_value("indoor") or message.get_value("trainer"),
                    "workout_name": message.get_value("workout_name"),
                    "pool_length": message.get_value("pool_length"),
                    "total_distance": message.get_value("total_distance"),
                    "start_time": message.get_value("start_time")
                })
            elif message.name == "activity":
                workout_info.append({
                    "type": message.get_value("type"),
                    "event": message.get_value("event"),
                    "event_type": message.get_value("event_type"),
                    "num_sessions": message.get_value("num_sessions"),
                    "product": message.get_value("product")
                })
            elif message.name == "workout":
                workout_info.append({
                    "sport": message.get_value("sport"),
                    "workout_name": message.get_value("wkt_name") or message.get_value("workout_name"),
                    "capabilities": message.get_value("capabilities"),
                })
            elif message.name == "file_id":
                
                workout_info.append({
                    "manufacturer" : message.get_value("manufacturer"),
                    "product" : message.get_value("product")
                })
            elif message.name == "device_info":
                device = self._parse_and_merge_device(device,message)

        if device:
            workout_info.append(device)

        if not record_rows:
            raise ValueError("Fant ingen record-data i FIT-filen.")

        df = pd.DataFrame(record_rows).set_index("timestamp").sort_index()
        print(df.columns.tolist())

        # Resample til 1 Hz ved behov
        deltas = df.index.to_series().diff().dropna().dt.total_seconds()
        if not deltas.empty and not np.allclose(deltas, 1.0, atol=0.5):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            other_cols = df.columns.difference(numeric_cols)

            # 1) Numeriske kolonner: resample + mean + interpolate
            df_numeric = (
                df[numeric_cols]
                .resample("1s")
                .mean()
                .infer_objects(copy=False)
                .interpolate(method="time", limit_direction="both")
            )

            # 2) Andre kolonner: velg en passende aggregering (first/ffill osv.)
            df_other = (
                df[other_cols]
                .resample("1s")
                .ffill()  # evt. .ffill() hvis du vil holde siste verdi
            )

            # 3) Kombiner igjen
            df = pd.concat([df_numeric, df_other], axis=1).sort_index()

        laps.sort(key=lambda lap: lap["start"])
        return df, laps, sets, workout_info

    def _extract_workout(self, wo : list[dict[str,object]]) -> Workout:
        sport = None
        sub_sport = None
        workout_name = None
        total_distance = None
        creator = None
        manufacturer = None
        sensors = None
        product = None
        prod_man = None
        start_time = None
        for row in wo:
            if row.get("sport"):
                sport = row.get("sport")
                sub_sport =  row.get("sub_sport")
                workout_name = row.get("workout_name")
                total_distance=row.get("total_distance")
                start_time = row.get("start_time")
            if row.get("product"):
                product = row.get("manufacturer")
                prod_man =row.get("product")
            if row.get("creator"):
                creator = row.get("creator")
                manufacturer = row.get("manufacturer")
                sensors = row.get("sensors")

        if total_distance is None:
            total_distance = self.data_frame["distance"].dropna().max() if "distance" in self.data_frame.columns else 0.0

        sensor_list =[]
        for m,s in sensors:
            sensor_list.append(Sensor(name=str(s), manufacturer=str(m)))

        device = Device()
        device._sensors=sensor_list
        device._product=creator or product
        device._manufacturer= manufacturer or prod_man

        workout = Workout(
            sport=sport,
            sub_sport=sub_sport,
            name=workout_name,
            device=device,
            start_time=start_time
        )
        
        workout._distance = float(total_distance) if total_distance is not None else 0.0
        return workout
