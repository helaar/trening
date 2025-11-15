from pydantic import BaseModel

class ModelFormatter:
    def format(self,log, obj:BaseModel) -> None:
        self._format_dict(log, obj.model_dump(exclude_none=True, exclude_computed_fields=False))


    def _format_dict(self,log, obj_dict:dict[str, object]) -> None:
        for key, value in obj_dict.items():
            label = key.removeprefix("_").replace("_", " ").capitalize()

            if isinstance(value, BaseModel):
                self.format(log, value)
            elif isinstance(value, dict):
                self._format_dict(log,value)
            else:
                log(f"{label}: {value}")

