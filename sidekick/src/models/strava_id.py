"""Type for Strava activity and athlete identifiers.

Strava switches these IDs to 64-bit values after 2027-06-01. Python ints and
MongoDB Int64 hold them exactly, but JavaScript numbers silently lose precision
above 2**53, so IDs are serialized as strings in JSON (API responses, LLM
prompts). Python-mode dumps used for MongoDB writes keep them as ints, and
validation accepts either ints or numeric strings, so existing documents and
string IDs sent by the frontend both parse.
"""

from typing import Annotated

from pydantic import PlainSerializer

StravaId = Annotated[int, PlainSerializer(str, return_type=str, when_used="json")]
