from dataclasses import dataclass
from pipecat.frames.frames import Frame


@dataclass
class FlightSearchFrame(Frame):
    source_city: str
    destination_city: str
    depart_date: str
    return_date: str
