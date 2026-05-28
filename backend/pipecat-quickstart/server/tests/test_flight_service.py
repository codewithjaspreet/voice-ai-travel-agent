import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import FlightService


async def main():

    service = FlightService()

    flights = await service.get_best_flights(
        source_city="Lucknow",
        destination_city="Bengaluru",
        depart_date="2026-05-26",
        return_date="2026-06-05",
    )

    print("\nFINAL FLIGHTS:\n")

    print(
        json.dumps(
            flights,
            indent=4,
        )
    )


asyncio.run(main())
