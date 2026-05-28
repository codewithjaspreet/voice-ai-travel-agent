import json

from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.frame_processor import (
    FrameProcessor,
)

from frames.flight_frames import FlightSearchFrame

from services import FlightService


class FlightProcessor(FrameProcessor):

    def __init__(self, context):

        super().__init__()

        self.context = context

        self.flight_service = FlightService()

    async def process_frame(
        self,
        frame,
        direction,
    ):

        await super().process_frame(
            frame,
            direction,
        )

        if isinstance(frame, FlightSearchFrame):

            print("RUNNING FLIGHT SEARCH")

            flights = await self.flight_service.get_best_flights(
                source_city=frame.source_city,
                destination_city=frame.destination_city,
                depart_date=frame.depart_date,
                return_date=frame.return_date,
            )

            print("\nFLIGHT RESULTS:")
            print(
                json.dumps(
                    flights,
                    indent=4,
                )
            )

            # Inject travel intelligence
            # into LLM context

            self.context.add_message(
                {
                    "role": "system",
                    "content": f"""
                    Flight search results:

                    {json.dumps(flights)}

                    Respond like a premium AI travel assistant.

                    Rules:
                    - Be conversational
                    - Mention best option first
                    - Keep response short
                    - Avoid bullet points
                    """,
                }
            )

            # Trigger LLM response

            await self.push_frame(
                LLMRunFrame(),
                direction,
            )

            return

        await self.push_frame(
            frame,
            direction,
        )
