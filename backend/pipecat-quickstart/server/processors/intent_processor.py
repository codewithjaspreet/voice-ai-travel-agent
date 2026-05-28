from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

from frames.flight_frames import FlightSearchFrame


class IntentProcessor(FrameProcessor):

    async def process_frame(
        self,
        frame,
        direction,
    ):

        await super().process_frame(
            frame,
            direction,
        )

        if isinstance(frame, TranscriptionFrame):

            text = frame.text.lower()

            print("\nUSER:", text)

            if "flight" in text:

                # TEMPORARY HARDCODED EXTRACTION
                # We’ll replace with semantic extraction later

                flight_frame = FlightSearchFrame(
                    source_city="Lucknow",
                    destination_city="Bengaluru",
                    depart_date="2026-05-26",
                    return_date="2026-06-05",
                )

                print("FLIGHT INTENT DETECTED")

                await self.push_frame(flight_frame)

                return

        await self.push_frame(
            frame,
            direction,
        )
