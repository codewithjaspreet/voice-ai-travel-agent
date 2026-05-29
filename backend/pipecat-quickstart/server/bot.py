#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Realtime Voice AI Travel Agent
Pipecat + Groq + Deepgram + Cartesia

Pipeline:
Speech-to-Text → LLM (with tools) → Text-to-Speech

The LLM drives all tool calls. No keyword matching or custom processors
sit between STT and the LLM — the model decides when and how to call tools.
"""

import json
import os
from datetime import date

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import (
    DailyRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from services import FlightService

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

FLIGHT_SEARCH_TOOL = FunctionSchema(
    name="search_flights",
    description=(
        "Search for available flights between two cities. "
        "Call this as soon as you have origin, destination, and departure date. "
        "Return date is optional — omit it for one-way trips."
    ),
    properties={
        "origin": {
            "type": "string",
            "description": "Departure city name (e.g. 'London', 'New York')",
        },
        "destination": {
            "type": "string",
            "description": "Destination city name (e.g. 'Tokyo', 'Paris')",
        },
        "depart_date": {
            "type": "string",
            "description": "Departure date in YYYY-MM-DD format",
        },
        "return_date": {
            "type": "string",
            "description": "Return date in YYYY-MM-DD format. Omit for one-way trips.",
        },
        "adults": {
            "type": "integer",
            "description": "Number of adult passengers. Defaults to 1.",
        },
        "cabin_class": {
            "type": "string",
            "enum": ["ECONOMY", "BUSINESS", "FIRST"],
            "description": "Cabin class. Defaults to ECONOMY.",
        },
    },
    required=["origin", "destination", "depart_date"],
)

TOOLS = ToolsSchema(standard_tools=[FLIGHT_SEARCH_TOOL])

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    today = date.today().strftime("%B %d, %Y")
    return f"""You are a premium AI travel assistant. Help users find flights and plan trips.

VOICE RULES — follow strictly:
- Every response must be 1-2 sentences maximum. Never longer.
- Speak naturally, as if talking to a friend.
- Ask only ONE follow-up question per turn.
- Never repeat information the user already gave you.
- Never use bullet points or lists — this is voice-only.

SLOT COLLECTION — gather in this order, one question at a time:
1. Origin city (where are they flying from?)
2. Destination city (where are they going?)
3. Departure date (when do they leave?)
4. Return date — or confirm it's one-way
5. Number of passengers (default 1 if not mentioned)
6. Cabin class (default economy if not mentioned)

TOOL USE:
- Call search_flights as soon as you have origin, destination, and departure date.
- Do not ask for optional fields before searching — search first, offer details after.
- If search returns an error, apologize briefly and ask if they'd like to try different dates.

Today's date: {today}"""


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport):
    """Main bot logic."""

    logger.info("Starting Voice Travel Agent")

    flight_service = FlightService()

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            top_p=0.8,
            max_completion_tokens=150,
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv(
                "CARTESIA_VOICE_ID",
                "71a7ad14-091c-4e8e-a314-022ece01c121",
            ),
        ),
    )

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.4,
            min_volume=0.5,
        )
    )

    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt()}],
        tools=TOOLS,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            vad_analyzer=vad_analyzer,
        ),
    )

    # -----------------------------------------------------------------------
    # Tool handler — called by Pipecat when the LLM emits a function call
    # -----------------------------------------------------------------------

    async def search_flights_handler(params: FunctionCallParams):
        args = params.arguments
        logger.info(
            f"Tool: search_flights | "
            f"{args.get('origin')} → {args.get('destination')} | "
            f"depart={args.get('depart_date')} return={args.get('return_date')}"
        )
        try:
            result = await flight_service.get_best_flights(
                source_city=args["origin"],
                destination_city=args["destination"],
                depart_date=args["depart_date"],
                return_date=args.get("return_date"),
                adults=int(args.get("adults", 1)),
                cabin_class=args.get("cabin_class", "ECONOMY"),
            )
            await params.result_callback(json.dumps(result))
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            await params.result_callback(
                json.dumps({"error": "search_failed", "message": str(e)})
            )

    llm.register_function("search_flights", search_flights_handler)

    # -----------------------------------------------------------------------
    # Pipeline — lean: STT → LLM (with tools) → TTS
    # -----------------------------------------------------------------------

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready")
        context.add_message(
            {
                "role": "user",
                "content": "You are Ami,Introduce yourself as a premium AI travel assistant in 1 sentence.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport = None

    match runner_args:

        case DailyRunnerArguments():
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "Voice Travel Agent",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )

        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )

        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
