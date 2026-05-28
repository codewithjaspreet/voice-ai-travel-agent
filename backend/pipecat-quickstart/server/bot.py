#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Realtime Voice AI Travel Agent
Pipecat + OpenAI + Deepgram + Cartesia

Pipeline:
Speech-to-Text → LLM → Text-to-Speech

Run:
    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from processors.intent_processor import IntentProcessor

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
    RunnerArguments,
    DailyRunnerArguments,
    SmallWebRTCRunnerArguments,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService

from pipecat.transports.base_transport import (
    BaseTransport,
    TransportParams,
)

from pipecat.transports.daily.transport import (
    DailyTransport,
    DailyParams,
)

from pipecat.transports.smallwebrtc.connection import (
    SmallWebRTCConnection,
)

from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from processors.intent_processor import (
    IntentProcessor,
)

from processors.flight_processor import (
    FlightProcessor,
)

load_dotenv(override=True)


SYSTEM_PROMPT = """
You are a premium AI travel assistant
- Keep responses under 2 sentences highly concise
- Speak naturally and conversationally
- Be concise and energetic
- Ask only one follow-up question at a time
"""


async def run_bot(transport: BaseTransport):
    """Main bot logic."""

    logger.info("Starting Voice Travel Agent")
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            top_p=0.8,
            max_completion_tokens=60,
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
            confidence=0.7,  # Minimum confidence for voice detection
            start_secs=0.2,  # Time to wait before confirming speech start
            stop_secs=0.2,  # Time to wait before confirming speech stop
            min_volume=0.6,  # Minimum volume threshold
        )
    )
    context = LLMContext()
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

    intent_processor = IntentProcessor()

    flight_processor = FlightProcessor(context=context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            intent_processor,
            flight_processor,
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
                "content": "Introduce yourself as a premium AI travel assistant in 1 sentence.",
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
