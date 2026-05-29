import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class FlightService:

    BASE_URL = "https://booking-com18.p.rapidapi.com"

    def __init__(self):
        self.headers = {
            "x-rapidapi-host": "booking-com18.p.rapidapi.com",
            "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
        }
        self._airport_cache: dict[str, str] = {}

    # =========================================================
    # Public API
    # =========================================================

    async def get_best_flights(
        self,
        source_city: str,
        destination_city: str,
        depart_date: str,
        return_date: str | None = None,
        adults: int = 1,
        cabin_class: str = "ECONOMY",
    ):
        raw_response = await self.search_flights(
            source_city=source_city,
            destination_city=destination_city,
            depart_date=depart_date,
            return_date=return_date,
            adults=adults,
            cabin_class=cabin_class,
        )

        return self.normalize_flights(
            api_response=raw_response,
            source_city=source_city,
            destination_city=destination_city,
        )

    # =========================================================
    # Flight Search
    # =========================================================

    async def search_flights(
        self,
        source_city: str,
        destination_city: str,
        depart_date: str,
        return_date: str | None = None,
        adults: int = 1,
        cabin_class: str = "ECONOMY",
    ):
        source_airport = await self.get_airport_code(source_city)
        destination_airport = await self.get_airport_code(destination_city)

        logger.info(
            f"Searching flights: "
            f"{source_city} ({source_airport}) → {destination_city} ({destination_airport})"
        )

        if return_date:
            url = f"{self.BASE_URL}/flights/v2/search-roundtrip"
            params = {
                "departId": source_airport,
                "arrivalId": destination_airport,
                "departDate": depart_date,
                "returnDate": return_date,
                "adults": str(adults),
                "cabinClass": cabin_class,
                "sort": "CHEAPEST",
            }
        else:
            url = f"{self.BASE_URL}/flights/v2/search-oneway"
            params = {
                "departId": source_airport,
                "arrivalId": destination_airport,
                "departDate": depart_date,
                "adults": str(adults),
                "cabinClass": cabin_class,
                "sort": "CHEAPEST",
            }

        logger.info(f"Params: {params}")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                params=params,
                timeout=40,
            )
            response.raise_for_status()
            return response.json()

    # =========================================================
    # Airport Resolution
    # =========================================================

    async def get_airport_code(self, city_name: str) -> str:
        cached = self._airport_cache.get(city_name.lower())
        if cached:
            return cached

        url = f"{self.BASE_URL}/flights/v2/auto-complete"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                params={"query": city_name},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

        airports = data.get("data", [])
        if not airports:
            raise Exception(f"No airport found for city: {city_name}")

        code = airports[0]["code"]
        self._airport_cache[city_name.lower()] = code
        return code

    # =========================================================
    # Flight Normalization
    # =========================================================

    def normalize_flights(
        self,
        api_response,
        source_city: str,
        destination_city: str,
    ):
        flight_offers = api_response.get("data", {}).get("flightOffers", [])

        normalized_flights = []
        seen_flights = set()

        for offer in flight_offers:
            normalized = self._normalize_single_flight(offer)
            if not normalized:
                continue

            unique_key = (
                normalized["airline"],
                normalized["price"],
                normalized["departure_time"],
            )

            if unique_key in seen_flights:
                continue

            seen_flights.add(unique_key)
            normalized_flights.append(normalized)

        if not normalized_flights:
            return {
                "summary": {
                    "source": source_city,
                    "destination": destination_city,
                    "total_results": 0,
                },
                "recommendations": {},
            }

        cheapest = min(normalized_flights, key=lambda x: x["price"])
        fastest = min(
            normalized_flights,
            key=lambda x: self._duration_to_minutes(x["duration"]),
        )
        best = sorted(normalized_flights, key=lambda x: (x["stops"], x["price"]))[0]

        return {
            "summary": {
                "source": source_city,
                "destination": destination_city,
                "currency": cheapest["currency"],
                "total_results": len(normalized_flights),
            },
            "recommendations": {
                "cheapest": cheapest,
                "fastest": fastest,
                "best": best,
            },
        }

    # =========================================================
    # Single Flight Transformation
    # =========================================================

    def _normalize_single_flight(self, offer):
        try:
            segment = offer["segments"][0]
            legs = segment.get("legs", [])
            if not legs:
                return None

            first_leg = legs[0]
            airline = first_leg.get("carriersData", [{}])[0].get("name")
            price_info = offer.get("priceBreakdown", {}).get("total", {})

            return {
                "airline": airline,
                "price": price_info.get("units"),
                "currency": price_info.get("currencyCode"),
                "departure_time": self._format_time(segment.get("departureTime")),
                "arrival_time": self._format_time(segment.get("arrivalTime")),
                "duration": self._format_duration(segment.get("totalTime", 0)),
                "stops": max(len(legs) - 1, 0),
            }

        except Exception as error:
            logger.error(f"Flight normalization failed: {error}")
            return None

    # =========================================================
    # Helpers
    # =========================================================

    def _format_time(self, iso_time: str):
        if not iso_time:
            return None
        return datetime.fromisoformat(iso_time).strftime("%I:%M %p")

    def _format_duration(self, duration_seconds: int):
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        return f"{hours}h {minutes}m"

    def _duration_to_minutes(self, duration: str):
        parts = duration.replace("m", "").split("h")
        hours = int(parts[0].strip())
        minutes = int(parts[1].strip())
        return (hours * 60) + minutes
