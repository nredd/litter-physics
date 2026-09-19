"""Deterministic stand-in for native `run_json` used ONLY in unit tests.

This is not a physics model. It emits contract-shaped output so that Python
orchestration (validation, artifacts, resume, budgets, sweeps, viewer) can be tested
before the numerical modules are registered. Integration tests against the real
extension live in `test_native_integration.py`.

References:
    docs/wire-contract.md
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeCore:
    """Configurable fake solver honoring the wire contract."""

    max_sim_time_per_call: float | None = None
    fail_with: str | None = None
    calls: int = 0

    def __call__(self, request_json: str, resume_json: str | None) -> str:
        """Mimic `run_json`.

        Parameters:
            request_json (str): Canonical request JSON.
            resume_json (str | None): Checkpoint JSON or None.

        Returns:
            str: Contract-shaped output JSON.

        Raises:
            ValueError: When configured to fail, or on a mismatched resume request.
        """
        self.calls += 1
        if self.fail_with is not None:
            raise ValueError(self.fail_with)
        request = json.loads(request_json)
        if not math.isfinite(request["max_wall_time_s"]) or request["max_wall_time_s"] <= 0:
            raise ValueError("fake core: invalid wall budget")
        resume = json.loads(resume_json) if resume_json is not None else None
        if resume is not None:
            stripped = dict(resume["request"]) | {"max_wall_time_s": request["max_wall_time_s"]}
            if stripped != request:
                raise ValueError("fake core: resume request mismatch")
        start = 0.0 if resume is None else float(resume["time_s"])
        rng_state = 0 if resume is None else int(resume["state"]["rng"])
        duration = float(request["duration_s"])
        end = duration
        if self.max_sim_time_per_call is not None:
            end = min(duration, start + self.max_sim_time_per_call)
        interval = float(request["record_interval_s"])
        frames = []
        metrics = []
        first_frame = 0 if resume is None else int(resume["state"]["next_frame"])
        frame_index = first_frame
        while True:
            time_s = frame_index * interval
            if time_s > end + 1e-12:
                break
            if time_s < start - 1e-12:
                frame_index += 1
                continue
            clamped = min(time_s, end)
            frames.append(self._frame(request, clamped))
            metrics.extend(self._metrics(request, clamped))
            frame_index += 1
            rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) % 2**64
        status = "completed" if end >= duration else "budget_exhausted"
        mode = request["mode"]
        output: dict[str, Any] = {
            "schema_version": 1,
            "mode": mode,
            "status": status,
            "fidelity": "preliminary_household" if mode == "household" else "research_unvalidated",
            "time_s": end,
            "frames": frames,
            "metrics": metrics,
            "observables": {"mass_residual_kg": 0.0, "fake_rng": float(rng_state % 1000)},
            "diagnostics": ["FAKE CORE: no physics; contract-shaped output for tests"],
            "checkpoint": {
                "schema_version": 1,
                "mode": mode,
                "request": dict(request) | {"max_wall_time_s": request["max_wall_time_s"]},
                "time_s": end,
                "state": {"rng": rng_state, "next_frame": frame_index},
            },
        }
        return json.dumps(output)

    @staticmethod
    def _frame(request: dict[str, Any], time_s: float) -> dict[str, Any]:
        """Build one frame with pellets settled on a grid.

        Parameters:
            request (dict[str, Any]): Parsed request.
            time_s (float): Frame time.

        Returns:
            dict[str, Any]: Frame payload.
        """
        positions = []
        radii = []
        materials = []
        box_ids = []
        for box in request["boxes"]:
            radius = float(box["pellet_radius_m"])
            size = box["size_m"]
            per_row = max(1, int(size[0] // (2.0 * radius)))
            for index in range(int(box["pellet_count"])):
                column = index % per_row
                row = index // per_row
                x = min(size[0] - radius, radius + 2.0 * radius * column)
                y = min(size[1] - radius, radius + 2.0 * radius * (row % 3))
                z = max(radius, size[2] * 0.5 * math.exp(-time_s) + radius)
                positions.append(
                    [x + box["origin_m"][0], y + box["origin_m"][1], z + box["origin_m"][2]]
                )
                radii.append(radius)
                materials.append("wood")
                box_ids.append(box["id"])
        return {
            "time_s": time_s,
            "positions_m": positions,
            "radii_m": radii,
            "materials": materials,
            "box_ids": box_ids,
        }

    @staticmethod
    def _metrics(request: dict[str, Any], time_s: float) -> list[dict[str, Any]]:
        """Build ledger rows with a trivially conserved budget.

        Parameters:
            request (dict[str, Any]): Parsed request.
            time_s (float): Sample time.

        Returns:
            list[dict[str, Any]]: Rows for every household compartment.
        """
        wood = 0.0
        for box in request["boxes"]:
            volume = math.pi * box["pellet_radius_m"] ** 2 * box["pellet_length_m"]
            wood += box["pellet_count"] * volume * box["pellet_density_kg_m3"]
        water = sum(
            event["amount_kg"] * event["water_fraction"]
            for event in request["events"]
            if event["time_s"] <= time_s and event["kind"] in {"urinate", "defecate"}
        )
        waste = sum(
            event["amount_kg"] * (1.0 - event["water_fraction"])
            for event in request["events"]
            if event["time_s"] <= time_s and event["kind"] in {"urinate", "defecate"}
        )
        compartments = ["bed", "drawer", "floor", "removed", "evaporated"]
        if request["mode"] == "research":
            compartments = ["domain", "outflow"]
        rows = []
        for compartment in compartments:
            first = compartment in {"bed", "domain"}
            rows.append(
                {
                    "time_s": time_s,
                    "compartment": compartment,
                    "wood_kg": wood if first else 0.0,
                    "waste_kg": waste if first else 0.0,
                    "water_kg": water if first else 0.0,
                }
            )
        return rows
