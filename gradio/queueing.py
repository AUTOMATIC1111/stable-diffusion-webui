from __future__ import annotations

import asyncio
import copy
import sys
import time
from asyncio import TimeoutError as AsyncTimeOutError
from collections import deque
from typing import Any, Deque

import fastapi
import httpx

from gradio.data_classes import Estimation, PredictBody, Progress, ProgressUnit
from gradio.helpers import TrackedIterable
from gradio.utils import AsyncRequest, run_coro_in_background, set_task_name


class Event:
    def __init__(
        self,
        websocket: fastapi.WebSocket,
        session_hash: str,
        fn_index: int,
    ):
        self.websocket = websocket
        self.session_hash: str = session_hash
        self.fn_index: int = fn_index
        self._id = f"{self.session_hash}_{self.fn_index}"
        self.data: PredictBody | None = None
        self.lost_connection_time: float | None = None
        self.token: str | None = None
        self.progress: Progress | None = None
        self.progress_pending: bool = False

    async def disconnect(self, code: int = 1000):
        await self.websocket.close(code=code)


class Queue:
    def __init__(
        self,
        live_updates: bool,
        concurrency_count: int,
        update_intervals: float,
        max_size: int | None,
        blocks_dependencies: list,
    ):
        self.event_queue: Deque[Event] = deque()
        self.events_pending_reconnection = []
        self.stopped = False
        self.max_thread_count = concurrency_count
        self.update_intervals = update_intervals
        self.active_jobs: list[None | list[Event]] = [None] * concurrency_count
        self.delete_lock = asyncio.Lock()
        self.server_path = None
        self.duration_history_total = 0
        self.duration_history_count = 0
        self.avg_process_time = 0
        self.avg_concurrent_process_time = None
        self.queue_duration = 1
        self.live_updates = live_updates
        self.sleep_when_free = 0.05
        self.progress_update_sleep_when_free = 0.1
        self.max_size = max_size
        self.blocks_dependencies = blocks_dependencies
        self.access_token = ""
        self.queue_client = None

    async def start(self, progress_tracking=False, ssl_verify=True):
        # So that the client is attached to the running event loop
        self.queue_client = httpx.AsyncClient(verify=ssl_verify)

        run_coro_in_background(self.start_processing)
        if progress_tracking:
            run_coro_in_background(self.start_progress_tracking)
        if not self.live_updates:
            run_coro_in_background(self.notify_clients)

    def close(self):
        self.stopped = True

    def resume(self):
        self.stopped = False

    def set_url(self, url: str):
        self.server_path = url

    def set_access_token(self, token: str):
        self.access_token = token

    def get_active_worker_count(self) -> int:
        count = 0
        for worker in self.active_jobs:
            if worker is not None:
                count += 1
        return count

    def get_events_in_batch(self) -> tuple[list[Event] | None, bool]:
        if not (self.event_queue):
            return None, False

        first_event = self.event_queue.popleft()
        events = [first_event]

        event_fn_index = first_event.fn_index
        batch = self.blocks_dependencies[event_fn_index]["batch"]

        if batch:
            batch_size = self.blocks_dependencies[event_fn_index]["max_batch_size"]
            rest_of_batch = [
                event for event in self.event_queue if event.fn_index == event_fn_index
            ][: batch_size - 1]
            events.extend(rest_of_batch)
            [self.event_queue.remove(event) for event in rest_of_batch]

        return events, batch

    async def start_processing(self) -> None:
        while not self.stopped:
            if not self.event_queue:
                await asyncio.sleep(self.sleep_when_free)
                continue

            if None not in self.active_jobs:
                await asyncio.sleep(self.sleep_when_free)
                continue
            # Using mutex to avoid editing a list in use
            async with self.delete_lock:
                events, batch = self.get_events_in_batch()

            if events:
                self.active_jobs[self.active_jobs.index(None)] = events
                task = run_coro_in_background(self.process_events, events, batch)
                run_coro_in_background(self.broadcast_live_estimations)
                set_task_name(task, events[0].session_hash, events[0].fn_index, batch)

    async def start_progress_tracking(self) -> None:
        while not self.stopped:
            if not any(self.active_jobs):
                await asyncio.sleep(self.progress_update_sleep_when_free)
                continue

            for job in self.active_jobs:
                if job is None:
                    continue
                for event in job:
                    if event.progress_pending and event.progress:
                        event.progress_pending = False
                        client_awake = await self.send_message(
                            event, event.progress.dict()
                        )
                        if not client_awake:
                            await self.clean_event(event)

            await asyncio.sleep(self.progress_update_sleep_when_free)

    def set_progress(
        self,
        event_id: str,
        iterables: list[TrackedIterable] | None,
    ):
        if iterables is None:
            return
        for job in self.active_jobs:
            if job is None:
                continue
            for evt in job:
                if evt._id == event_id:
                    progress_data: list[ProgressUnit] = []
                    for iterable in iterables:
                        progress_unit = ProgressUnit(
                            index=iterable.index,
                            length=iterable.length,
                            unit=iterable.unit,
                            progress=iterable.progress,
                            desc=iterable.desc,
                        )
                        progress_data.append(progress_unit)
                    evt.progress = Progress(progress_data=progress_data)
                    evt.progress_pending = True

    def push(self, event: Event) -> int | None:
        """
        Add event to queue, or return None if Queue is full
        Parameters:
            event: Event to add to Queue
        Returns:
            rank of submitted Event
        """
        queue_len = len(self.event_queue)
        if self.max_size is not None and queue_len >= self.max_size:
            return None
        self.event_queue.append(event)
        return queue_len

    async def clean_event(self, event: Event) -> None:
        if event in self.event_queue:
            async with self.delete_lock:
                self.event_queue.remove(event)

    async def broadcast_live_estimations(self) -> None:
        """
        Runs 2 functions sequentially instead of concurrently. Otherwise dced clients are tried to get deleted twice.
        """
        if self.live_updates:
            await self.broadcast_estimations()

    async def gather_event_data(self, event: Event, receive_timeout=60) -> bool:
        """
        Gather data for the event
        Parameters:
            event: the Event to gather data for
            receive_timeout: how long to wait for data to be received from frontend
        """
        if not event.data:
            client_awake = await self.send_message(event, {"msg": "send_data"})
            if not client_awake:
                return False
            data, client_awake = await self.get_message(event, timeout=receive_timeout)
            if not client_awake:
                # In the event, we timeout due to large data size
                # Let the client know, otherwise will hang
                await self.send_message(
                    event,
                    {
                        "msg": "process_completed",
                        "output": {"error": "Time out uploading data to server"},
                        "success": False,
                    },
                )
                return False
            event.data = data
        return True

    async def notify_clients(self) -> None:
        """
        Notify clients about events statuses in the queue periodically.
        """
        while not self.stopped:
            await asyncio.sleep(self.update_intervals)
            if self.event_queue:
                await self.broadcast_estimations()

    async def broadcast_estimations(self) -> None:
        estimation = self.get_estimation()
        # Send all messages concurrently
        await asyncio.gather(
            *[
                self.send_estimation(event, estimation, rank)
                for rank, event in enumerate(self.event_queue)
            ]
        )

    async def send_estimation(
        self, event: Event, estimation: Estimation, rank: int
    ) -> Estimation:
        """
        Send estimation about ETA to the client.

        Parameters:
            event:
            estimation:
            rank:
        """
        estimation.rank = rank

        if self.avg_concurrent_process_time is not None:
            estimation.rank_eta = (
                estimation.rank * self.avg_concurrent_process_time
                + self.avg_process_time
            )
            if None not in self.active_jobs:
                # Add estimated amount of time for a thread to get empty
                estimation.rank_eta += self.avg_concurrent_process_time
        client_awake = await self.send_message(event, estimation.dict())
        if not client_awake:
            await self.clean_event(event)
        return estimation

    def update_estimation(self, duration: float) -> None:
        """
        Update estimation by last x element's average duration.

        Parameters:
            duration:
        """
        self.duration_history_total += duration
        self.duration_history_count += 1
        self.avg_process_time = (
            self.duration_history_total / self.duration_history_count
        )
        self.avg_concurrent_process_time = self.avg_process_time / min(
            self.max_thread_count, self.duration_history_count
        )
        self.queue_duration = self.avg_concurrent_process_time * len(self.event_queue)

    def get_estimation(self) -> Estimation:
        return Estimation(
            queue_size=len(self.event_queue),
            avg_event_process_time=self.avg_process_time,
            avg_event_concurrent_process_time=self.avg_concurrent_process_time,
            queue_eta=self.queue_duration,
        )

    def get_request_params(self, websocket: fastapi.WebSocket) -> dict[str, Any]:
        return {
            "url": str(websocket.url),
            "headers": dict(websocket.headers),
            "query_params": dict(websocket.query_params),
            "path_params": dict(websocket.path_params),
            "client": {"host": websocket.client.host, "port": websocket.client.port},  # type: ignore
        }

    async def call_prediction(self, events: list[Event], batch: bool):
        data = events[0].data
        assert data is not None, "No event data"
        token = events[0].token
        data.event_id = events[0]._id if not batch else None
        try:
            data.request = self.get_request_params(events[0].websocket)
        except ValueError:
            pass

        if batch:
            data.data = list(zip(*[event.data.data for event in events if event.data]))
            data.request = [
                self.get_request_params(event.websocket)
                for event in events
                if event.data
            ]
            data.batched = True
        response = await AsyncRequest(
            method=AsyncRequest.Method.POST,
            url=f"{self.server_path}api/predict",
            json=dict(data),
            headers={"Authorization": f"Bearer {self.access_token}"},
            cookies={"access-token": token} if token is not None else None,
            client=self.queue_client,
        )
        return response

    async def process_events(self, events: list[Event], batch: bool) -> None:
        awake_events: list[Event] = []
        try:
            for event in events:
                client_awake = await self.gather_event_data(event)
                if client_awake:
                    client_awake = await self.send_message(
                        event, {"msg": "process_starts"}
                    )
                if client_awake:
                    awake_events.append(event)
            if not awake_events:
                return
            begin_time = time.time()
            response = await self.call_prediction(awake_events, batch)
            if response.has_exception:
                for event in awake_events:
                    await self.send_message(
                        event,
                        {
                            "msg": "process_completed",
                            "output": {"error": str(response.exception)},
                            "success": False,
                        },
                    )
            elif response.json.get("is_generating", False):
                old_response = response
                while response.json.get("is_generating", False):
                    # Python 3.7 doesn't have named tasks.
                    # In order to determine if a task was cancelled, we
                    # ping the websocket to see if it was closed mid-iteration.
                    if sys.version_info < (3, 8):
                        is_alive = await self.send_message(event, {"msg": "alive?"})
                        if not is_alive:
                            return
                    old_response = response
                    open_ws = []
                    for event in awake_events:
                        open = await self.send_message(
                            event,
                            {
                                "msg": "process_generating",
                                "output": old_response.json,
                                "success": old_response.status == 200,
                            },
                        )
                        open_ws.append(open)
                    awake_events = [
                        e for e, is_open in zip(awake_events, open_ws) if is_open
                    ]
                    if not awake_events:
                        return
                    response = await self.call_prediction(awake_events, batch)
                for event in awake_events:
                    if response.status != 200:
                        relevant_response = response
                    else:
                        relevant_response = old_response

                    await self.send_message(
                        event,
                        {
                            "msg": "process_completed",
                            "output": relevant_response.json,
                            "success": relevant_response.status == 200,
                        },
                    )
            else:
                output = copy.deepcopy(response.json)
                for e, event in enumerate(awake_events):
                    if batch and "data" in output:
                        output["data"] = list(zip(*response.json.get("data")))[e]
                    await self.send_message(
                        event,
                        {
                            "msg": "process_completed",
                            "output": output,
                            "success": response.status == 200,
                        },
                    )
            end_time = time.time()
            if response.status == 200:
                self.update_estimation(end_time - begin_time)
        finally:
            for event in awake_events:
                try:
                    await event.disconnect()
                except Exception:
                    pass
            self.active_jobs[self.active_jobs.index(events)] = None
            for event in events:
                await self.clean_event(event)
                # Always reset the state of the iterator
                # If the job finished successfully, this has no effect
                # If the job is cancelled, this will enable future runs
                # to start "from scratch"
                await self.reset_iterators(event.session_hash, event.fn_index)

    async def send_message(self, event, data: dict, timeout: float | int = 1) -> bool:
        try:
            await asyncio.wait_for(
                event.websocket.send_json(data=data), timeout=timeout
            )
            return True
        except Exception:
            await self.clean_event(event)
            return False

    async def get_message(self, event, timeout=5) -> tuple[PredictBody | None, bool]:
        try:
            data = await asyncio.wait_for(
                event.websocket.receive_json(), timeout=timeout
            )
            return PredictBody(**data), True
        except AsyncTimeOutError:
            await self.clean_event(event)
            return None, False

    async def reset_iterators(self, session_hash: str, fn_index: int):
        await AsyncRequest(
            method=AsyncRequest.Method.POST,
            url=f"{self.server_path}reset",
            json={
                "session_hash": session_hash,
                "fn_index": fn_index,
            },
            client=self.queue_client,
        )
