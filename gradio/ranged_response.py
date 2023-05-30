# Taken from https://gist.github.com/kevinastone/a6a62db57577b3f24e8a6865ed311463
# Context: https://github.com/encode/starlette/pull/1090
from __future__ import annotations

import os
import re
import stat
from typing import NamedTuple
from urllib.parse import quote

import aiofiles
from aiofiles.os import stat as aio_stat
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response, guess_type
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

RANGE_REGEX = re.compile(r"^bytes=(?P<start>\d+)-(?P<end>\d*)$")


class ClosedRange(NamedTuple):
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __bool__(self) -> bool:
        return len(self) > 0


class OpenRange(NamedTuple):
    start: int
    end: int | None = None

    def clamp(self, start: int, end: int) -> ClosedRange:
        begin = max(self.start, start)
        end = min(x for x in (self.end, end) if x)

        begin = min(begin, end)
        end = max(begin, end)

        return ClosedRange(begin, end)


class RangedFileResponse(Response):
    chunk_size = 4096

    def __init__(
        self,
        path: str | os.PathLike,
        range: OpenRange,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        filename: str | None = None,
        stat_result: os.stat_result | None = None,
        method: str | None = None,
    ) -> None:
        assert aiofiles is not None, "'aiofiles' must be installed to use FileResponse"
        self.path = path
        self.range = range
        self.filename = filename
        self.background = None
        self.send_header_only = method is not None and method.upper() == "HEAD"
        if media_type is None:
            media_type = guess_type(filename or path)[0] or "text/plain"
        self.media_type = media_type
        self.init_headers(headers or {})
        if self.filename is not None:
            content_disposition_filename = quote(self.filename)
            if content_disposition_filename != self.filename:
                content_disposition = (
                    f"attachment; filename*=utf-8''{content_disposition_filename}"
                )
            else:
                content_disposition = f'attachment; filename="{self.filename}"'
            self.headers.setdefault("content-disposition", content_disposition)
        self.stat_result = stat_result

    def set_range_headers(self, range: ClosedRange) -> None:
        assert self.stat_result
        total_length = self.stat_result.st_size
        content_length = len(range)
        self.headers[
            "content-range"
        ] = f"bytes {range.start}-{range.end}/{total_length}"
        self.headers["content-length"] = str(content_length)
        pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.stat_result is None:
            try:
                stat_result = await aio_stat(self.path)
                self.stat_result = stat_result
            except FileNotFoundError as fnfe:
                raise RuntimeError(
                    f"File at path {self.path} does not exist."
                ) from fnfe
            else:
                mode = stat_result.st_mode
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f"File at path {self.path} is not a file.")

        byte_range = self.range.clamp(0, self.stat_result.st_size)
        self.set_range_headers(byte_range)

        async with aiofiles.open(self.path, mode="rb") as file:
            await file.seek(byte_range.start)
            await send(
                {
                    "type": "http.response.start",
                    "status": 206,
                    "headers": self.raw_headers,
                }
            )
            if self.send_header_only:
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
            else:
                remaining_bytes = len(byte_range)

                if not byte_range:
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                    return

                while remaining_bytes > 0:
                    chunk_size = min(self.chunk_size, remaining_bytes)
                    chunk = await file.read(chunk_size)
                    remaining_bytes -= len(chunk)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": remaining_bytes > 0,
                        }
                    )


class RangedStaticFiles(StaticFiles):
    def file_response(
        self,
        full_path: str | os.PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        request_headers = Headers(scope=scope)

        if request_headers.get("range"):
            response = self.ranged_file_response(
                full_path, stat_result=stat_result, scope=scope
            )
        else:
            response = super().file_response(
                full_path, stat_result=stat_result, scope=scope, status_code=status_code
            )
        response.headers["accept-ranges"] = "bytes"
        return response

    def ranged_file_response(
        self,
        full_path: str | os.PathLike,
        stat_result: os.stat_result,
        scope: Scope,
    ) -> Response:
        method = scope["method"]
        request_headers = Headers(scope=scope)

        range_header = request_headers["range"]

        match = RANGE_REGEX.search(range_header)
        if not match:
            raise HTTPException(400)

        start, end = match.group("start"), match.group("end")

        range = OpenRange(int(start), int(end) if end else None)

        return RangedFileResponse(
            full_path, range, stat_result=stat_result, method=method
        )
