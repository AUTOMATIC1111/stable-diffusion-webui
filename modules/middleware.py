import ssl
import time
import datetime
import logging
from asyncio.exceptions import CancelledError
import anyio
import starlette
import uvicorn
import fastapi
from starlette.responses import JSONResponse
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from installer import log
import modules.errors as errors

errors.install()


def setup_middleware(app: FastAPI, cmd_opts):
    log.info('Initializing middleware')
    ssl._create_default_https_context = ssl._create_unverified_context # pylint: disable=protected-access
    uvicorn_logger=logging.getLogger("uvicorn.error")
    uvicorn_logger.disabled = True
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if cmd_opts.cors_origins and cmd_opts.cors_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_origins.split(','), allow_origin_regex=cmd_opts.cors_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if cmd_opts.api_log and endpoint.startswith('/sdapi'):
            log.info('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format( # pylint: disable=consider-using-f-string, logging-format-interpolation
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code = res.status_code,
                ver = req.scope.get('http_version', '0.0'),
                cli = req.scope.get('client', ('0:0.0.0', 0))[0],
                prot = req.scope.get('scheme', 'err'),
                method = req.scope.get('method', 'err'),
                endpoint = endpoint,
                duration = duration,
            ))
        return res

    def handle_exception(req: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException) and err['error'] != 'TypeError': # do not print backtrace on known httpexceptions
            log.error(f"API error: {req.method}: {req.url} {err}")
            errors.display(e, 'HTTP API', [anyio, fastapi, uvicorn, starlette])
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(req: Request, call_next):
        try:
            return await call_next(req)
        except CancelledError:
            log.warning('WebSocket closed (ignore asyncio.exceptions.CancelledError)')
        except BaseException as e:
            return handle_exception(req, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(req: Request, e: HTTPException):
        return handle_exception(req, e)

    @app.exception_handler(Exception)
    async def general_exception_handler(req: Request, e: Exception):
        if isinstance(e, TypeError):
            return JSONResponse(status_code=500, content=jsonable_encoder(str(e)))
        else:
            return handle_exception(req, e)

    app.build_middleware_stack() # rebuild middleware stack on-the-fly
