import ssl
import time
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
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    if cmd_opts.cors_origins and cmd_opts.cors_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_origins.split(','), allow_origin_regex=cmd_opts.cors_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        try:
            ts = time.time()
            res: Response = await call_next(req)
            duration = str(round(time.time() - ts, 4))
            res.headers["X-Process-Time"] = duration
            endpoint = req.scope.get('path', 'err')
            token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
            if (cmd_opts.api_log or cmd_opts.api_only) and endpoint.startswith('/sdapi'):
                if '/sdapi/v1/log' in endpoint:
                    return res
                log.info('API {user} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format( # pylint: disable=consider-using-f-string, logging-format-interpolation
                    user = app.tokens.get(token),
                    code = res.status_code,
                    ver = req.scope.get('http_version', '0.0'),
                    cli = req.scope.get('client', ('0:0.0.0', 0))[0],
                    prot = req.scope.get('scheme', 'err'),
                    method = req.scope.get('method', 'err'),
                    endpoint = endpoint,
                    duration = duration,
                ))
            return res
        except CancelledError:
            log.warning('WebSocket closed (ignore asyncio.exceptions.CancelledError)')
        except BaseException as e:
            return handle_exception(req, e)

    def handle_exception(req: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "code": vars(e).get('status_code', 500),
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        log.error(f"API error: {req.method}: {req.url} {err}")
        if not isinstance(e, HTTPException) and err['error'] != 'TypeError': # do not print backtrace on known httpexceptions
            errors.display(e, 'HTTP API', [anyio, fastapi, uvicorn, starlette])
        elif err['code'] == 404 or err['code'] == 401:
            pass
        else:
            log.debug(e, exc_info=True) # print stack trace
        return JSONResponse(status_code=err['code'], content=jsonable_encoder(err))

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
