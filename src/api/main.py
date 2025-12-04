# from src.api.utils.config import get_allowed_model_types
# from src.api.utils.error_handlers import api_exception_handler, validation_exception_handler
# from src.api.db import engine, create_admin
# from src.api.ml_models import load_all_models, clear_models, get_all_models_info
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.responses import Response
# import logging
# import sys
# import os
# from pathlib import Path
# from contextlib import asynccontextmanager
# from dotenv import load_dotenv
# from sqlmodel import SQLModel
# from prometheus_client import make_asgi_app, Counter, Gauge, Summary

# load_dotenv()
# sys.path.append(str(Path(__file__).parent.parent))

# # PROMETHEUS METRIC DEFINITIONS
# try:
#     # Counter: Tracks total number of requests for specific endpoints
#     REQUEST_COUNT = Counter(
#         'fastapi_requests_total',
#         'Total number of prediction requests received',
#         ['endpoint']
#     )
#     # Summary: Tracks request latency (duration)
#     REQUEST_LATENCY = Summary(
#         'fastapi_request_latency_seconds',
#         'Request latency in seconds',
#         ['endpoint']
#     )
#     # Gauge: Tracks the number of models currently loaded (stateful metric)
#     MODEL_LOADED_GAUGE = Gauge(
#         'fastapi_loaded_models_count',
#         'Number of ML models currently loaded in memory'
#     )
#     # Error Count
#     ERROR_COUNT = Counter(
#         'fastapi_request_error',
#         'Total number of failed requests received',
#         ['endpoint', 'error_code']
#     )
# except ValueError:
#     pass

# # Function to update the model count for the Gauge
# def update_model_count():
#     """Updates the Prometheus gauge with the current number of loaded models."""
#     info = get_all_models_info()
#     MODEL_LOADED_GAUGE.set(len(info))


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def validate_startup():
#     try:
#         logger.info("Running startup validation...")
#         from src.api.utils.setup_validator import validate_api_setup
#         success, errors, warnings = validate_api_setup()
#         for w in warnings:
#             logger.warning(f"Startup warning: {w}")
#         for e in errors:
#             logger.error(f"Startup error: {e}")
#         if not success:
#             logger.error("Startup validation failed with critical errors")
#             return False
#         logger.info("Startup validation completed successfully")
#         return True
#     except Exception as e:
#         logger.error(f"Startup validation crashed: {str(e)}")
#         return False


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Starting API server...")
#     if not validate_startup():
#         logger.error("Startup validation failed, but continuing...")

#     # Load users at startup
#     try:
#         logger.info("Loading users...")
#         create_admin()
#         logger.info("Users loaded successfully")
#     except Exception as e:
#         logger.error(f"Error loading users: {e}")
#         logger.warning(
#             "API will start – some user endpoints may be unavailable")

#     # Load ML models
#     try:
#         logger.info("Loading ML models...")
#         models = load_all_models()
#         logger.info(f"Loaded {len(models)} models")

#         # Update Prometheus Gauge on startup
#         update_model_count()

#         for mt, info in get_all_models_info().items():
#             if info["loaded"]:
#                 logger.info(f"  - {mt}: {info['metadata'].get('path')}")
#             else:
#                 logger.warning(f"  - {mt}: NOT loaded")

#     except Exception as e:
#         logger.error(f"Model loading error: {e}")
#         logger.warning("API will start – some endpoints may be unavailable")

#     yield

#     logger.info("Shutting down API server...")
#     try:
#         clear_models()
#         logger.info("Cleared models from memory")
#     except Exception as e:
#         logger.error(f"Shutdown error: {e}")


# app = FastAPI(
#     title="Churn Prediction API",
#     description="API for training and predicting customer churn using multiple ML models",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Mount Prometheus Metrics (default endpoint for scraping)
# metrics_app = make_asgi_app()
# app.mount("/metrics", metrics_app)

# origins = [
#     "http://localhost:8000",
#     "http://127.0.0.1:8000"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Add Prometheus Middleware for Request Tracking
# @app.middleware("http")
# async def prometheus_middleware(request: Request, call_next):
#     """Middleware to record request count and latency for /predict and /train."""
#     path = request.url.path
#     method = request.method

#     # Only recording metrics for the endpoints we care about monitoring
#     if path.startswith("/predict") or path.startswith("/train"):

#         with REQUEST_LATENCY.labels(endpoint=path).time():
#             REQUEST_COUNT.labels(endpoint=path).inc()
#             response = await call_next(request)
#         return response

#     response = None
#     try:
#         response = await call_next(request)
#         status_code = response.status_code
#         if status_code >= 400:
#             ERROR_COUNT.labels(endpoint=path, error_code=status_code).inc()
#         return response

#     except Exception as e:
#         status_code = 500
#         ERROR_COUNT.labels(endpoint=path, error_code=status_code).inc()
#         if response is None:
#             response = Response("Internal Server Error", status_code=500)
#         return response


# # templates
# templates = Jinja2Templates(directory="src/api/templates")
# app.state.allowed_models = get_allowed_model_types()
# app.state.environment = os.getenv("ENVIRONMENT", "development")
# app.add_exception_handler(Exception, api_exception_handler)

# # INCLUDE ROUTERS
# if os.getenv("ENVIRONMENT", "development") != "test":
#     from src.api.routers import predict, train, validate, metrics, ingest, auth, prometheus_metrics
#     app.include_router(predict.router, tags=["predictions"])
#     app.include_router(train.router,   tags=["training"])
#     app.include_router(validate.router, tags=["Data Validation"])
#     app.include_router(metrics.router, tags=["metrics"])
#     app.include_router(ingest.router,  tags=["Data ingestion"])
#     app.include_router(auth.router, tags=["auth"])
#     app.include_router(prometheus_metrics.router, tags=["prometheus_metrics"])
# else:
#     from src.api.routers import predict, train
#     app.include_router(predict.router, tags=["predictions"])
#     app.include_router(train.router,   tags=["training"])
#     logger.info("Test environment detected: including predict and train routers")

# # Existing UI and health endpoints remain unchanged 
# @app.get("/pages/register", response_class=HTMLResponse)
# async def get_register_page(request: Request):
#     return templates.TemplateResponse("register.html", {"request": request})

# @app.get("/", response_class=HTMLResponse)
# async def get_login_page(request: Request):
#     return templates.TemplateResponse("login.html", {"request": request})

# @app.get("/home", response_class=HTMLResponse)
# async def ui_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/ingest", response_class=HTMLResponse)
# async def ui_ingest(request: Request):
#     return templates.TemplateResponse("ingest.html", {"request": request})


# @app.get("/predict", response_class=HTMLResponse)
# async def ui_predict(request: Request):
#     return templates.TemplateResponse("predict.html", {"request": request})


# @app.get("/train", response_class=HTMLResponse)
# async def ui_train(request: Request):
#     return templates.TemplateResponse("train.html", {"request": request})


# @app.get("/metrics-ui", response_class=HTMLResponse)
# async def ui_metrics(request: Request):
#     return templates.TemplateResponse("metrics.html", {"request": request})


# @app.get("/users", response_class=HTMLResponse)
# async def ui_users(request: Request):
#     return templates.TemplateResponse("users.html", {"request": request})


# @app.get("/data_view", response_class=HTMLResponse)
# async def ui_data_view(request: Request):
#     return templates.TemplateResponse("data_view.html", {"request": request})


# @app.get("/health-ui", response_class=HTMLResponse)
# async def health_ui(request: Request):
#     """Human-readable health page (uses health.html)"""
#     health_data = await health_check()
#     return templates.TemplateResponse(
#         "health.html",
#         {"request": request, "data": health_data}
#     )


# @app.get("/health")
# async def health_check():
#     models_info = get_all_models_info()
#     loaded = [mt for mt, info in models_info.items() if info['loaded']]
#     return {
#         "status": "healthy",
#         "models_loaded": len(loaded),
#         "models": models_info,
#         "environment": os.getenv('ENVIRONMENT', 'development'),
#         "version": "2.0.0"
#     }


# @app.get("/models")
# async def get_models_status():
#     return {"models": get_all_models_info()}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )
from src.api.utils.config import get_allowed_model_types
from src.api.utils.error_handlers import api_exception_handler, validation_exception_handler
from src.api.db import engine, create_admin
from src.api.ml_models import load_all_models, clear_models, get_all_models_info
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sqlmodel import SQLModel
from prometheus_client import make_asgi_app, Counter, Gauge, Summary

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

# PROMETHEUS METRIC DEFINITIONS
try:
    # Counter: Tracks total number of requests for specific endpoints
    REQUEST_COUNT = Counter(
        'fastapi_requests_total',
        'Total number of prediction requests received',
        ['endpoint']
    )
    # Summary: Tracks request latency (duration)
    REQUEST_LATENCY = Summary(
        'fastapi_request_latency_seconds',
        'Request latency in seconds',
        ['endpoint']
    )
    # Gauge: Tracks the number of models currently loaded (stateful metric)
    MODEL_LOADED_GAUGE = Gauge(
        'fastapi_loaded_models_count',
        'Number of ML models currently loaded in memory'
    )
    # Error Count
    ERROR_COUNT = Counter(
        'fastapi_request_error',
        'Total number of failed requests received',
        ['endpoint', 'error_code']
    )
    # Counter: Tracks the count of predictions for each class (0 or 1) per model
    PREDICTION_CLASS_COUNT = Counter(
        'fastapi_prediction_class_total',
        'Total number of predictions by class (0 or 1) for each model.',
        ['model_name', 'prediction_class']
    )
    # Gauge: Tracks the average prediction score/probability for each model
    AVG_PREDICTION_PROBABILITY = Gauge(
        'model_avg_prediction_probability',
        'The average prediction probability score returned by each model.',
        ['model_name']
    )

    # Counter: Tracks the total number of successful predictions made by each model (Operational)
    MODEL_PREDICTION_COUNT = Counter(
        'model_successful_predictions_total',
        'Total count of successful predictions served by each model.',
        ['model_name']
    )

# Gauge: Tracks the Average Prediction Drift (e.g., Average-Mean) over a window.

    CURRENT_BATCH_AVG_PROB = Gauge(
        'model_current_batch_avg_prob',
        'Average prediction probability for the latest batch processed by the model.',
        ['model_name']
    )
except ValueError:
    pass
# Function to update the model count for the Gauge


def update_model_count():
    """Updates the Prometheus gauge with the current number of loaded models."""
    info = get_all_models_info()
    MODEL_LOADED_GAUGE.set(len(info))

# from src.api.template_context import get_template_context


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_startup():
    try:
        logger.info("Running startup validation...")
        from src.api.utils.setup_validator import validate_api_setup
        success, errors, warnings = validate_api_setup()
        for w in warnings:
            logger.warning(f"Startup warning: {w}")
        for e in errors:
            logger.error(f"Startup error: {e}")
        if not success:
            logger.error("Startup validation failed with critical errors")
            return False
        logger.info("Startup validation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Startup validation crashed: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API server...")
    if not validate_startup():
        logger.error("Startup validation failed, but continuing...")

    # Load users at startup
    try:
        logger.info("Loading users...")
        create_admin()
        logger.info("Users loaded successfully")
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        logger.warning(
            "API will start – some user endpoints may be unavailable")

    # Load ML models
    try:
        logger.info("Loading ML models...")
        models = load_all_models()
        logger.info(f"Loaded {len(models)} models")

        # Update Prometheus Gauge on startup
        update_model_count()

        for mt, info in get_all_models_info().items():
            if info["loaded"]:
                logger.info(f"  - {mt}: {info['metadata'].get('path')}")
            else:
                logger.warning(f"  - {mt}: NOT loaded")

    except Exception as e:
        logger.error(f"Model loading error: {e}")
        logger.warning("API will start – some endpoints may be unavailable")

    # Yield control to FastAPI
    yield

    logger.info("Shutting down API server...")
    try:
        clear_models()
        logger.info("Cleared models from memory")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

 # Update Prometheus Gauge on startup
        update_model_count()

        for mt, info in get_all_models_info().items():
            if info['loaded']:
                logger.info(f"  - {mt}: {info['metadata'].get('path')}")
            else:
                logger.warning(f"  - {mt}: NOT loaded")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        logger.warning("API will start – some endpoints may be unavailable")

    yield

    logger.info("Shutting down API server...")
    try:
        clear_models()
        logger.info("Cleared models from memory")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="Churn Prediction API",
    description="API for training and predicting customer churn using multiple ML models",
    version="1.0.0",
    lifespan=lifespan
)

# Mount Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus Middleware for Request Tracking


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Middleware to record request count and latency for /predict and /train."""
    path = request.url.path
    method = request.method

    # Only record metrics for the endpoints we care about monitoring
    if path.startswith("/predict") or path.startswith("/train"):

        # Start timer and record when the 'with' block exits
        with REQUEST_LATENCY.labels(endpoint=path).time():
            # Increment the counter
            REQUEST_COUNT.labels(endpoint=path).inc()

            response = await call_next(request)
        return response

    response = None
    try:
        # Call the next middleware/endpoint handler
        response = await call_next(request)

        # Check for application-generated errors (Status >= 400)
        status_code = response.status_code

        if status_code >= 400:
            # Log the specific application error code (e.g., 404, 403, 503)
            ERROR_COUNT.labels(endpoint=path, error_code=status_code).inc()

        return response

    except Exception as e:
        # For unhandled exceptions, we default to the standard 500 Internal Server Error
        status_code = 500

        # Log the 500 error code
        ERROR_COUNT.labels(endpoint=path, error_code=status_code).inc()

        # Must generate a 500 response to return to the client if none was created
        if response is None:
            response = Response("Internal Server Error", status_code=500)

        # In some frameworks, you might need to raise the exception again
        # for higher-level error handlers, but here we return the 500 Response.
        return response

    response = await call_next(request)
    return response


async def add_user_to_request(request: Request, call_next):
    """Add user information to request state if authenticated."""
    try:
        # Skip authentication for public routes

        public_routes = ["/", "/pages/register", "/auth/login",
                         "/auth/register", "/health", "/metrics"]
        if request.url.path in public_routes or request.url.path.startswith("/static"):
            response = await call_next(request)
            return response

        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Create a mock dependency context
            from fastapi.security import HTTPAuthorizationCredentials

            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=auth_header[7:])

            # Get database session
            from src.api.db import get_db
            db = next(get_db())

            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=auth_header[7:])

            # Get database session
            from src.api.db import get_db
            db = next(get_db())
            try:
                # Get current user
                user = await get_current_user(credentials, db)
                request.state.user = user
            except HTTPException:
                request.state.user = None
        else:
            request.state.user = None
    except Exception:
        request.state.user = None

    response = await call_next(request)
    return response

# templates

templates = Jinja2Templates(directory="src/api/templates")

app.state.allowed_models = get_allowed_model_types()
app.state.environment = os.getenv("ENVIRONMENT", "development")


app.add_exception_handler(Exception, api_exception_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# INCLUDE ROUTERS FIRST (before app routes) so they take priority
if os.getenv("ENVIRONMENT", "development") != "test":
    from src.api.routers import predict, train, validate, metrics, ingest, auth
    app.include_router(predict.router, tags=["predictions"])
    app.include_router(train.router,   tags=["training"])
    app.include_router(validate.router, tags=["Data Validation"])
    app.include_router(metrics.router, tags=["metrics"])
    app.include_router(ingest.router,  tags=["Data ingestion"])
    app.include_router(auth.router, tags=["auth"])
else:
    from src.api.routers import predict, train
    app.include_router(predict.router, tags=["predictions"])
    app.include_router(train.router,   tags=["training"])
    logger.info("Test environment detected: including predict and train routers")


@app.get("/pages/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def ui_root(request: Request):
    """Home page – uses index.html"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ingest", response_class=HTMLResponse)
async def ui_ingest(request: Request):
    return templates.TemplateResponse("ingest.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def ui_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.get("/train", response_class=HTMLResponse)
async def ui_train(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.get("/metrics-ui", response_class=HTMLResponse)
async def ui_metrics(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request})


@app.get("/users", response_class=HTMLResponse)
async def ui_users(request: Request):
    return templates.TemplateResponse("users.html", {"request": request})


@app.get("/data_view", response_class=HTMLResponse)
async def ui_data_view(request: Request):
    return templates.TemplateResponse("data_view.html", {"request": request})


@app.get("/health-ui", response_class=HTMLResponse)
async def health_ui(request: Request):
    """Human-readable health page (uses health.html)"""
    health_data = await health_check()
    return templates.TemplateResponse(
        "health.html",
        {"request": request, "data": health_data}
    )


@app.get("/health")
async def health_check():
    models_info = get_all_models_info()
    loaded = [mt for mt, info in models_info.items() if info['loaded']]
    return {
        "status": "healthy",
        "models_loaded": len(loaded),
        "models": models_info,
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "version": "2.0.0"
    }


@app.get("/models")
async def get_models_status():
    return {"models": get_all_models_info()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )