# uvicorn reload trigger: 2026-01-27 12:18
# start : python -m uvicorn api:app --reload

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from inference.predict import analyze_text
# from inference.explain import explain_text
# from datetime import datetime
# import uuid

# app = FastAPI(title="MedVerax – Robust Health AI")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Request(BaseModel):
#     text: str

# @app.post("/analyze")
# def analyze(req: Request):
#     pred = analyze_text(req.text)
#     exp = explain_text(req.text)

#     return {
#         "success": True,
#         "analysis": {
#             **pred,
#             **exp
#         },
#         "meta": {
#             "id": str(uuid.uuid4()),
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from inference.analyze import analyze_text
from datetime import datetime, timezone
import uuid

app = FastAPI(title="MedVerax – Robust Health AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    text: str

@app.post("/analyze")
def analyze(req: Request):
    return {
        "success": True,
        "analysis": analyze_text(req.text),
        "meta": {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }



# from src.loggers.logger import get_logger
# from src.exceptions.custom_exceptions import MedVeraxException

# logger = get_logger(__name__)

# @app.post("/analyze")
# def analyze(req: Request):
#     try:
#         logger.info("API request received")

#         result = analyze_text(req.text)

#         return {
#             "success": True,
#             "analysis": result,
#             "meta": {
#                 "id": str(uuid.uuid4()),
#                 "timestamp": datetime.now(timezone.utc).isoformat()
#             }
#         }

#     except MedVeraxException as e:
#         logger.error("Handled MedVeraxException", exc_info=True)
#         return {"success": False, "error": str(e)}
