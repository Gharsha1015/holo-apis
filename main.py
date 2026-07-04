from fastapi import FastAPI
from app.db.database import engine
from app.db.base import Base
from fastapi.middleware.cors import CORSMiddleware

from app.api.stock_scan import router as stock_router

Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="Stock Scanner API",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",      
    "http://127.0.0.1:5173",
    "https://holo-frontend-three.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stock_router)