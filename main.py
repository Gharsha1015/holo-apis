from fastapi import FastAPI
from app.db.database import engine
from app.db.base import Base

from app.api.stock_scan import router as stock_router

Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="Stock Scanner API",
    version="1.0.0"
)

app.include_router(stock_router)