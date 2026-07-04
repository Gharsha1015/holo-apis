from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from engine.stock_predictor import run_stock_scan
from app.models.stocks import Stocks

router = APIRouter(
    prefix="/stocks",
    tags=["Stocks"]
)


@router.post("/scan")
def scan_market(
    db: Session = Depends(get_db)
):

    df = run_stock_scan()

    if df.empty:
        return {
            "message": "No stocks found."
        }

    scan_time = datetime.utcnow()

    stocks = []

    for _, row in df.iterrows():

        stocks.append(

            Stocks(
                ticker=row["Ticker"],
                score=row["Score"],
                price=row["Price"],
                entry=row["Entry"],
                sl=row["SL"],
                target=row["Target"],
                rsi=row["RSI"],
                atr_percent=row["ATR%"],
                move_30d_percent=row["Move30D%"],
                pullback_percent=row["Pullback%"],
                ema20_distance_percent=row["EMA20Dist%"],
                days_since_60d_peak=row["DaysSince60DPeak"],
                volume_spike=row["VolumeSpike"],
                run_datetime=scan_time,
            )

        )

    db.bulk_save_objects(stocks)

    db.commit()

    return {
        "message": "Scan Completed",
        "stocks_found": len(stocks)
    }
    
