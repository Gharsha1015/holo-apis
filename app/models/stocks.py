from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from app.db.base import Base


class Stocks(Base):
    __tablename__ = "stocks"
    __table_args__ = (
                        UniqueConstraint(
                            "ticker",
                            "run_datetime",
                            name="uq_stocks_ticker_run_datetime",
                        ),
                        {"schema": "public"},
                    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    ticker: Mapped[str] = mapped_column(
        String(25),
        index=True,
        nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    sl: Mapped[float] = mapped_column(Float, nullable=False)
    target: Mapped[float] = mapped_column(Float, nullable=False)
    rsi: Mapped[float] = mapped_column(Float, nullable=False)
    atr_percent: Mapped[float] = mapped_column(Float, nullable=False)
    move_30d_percent: Mapped[float] = mapped_column(Float, nullable=False)
    pullback_percent: Mapped[float] = mapped_column(Float, nullable=False)
    ema20_distance_percent: Mapped[float] = mapped_column(Float, nullable=False)
    days_since_60d_peak: Mapped[int] = mapped_column(Integer, nullable=False)
    volume_spike: Mapped[float] = mapped_column(Float, nullable=False)
    run_datetime: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    server_default=func.now(),
    nullable=False
)