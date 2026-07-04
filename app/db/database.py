from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

def get_db():
    SessionLocal = sessionmaker(
        autoflush=False,
        autocommit=False,
        bind=engine
    )
    db=SessionLocal()
    try:
        yield  db
    finally:
        db.close()