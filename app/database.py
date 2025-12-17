# app/database.py (The clean version)

from sqlmodel import Session, create_engine
from typing import Generator
# You will likely need to import the models here if create_db_and_tables is used.
# from .models import SQLModel.metadata # or similar

# Database Connection Setup
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

# Dependency Function
def get_session() -> Generator[Session, None, None]:
    """Provides a database session for a single request."""
    with Session(engine) as session:
        yield session

# Note: The function create_db_and_tables() should also be defined here or in models.py
# If it's defined elsewhere, ensure you import it in main.py.