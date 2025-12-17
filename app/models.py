from sqlmodel import SQLModel, Field, Session, create_engine
from datetime import datetime

# app/models.py

from sqlmodel import SQLModel, Field
from datetime import datetime
from pydantic import BaseModel  # <--- 1. Add this import

# ... (Existing PredictionHistory class) ...
class PredictionHistory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    input_text: str
    input_topic: str | None
    prediction_label: str
    timestamp: datetime = Field(default_factory=datetime.utcnow, nullable=False)

# ... (Add this class to the bottom of the file) ...

# 2. Define the Request Body Model here
class NewsRequest(BaseModel):
    text: str



# 2. Database Connection Setup
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

def create_db_and_tables():
    """Initializes the database file and creates the table if it doesn't exist."""
    SQLModel.metadata.create_all(engine)