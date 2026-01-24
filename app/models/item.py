from sqlalchemy import Column, Integer, String, Text
from app.core.database import Base


class Item(Base):
    """
    Example Item model - replace with your actual LastFm tables
    """
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
