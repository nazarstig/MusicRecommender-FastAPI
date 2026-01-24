from sqlalchemy import Column, Integer, String
from app.core.database import Base


class Artist(Base):
    """
    Artist model for dbo.Artists table
    """
    __tablename__ = "Artists"
    __table_args__ = {'schema': 'dbo'}
    
    ArtistId = Column(String(50), primary_key=True)
    ArtistName = Column(String(500))