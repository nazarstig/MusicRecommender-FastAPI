from sqlalchemy import Column, String, ForeignKey
from app.core.database import Base


class Song(Base):
    """
    Song model for dbo.Songs table
    """
    __tablename__ = "Songs"
    __table_args__ = {'schema': 'dbo'}
    
    TrackId = Column(String(50), primary_key=True)
    ArtistId = Column(String(50), ForeignKey('dbo.Artists.ArtistId'), nullable=False)
    TrackName = Column(String(500), nullable=False)