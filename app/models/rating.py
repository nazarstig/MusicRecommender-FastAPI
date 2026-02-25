from sqlalchemy import Column, String, Integer, ForeignKey
from app.core.database import Base


class Rating(Base):
    """
    Rating model for dbo.Ratings table
    """
    __tablename__ = "Ratings"
    __table_args__ = {'schema': 'dbo'}
    
    UserId = Column(String(50, collation='SQL_Latin1_General_CP1_CI_AS'), ForeignKey('dbo.Users.Id'), primary_key=True)
    TrackId = Column(String(50, collation='SQL_Latin1_General_CP1_CI_AS'), ForeignKey('dbo.Songs.TrackId'), primary_key=True)
    Rating = Column(Integer, nullable=False)
