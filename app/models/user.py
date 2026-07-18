from sqlalchemy import Column, String, Integer
from app.core.database import Base


class User(Base):
    """
    User model for dbo.Users table
    """
    __tablename__ = "Users"
    __table_args__ = {'schema': 'dbo'}
    
    Id = Column(String(50, collation='SQL_Latin1_General_CP1_CI_AS'), primary_key=True)
    Gender = Column(String(10, collation='SQL_Latin1_General_CP1_CI_AS'), nullable=True)
    Age = Column(Integer, nullable=True)
    Country = Column(String(100, collation='SQL_Latin1_General_CP1_CI_AS'), nullable=True)
    Registered = Column(String(50, collation='SQL_Latin1_General_CP1_CI_AS'), nullable=True)
