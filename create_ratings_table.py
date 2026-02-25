from app.core.database import engine, Base
from app.models.user import User
from app.models.rating import Rating
from app.models.song import Song
from app.models.artist import Artist

# Create only the tables that don't exist yet
Base.metadata.create_all(bind=engine)
print("✓ Ratings table created successfully!")
