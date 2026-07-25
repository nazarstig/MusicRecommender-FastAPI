import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.core.config import settings
from app.core.database import engine, SessionLocal
from app.core.dependancies import get_recommendation_service
from app.models import item
from sqlalchemy import text

# Create database tables
# item.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Test DB connection and warm up recommendation matrices on startup."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            print("✓ Database connection successful!")

        db = SessionLocal()
        try:
            recommendation_service = get_recommendation_service()
            filepath = "recommendations_manual.npz"
            if not os.path.exists(filepath):
                recommendation_service.create_recommendations(db)
            else:
                recommendation_service.load_prediction_matrices(filepath, db)
            print("✓ Recommendation matrices initialized")
        finally:
            db.close()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

# Include routers
app.include_router(routes.router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Welcome to FastAPI",
        "docs": "/docs",
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
