# FastAPI Recommender System

A modern, fast (high-performance) web API framework for building APIs with Python 3.8+.

## Project Structure

```
MusicRecommender/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   └── schemas/
│       ├── __init__.py
│       └── item.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file from `.env.example`:
```bash
copy .env.example .env
```

5. Run the application:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Available Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `GET /api/items` - Get all items
- `GET /api/items/{item_id}` - Get item by ID

## Development

Add your business logic in the appropriate modules:
- **Routes**: `app/api/routes.py`
- **Schemas**: `app/schemas/`
- **Configuration**: `app/core/config.py`

## Next Steps

- Add database integration (SQLAlchemy, MongoDB, etc.)
- Implement authentication and authorization
- Add more endpoints for your recommender system
- Write tests
- Add logging and monitoring
