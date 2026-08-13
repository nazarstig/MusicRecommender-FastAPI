FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for pyodbc and SQL Server ODBC driver
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       unixodbc-dev \
       curl \
       gnupg \
       apt-transport-https \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
