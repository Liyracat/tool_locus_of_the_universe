# tool_locus_of_the_universe

## Backend (FastAPI)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

## Frontend (React + Vite)

```powershell
cd frontend
npm install
npm run dev
```

API base URL defaults to `http://localhost:8000`. Override with `VITE_API_BASE` if needed.
