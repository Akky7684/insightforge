"""Locust Load Testing Suite for InsightForge FastAPI Backend.

Simulates multi-user concurrent traffic across 3 user persona tiers:
1. CasualAnalystUser (Weight: 4) — Checks health, lists datasets, profiles datasets.
2. PowerAnalystUser (Weight: 2) — Runs automated EDA reports and Auto-ML predictive modeling.
3. ConversationalUser (Weight: 3) — Sends natural-language analytical questions.
"""

from locust import HttpUser, between, task


class CasualAnalystUser(HttpUser):
    """Simulates casual users browsing data, inspecting schemas and deep profiles."""
    weight = 4
    wait_time = between(1, 3)

    @task(3)
    def check_health(self):
        """Ping API health endpoint."""
        self.client.get("/health", name="[GET] /health")

    @task(2)
    def list_sample_datasets(self):
        """List bundled evaluation datasets."""
        self.client.get("/api/sample-datasets", name="[GET] /api/sample-datasets")

    @task(3)
    def get_titanic_profile(self):
        """Request cached statistical profile of Titanic dataset."""
        self.client.get("/api/profile?dataset_path=data/titanic.csv", name="[GET] /api/profile?dataset=titanic")

    @task(1)
    def get_superstore_profile(self):
        """Request cached statistical profile of Superstore dataset."""
        self.client.get("/api/profile?dataset_path=data/superstore.csv", name="[GET] /api/profile?dataset=superstore")


class PowerAnalystUser(HttpUser):
    """Simulates power users running 1-Click Executive EDA and Auto-ML models."""
    weight = 2
    wait_time = between(2, 5)

    @task(2)
    def trigger_eda_titanic(self):
        """Execute automated EDA scan on Titanic dataset."""
        self.client.post("/api/eda/generate?dataset_path=data/titanic.csv", name="[POST] /api/eda/generate (titanic)")

    @task(1)
    def trigger_predictive_titanic(self):
        """Train Auto-ML predictive model on Titanic Survived target."""
        self.client.post(
            "/api/predictive/train?dataset_path=data/titanic.csv&target_column=Survived&model_type=random_forest",
            name="[POST] /api/predictive/train (titanic)",
        )


class ConversationalUser(HttpUser):
    """Simulates users sending natural language analytical queries."""
    weight = 3
    wait_time = between(2, 6)

    @task(2)
    def ask_factual_query(self):
        """Submit natural-language factual query."""
        payload = {
            "message": "What is the average age of passengers on the Titanic?",
            "dataset_path": "data/titanic.csv",
            "session_id": "locust-load-session",
        }
        self.client.post("/api/chat", json=payload, name="[POST] /api/chat (factual)")

    @task(1)
    def ask_superstore_sales(self):
        """Submit aggregate sales query on superstore."""
        payload = {
            "message": "What is the total sales amount across all orders in superstore.csv?",
            "dataset_path": "data/superstore.csv",
            "session_id": "locust-load-session",
        }
        self.client.post("/api/chat", json=payload, name="[POST] /api/chat (aggregate)")
