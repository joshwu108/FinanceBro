import os
from dotenv import load_dotenv, find_dotenv
from supabase import create_client

load_dotenv(find_dotenv(usecwd=True))
class DBClient:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file")
            
        self.client = create_client(url, key)

    def insert_experiment(self, data):
        return self.client.table("experiments").insert(data).execute()

    def get_experiments(self):
        return self.client.table("experiments").select("*").execute()