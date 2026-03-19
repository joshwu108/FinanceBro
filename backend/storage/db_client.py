from supabase import create_client

class DBClient:
    def __init__(self):
        self.client = create_client(
            "SUPABASE_URL",
            "SUPABASE_KEY"
        )

    def insert_experiment(self, data):
        return self.client.table("experiments").insert(data).execute()

    def get_experiments(self):
        return self.client.table("experiments").select("*").execute()