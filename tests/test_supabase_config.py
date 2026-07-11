import unittest

from db import normalize_supabase_url, build_supabase_storage_url


class SupabaseConfigTests(unittest.TestCase):
    def test_normalize_dashboard_url(self):
        self.assertEqual(
            normalize_supabase_url("https://supabase.com/dashboard/project/ywsadpvfvtpxxwgqnyon"),
            "https://ywsadpvfvtpxxwgqnyon.supabase.co",
        )

    def test_storage_url_uses_project_base(self):
        self.assertEqual(
            build_supabase_storage_url(
                "https://supabase.com/dashboard/project/ywsadpvfvtpxxwgqnyon",
                "recordings/example.ogg",
            ),
            "https://ywsadpvfvtpxxwgqnyon.supabase.co/storage/v1/object/public/recordings/example.ogg",
        )


if __name__ == "__main__":
    unittest.main()
