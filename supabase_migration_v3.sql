-- ═══════════════════════════════════════════════════════════════════════════════
-- supabase_migration_v3.sql
-- Adds quality_score and improvement_note columns for self-training loop.
-- Run this in: Supabase Dashboard → SQL Editor → New Query → Run
-- ═══════════════════════════════════════════════════════════════════════════════

-- Add quality_score column (0-10 scale)
ALTER TABLE call_logs
  ADD COLUMN IF NOT EXISTS quality_score INTEGER DEFAULT NULL;

-- Add improvement_note column (auto-generated lesson for low-quality calls)
ALTER TABLE call_logs
  ADD COLUMN IF NOT EXISTS improvement_note TEXT DEFAULT '';

-- Index for fast retrieval of recent improvement notes
CREATE INDEX IF NOT EXISTS idx_call_logs_improvement_note
  ON call_logs (created_at DESC)
  WHERE improvement_note IS NOT NULL AND improvement_note != '';

-- Index for quality score analytics
CREATE INDEX IF NOT EXISTS idx_call_logs_quality_score
  ON call_logs (quality_score);

-- Refresh the PostgREST schema cache so the new columns are visible immediately
-- (If you see PGRST204 errors, run this line manually in the SQL editor)
NOTIFY pgrst, 'reload schema';
