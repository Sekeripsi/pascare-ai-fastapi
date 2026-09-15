/*
# Create llm_metrics table for request logging

1. New Tables
- `llm_metrics`
  - `id` (uuid, primary key, auto-generated)
  - `scenario_id` (text, not null) — unique ID per chat request for correlation
  - `agent` (text, not null) — semicolon-separated list of agents that ran
  - `ttft_s` (double precision, nullable) — time to first token in seconds
  - `total_latency_s` (double precision, nullable) — total request latency in seconds
  - `prompt_tokens` (integer, not null, default 0) — sum of input/prompt tokens across all LLM calls
  - `completion_tokens` (integer, not null, default 0) — sum of output/completion tokens across all LLM calls
  - `total_tokens` (integer, not null, default 0) — sum of total tokens across all LLM calls
  - `llm_calls` (integer, not null, default 0) — number of LLM invocations in the chain
  - `graph_variant` (text, not null) — which graph variant served the request (baseline or multi)
  - `created_at` (timestamptz, default now()) — when the record was logged

2. Security
- Enable RLS on `llm_metrics`.
- Allow anon + authenticated CRUD: this is a server-side logging table written by
  the FastAPI backend using the anon key. No user-facing sign-in exists, so the
  backend is the sole writer and reader. USING (true) is acceptable because the
  data is operational metrics shared across the single backend service.
*/

CREATE TABLE IF NOT EXISTS llm_metrics (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id text NOT NULL,
    agent text NOT NULL DEFAULT '-',
    ttft_s double precision,
    total_latency_s double precision,
    prompt_tokens integer NOT NULL DEFAULT 0,
    completion_tokens integer NOT NULL DEFAULT 0,
    total_tokens integer NOT NULL DEFAULT 0,
    llm_calls integer NOT NULL DEFAULT 0,
    graph_variant text NOT NULL DEFAULT 'baseline',
    created_at timestamptz DEFAULT now()
);

ALTER TABLE llm_metrics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_select_llm_metrics" ON llm_metrics;
CREATE POLICY "anon_select_llm_metrics" ON llm_metrics FOR SELECT
    TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_llm_metrics" ON llm_metrics;
CREATE POLICY "anon_insert_llm_metrics" ON llm_metrics FOR INSERT
    TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_llm_metrics" ON llm_metrics;
CREATE POLICY "anon_update_llm_metrics" ON llm_metrics FOR UPDATE
    TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_llm_metrics" ON llm_metrics;
CREATE POLICY "anon_delete_llm_metrics" ON llm_metrics FOR DELETE
    TO anon, authenticated USING (true);
