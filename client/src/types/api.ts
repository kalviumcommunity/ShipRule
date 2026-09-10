export interface QueryRequest {
  question: string;
  use_reranking?: boolean;
  final_k?: number;
}

export interface Source {
  source: string;
  chunk_id?: string;
  score?: number;
}

export interface TokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export interface TimingMetrics {
  embedding_ms?: number;
  retrieval_ms?: number;
  reranking_ms?: number;
  context_assembly_ms?: number;
  generation_ms?: number;
  total_pipeline_time_ms?: number;
}

export interface QueryMetadata {
  token_usage?: TokenUsage;
  timing?: TimingMetrics;
  guardrail_decision?: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  status: 'SUPPORTED' | 'INSUFFICIENT_CONTEXT' | 'OUT_OF_SCOPE' | 'SECURITY_BLOCKED' | 'answered' | 'refused' | string;
  metadata?: QueryMetadata;
}

export interface IndexingSummary {
  document: string;
  chunks: number;
  indexed: number;
}

export interface DocumentUploadResponse {
  status: string;
  filename: string;
  summary: IndexingSummary;
}

export interface DocumentItem {
  filename: string;
  stored_filename: string;
  size_bytes: number;
  created_at: string;
  document_type: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  environment: {
    embedding_model_configured: boolean;
    chat_model_configured: boolean;
    vector_db_path: string;
    collection_name: string;
    max_upload_size_mb: number;
  };
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  role: 'user' | 'admin';
  email: string;
}

export interface AdminSettings {
  default_top_k: number;
  max_upload_size_mb: number;
  embedding_model: string;
  chat_model: string;
  collection_name: string;
}
