'use client';

import React, { useEffect, useState } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Clock,
  HardDrive,
  RefreshCw,
  FileType
} from 'lucide-react';
import { apiService, ApiError } from '@/services/api';
import { DocumentItem, DocumentUploadResponse } from '@/types/api';
import { formatBytes } from '@/lib/utils';

export function DocumentManager() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<DocumentUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const fetchDocs = async () => {
    setLoading(true);
    try {
      const list = await apiService.getDocuments();
      setDocuments(list);
    } catch {
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setError(null);
      setUploadSuccess(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setError(null);
    setUploadSuccess(null);

    try {
      const res = await apiService.uploadDocument(selectedFile);
      setUploadSuccess(res);
      setSelectedFile(null);
      fetchDocs();
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(err.message || 'Failed to upload and index document.');
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-5xl space-y-8">
      {/* Page Title Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <FileText className="h-6 w-6 text-cyan-400" />
          Document Management & Vector Ingestion
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload and index customs tariffs, regulatory policies, maritime shipping guidelines (.txt, .md, .pdf).
        </p>
      </div>

      {/* Upload Box */}
      <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/50 p-8 text-center transition hover:border-slate-600">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-950/60 border border-cyan-800 text-cyan-400 mb-4 shadow-lg shadow-cyan-500/10">
          <UploadCloud className="h-7 w-7" />
        </div>
        <h3 className="font-semibold text-white text-base">Select Document to Ingest</h3>
        <p className="mt-1 text-xs text-slate-400 max-w-md mx-auto">
          Supported formats: <strong className="text-slate-300">.txt, .md, .pdf</strong> (Max limit: 10MB). Uploaded documents are immediately indexed for RAG queries.
        </p>

        <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-4">
          <label className="cursor-pointer rounded-xl border border-slate-700 bg-slate-800 px-4 py-2.5 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white">
            Choose File
            <input
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>

          {selectedFile && (
            <div className="flex items-center gap-2 text-xs text-cyan-300 font-mono bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
              <FileType className="h-4 w-4 text-cyan-400" />
              <span>{selectedFile.name} ({formatBytes(selectedFile.size)})</span>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-5 py-2.5 text-xs font-semibold text-white transition hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-cyan-500/20"
          >
            {uploading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                Indexing Document...
              </>
            ) : (
              <>
                <UploadCloud className="h-4 w-4" />
                Start Ingestion
              </>
            )}
          </button>
        </div>
      </div>

      {/* Success Notification */}
      {uploadSuccess && (
        <div className="rounded-xl border border-emerald-800/80 bg-emerald-950/30 p-4 space-y-2 text-emerald-300">
          <div className="flex items-center gap-2 font-semibold text-emerald-200">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            Document Successfully Indexed!
          </div>
          <div className="text-xs space-y-1 font-mono text-emerald-400/90 pl-7">
            <p>Filename: {uploadSuccess.filename}</p>
            <p>Generated Chunks: {uploadSuccess.summary.chunks}</p>
            <p>Vectors Indexed: {uploadSuccess.summary.indexed}</p>
            <p>Stored Path: {uploadSuccess.summary.document}</p>
          </div>
        </div>
      )}

      {/* Error Notification */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-rose-900/50 bg-rose-950/30 p-4 text-rose-300">
          <AlertCircle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-200">Upload / Indexing Error</h4>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Documents List Section */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <HardDrive className="h-4 w-4 text-cyan-400" />
            Uploaded Documents ({documents.length})
          </h3>

          <button
            onClick={fetchDocs}
            disabled={loading}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="py-8 text-center text-xs text-slate-500 animate-pulse">
            Loading document corpus...
          </div>
        ) : documents.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500">
            No uploaded documents found in uploads/ folder. Use the upload tool above to add new files.
          </div>
        ) : (
          <div className="divide-y divide-slate-800">
            {documents.map((doc, idx) => (
              <div key={idx} className="flex items-center justify-between py-3.5 text-xs">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-slate-300">
                    <FileText className="h-4 w-4 text-cyan-400" />
                  </div>
                  <div>
                    <h5 className="font-medium text-slate-200">{doc.filename}</h5>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {doc.stored_filename}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-6 text-slate-400">
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-300 uppercase">
                    .{doc.document_type}
                  </span>
                  <span>{formatBytes(doc.size_bytes)}</span>
                  <span className="flex items-center gap-1 text-[11px] text-slate-500">
                    <Clock className="h-3 w-3" />
                    {new Date(doc.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
