import type { JobStatus, JobStatusResponse } from '../client';

export type ProblemAssignment = {
  page: number;
  problems: {
    problem_id: string;
    line_indices: number[];
    lines: { index: number; text: string; matches_problem_text: boolean | null }[];
    full_text: string;
    bbox?: { x: number; y: number; width: number; height: number } | null;
  }[];
};

export type OCRLine = {
  bbox: [number, number, number, number];
  text: string;
};

export type OCRPage = {
  page: number;
  text_lines: OCRLine[];
};

export type OcrResponse = {
  pages?: Array<{
    page: number;
    text_lines?: Array<{
      bbox?: [number, number, number, number];
      text?: string;
    }>;
  }>;
};

export const STATUS_LABELS: Record<JobStatus, string> = {
  queued: 'Queued for processing',
  processing: 'Processing PDF',
  ocr_running: 'Running OCR',
  ocr_complete: 'OCR complete',
  pipeline_running: 'Running extraction pipeline',
  completed: 'Finished',
  failed: 'Failed',
  unknown: 'Idle'
};

export const formatTimestamp = (value: string | null | undefined): string => {
  if (!value) {
    return 'Unknown';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
};

export const isProblemAssignmentList = (value: unknown): value is ProblemAssignment[] => {
  if (!Array.isArray(value)) {
    return false;
  }
  return value.every((entry) => {
    if (!entry || typeof entry !== 'object') {
      return false;
    }
    const typed = entry as Partial<ProblemAssignment>;
    return typeof typed.page === 'number' && Array.isArray(typed.problems);
  });
};

export const normaliseOcrResponse = (response: OcrResponse): OCRPage[] =>
  (response.pages ?? []).map((page) => ({
    page: page.page,
    text_lines: (page.text_lines ?? []).map((line) => ({
      bbox: line.bbox ?? [0, 0, 0, 0],
      text: line.text ?? ''
    }))
  }));

export const isJobStatusResponse = (value: unknown): value is JobStatusResponse => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<JobStatusResponse>;
  if ('job_id' in candidate && candidate.job_id != null && typeof candidate.job_id !== 'string') {
    return false;
  }
  if ('status' in candidate && candidate.status != null && typeof candidate.status !== 'string') {
    return false;
  }
  if ('updated_at' in candidate && candidate.updated_at != null && typeof candidate.updated_at !== 'string') {
    return false;
  }
  return true;
};
