import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Spinner,
  Text,
  VStack,
  Input,
  Link
} from '@chakra-ui/react';
import { HiChevronLeft, HiChevronRight } from 'react-icons/hi';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { createFileRoute, useNavigate } from '@tanstack/react-router';
import useWebSocket from 'react-use-websocket';
import { ApiError, DefaultService, type JobStatus, type JobStatusResponse } from '../client';
import { ProblemCanvas } from '../components/ProblemCanvas';
import { buildApiUrl, buildStorageUrl, WS_BASE_URL } from '../lib/api';
import {
  STATUS_LABELS,
  formatTimestamp,
  isProblemAssignmentList,
  isJobStatusResponse,
  normaliseOcrResponse,
  type OcrResponse,
  type OCRPage,
  type ProblemAssignment
} from '../lib/jobs';

type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type PageBoxes = Record<string, BoundingBox>;
type BoxesState = Record<number, PageBoxes>;
type PageAnswers = Record<string, string>;
type AnswersState = Record<number, PageAnswers>;

type JobArtifacts = {
  assignments: ProblemAssignment[] | null;
  ocrPages: OCRPage[] | null;
  imagePaths: string[];
};

export const Route = createFileRoute('/job/$jobId')({
  component: JobRoute
});

function JobRoute(): JSX.Element {
  const { jobId } = Route.useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const notFoundNotifiedRef = useRef(false);
  const websocketErrorNotifiedRef = useRef(false);
  const jobStatusErrorNotifiedRef = useRef(false);
  const jobArtifactsErrorNotifiedRef = useRef(false);
  const previousStatusRef = useRef<JobStatus | null>(null);
  const [boxes, setBoxes] = useState<BoxesState>({});
  const [answers, setAnswers] = useState<AnswersState>({});
  const [expandedComments, setExpandedComments] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const [pageInput, setPageInput] = useState('1');

  useEffect(() => {
    setBoxes({});
    setAnswers({});
    setExpandedComments(new Set());
    setCurrentPage(1);
    setPageInput('1');
  }, [jobId]);

  useEffect(() => {
    notFoundNotifiedRef.current = false;
    websocketErrorNotifiedRef.current = false;
    jobStatusErrorNotifiedRef.current = false;
    jobArtifactsErrorNotifiedRef.current = false;
  }, [jobId]);

  const jobStatusQuery = useQuery<JobStatusResponse>({
    queryKey: ['job-status', jobId],
    queryFn: async () => {
      try {
        return await DefaultService.jobStatusApiJobsJobIdGet({ jobId });
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          if (!notFoundNotifiedRef.current) {
            notFoundNotifiedRef.current = true;
            // Toast: Job not found
            console.warn('Job not found');
          }
          navigate({ to: '/' });
        }
        throw error;
      }
    },
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 404) {
        return false;
      }
      return failureCount < 3;
    },
    refetchInterval: (query) => {
      const data = query.state.data;
      return data && (data.status === 'completed' || data.status === 'failed') ? false : 5000;
    }
  });

  const jobStatus = jobStatusQuery.data ?? null;
  const jobExtra = jobStatus?.extra;

  const ocrPath =
    jobExtra && typeof jobExtra.ocr_path === 'string' && jobExtra.ocr_path.length > 0 ? jobExtra.ocr_path : null;
  const extraImagePaths = Array.isArray(jobExtra?.image_paths)
    ? jobExtra.image_paths.filter((entry): entry is string => typeof entry === 'string' && entry.length > 0)
    : [];

  const jobArtifactsQuery = useQuery<JobArtifacts>({
    queryKey: ['job-artifacts', jobId, ocrPath, extraImagePaths.join('|')],
    enabled: jobStatus?.status === 'completed',
    staleTime: 60_000,
    queryFn: async (): Promise<JobArtifacts> => {
      const resultResponse = await DefaultService.jobResultApiJobsJobIdResultGet({ jobId });
      const assignments = isProblemAssignmentList(resultResponse.result) ? resultResponse.result : null;

      let ocrPages: OCRPage[] | null = null;
      if (ocrPath) {
        try {
          const response = await fetch(buildStorageUrl(ocrPath));
          if (!response.ok) {
            throw new Error(`Failed to fetch OCR artefacts (${response.status})`);
          }
          const json = (await response.json()) as OcrResponse;
          ocrPages = normaliseOcrResponse(json);
        } catch (error) {
          console.error('Unable to fetch OCR artefacts', error);
          ocrPages = null;
        }
      }

      return {
        assignments,
        ocrPages,
        imagePaths: extraImagePaths
      };
    }
  });

  useEffect(() => {
    const status = jobStatus?.status ?? null;
    const previous = previousStatusRef.current;
    if (status !== previous) {
      previousStatusRef.current = status;
      if (status === 'completed') {
        queryClient.invalidateQueries({ queryKey: ['jobs', 'completed'] }).catch(() => {
          /* ignore */
        });
      }
    }
  }, [jobStatus?.status, queryClient]);

  const websocketUrl = jobId ? `${WS_BASE_URL.replace(/\/$/, '')}/ws/jobs/${jobId}` : null;

  useWebSocket<JobStatusResponse>(websocketUrl, {
    shouldReconnect: () => true,
    retryOnError: true,
    share: false,
    onOpen: () => {
      websocketErrorNotifiedRef.current = false;
    },
    onError: (event) => {
      console.error('Websocket error', event);
      if (websocketErrorNotifiedRef.current) {
        return;
      }
      websocketErrorNotifiedRef.current = true;
      // Toast: Websocket error
      console.warn('Websocket error');
    },
    onMessage: (event) => {
      try {
        if (typeof event.data !== 'string') {
          throw new Error('Websocket payload is not a string');
        }
        const payload = JSON.parse(event.data) as unknown;
        if (!isJobStatusResponse(payload)) {
          throw new Error('Unexpected websocket payload');
        }
        queryClient.setQueryData<JobStatusResponse>(['job-status', jobId], payload);
      } catch (error) {
        console.error('Failed to handle websocket payload', error);
      }
    }
  }, Boolean(jobId));

  useEffect(() => {
    if (!jobStatusQuery.isError) {
      jobStatusErrorNotifiedRef.current = false;
      return;
    }
    const error = jobStatusQuery.error;
    if (error instanceof ApiError && error.status === 404) {
      return;
    }
    if (jobStatusErrorNotifiedRef.current) {
      return;
    }
    jobStatusErrorNotifiedRef.current = true;
    console.error('Failed to load job status', error);
    // Toast: Unable to load job status
    console.error('Unable to load job status');
  }, [jobStatusQuery.error, jobStatusQuery.isError]);

  useEffect(() => {
    if (!jobArtifactsQuery.isError) {
      jobArtifactsErrorNotifiedRef.current = false;
      return;
    }
    if (jobArtifactsErrorNotifiedRef.current) {
      return;
    }
    jobArtifactsErrorNotifiedRef.current = true;
    console.error('Unable to fetch job artefacts', jobArtifactsQuery.error);
    // Toast: Failed to load job outputs
    console.error('Failed to load job outputs');
  }, [jobArtifactsQuery.error, jobArtifactsQuery.isError]);

  useEffect(() => {
    const assignments = jobArtifactsQuery.data?.assignments ?? null;
    const ocrPages = jobArtifactsQuery.data?.ocrPages ?? null;
    if (!assignments) {
      setBoxes({});
      return;
    }
    const nextBoxes: BoxesState = {};
    for (const page of assignments) {
      const pageBoxes: PageBoxes = {};
      for (const problem of page.problems) {
        // Use bbox from the API response if available
        if (problem.bbox && problem.bbox.x != null && problem.bbox.y != null && problem.bbox.width != null && problem.bbox.height != null) {
          pageBoxes[problem.problem_id] = {
            x: problem.bbox.x,
            y: problem.bbox.y,
            width: problem.bbox.width,
            height: problem.bbox.height
          };
          continue;
        }
        
        // Fallback: calculate from OCR lines if bbox is not available
        if (!ocrPages) {
          continue;
        }
        const ocrPage = ocrPages.find((entry) => entry.page === page.page);
        if (!ocrPage) {
          continue;
        }
        const bboxes = problem.line_indices
          .map((index) => ocrPage.text_lines[index])
          .filter(Boolean)
          .map((line) => line.bbox ?? [0, 0, 0, 0]);
        if (!bboxes.length) {
          continue;
        }
        const [minX, minY, maxX, maxY] = bboxes.reduce<[number, number, number, number]>(
          (acc, bbox) => [
            Math.min(acc[0], bbox[0]),
            Math.min(acc[1], bbox[1]),
            Math.max(acc[2], bbox[2]),
            Math.max(acc[3], bbox[3])
          ],
          [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0, 0]
        );
        if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
          continue;
        }
        pageBoxes[problem.problem_id] = {
          x: minX,
          y: minY,
          width: maxX - minX,
          height: maxY - minY
        };
      }
      nextBoxes[page.page] = pageBoxes;
    }
    setBoxes(nextBoxes);
  }, [jobArtifactsQuery.data?.assignments, jobArtifactsQuery.data?.ocrPages]);

  const updateBoundingBox = (pageNumber: number, problemId: string, box: BoundingBox) => {
    setBoxes((prev) => {
      const pageBoxes = { ...(prev[pageNumber] ?? {}) };
      pageBoxes[problemId] = box;
      return {
        ...prev,
        [pageNumber]: pageBoxes
      };
    });
  };

  const updateAnswer = (pageNumber: number, problemId: string, answer: string) => {
    setAnswers((prev) => {
      const pageAnswers = { ...(prev[pageNumber] ?? {}) };
      pageAnswers[problemId] = answer;
      return {
        ...prev,
        [pageNumber]: pageAnswers
      };
    });
  };

  const toggleComment = (problemId: string) => {
    setExpandedComments((prev) => {
      const next = new Set(prev);
      if (next.has(problemId)) {
        next.delete(problemId);
      } else {
        next.add(problemId);
      }
      return next;
    });
  };

  const statusLabel = useMemo(() => {
    if (!jobStatus?.status) {
      return 'Idle';
    }
    return STATUS_LABELS[jobStatus.status] ?? jobStatus.status;
  }, [jobStatus?.status]);

  const jobArtifacts = jobArtifactsQuery.data ?? null;
  const jobResult = jobArtifacts?.assignments ?? null;
  const imagePaths = jobArtifacts?.imagePaths ?? [];

  const hasDetectedProblems = useMemo(() => {
    if (!jobResult) {
      return false;
    }
    return jobResult.some((page) => page.problems.length > 0);
  }, [jobResult]);

  const totalPages = jobResult?.length ?? 0;
  const currentPageData = jobResult?.find((p) => p.page === currentPage);

  const processedVisual =
    jobExtra?.problem_visualisations?.find((entry) => typeof entry === 'string' && entry.length > 0) ?? undefined;

  const downloadablePdf = jobStatus?.job_id ? buildApiUrl(`/api/jobs/${jobStatus.job_id}/download`) : undefined;

  return (
    <Box maxW="900px" mx="auto" w="full" p={8} bg="white" rounded="lg" shadow="md">
      <VStack gap={8} align="stretch">
        <Button
          alignSelf="flex-start"
          variant="ghost"
          colorScheme="blue"
          onClick={() => navigate({ to: '/' })}
        >
          ← Back to uploads
        </Button>

        <Box>
          <Heading size="lg">Textbook Problem Detection</Heading>
          <Text color="gray.600">Upload a textbook PDF to detect problem statements automatically.</Text>
        </Box>

        <Box>
        <Heading size="md" mb={4}>
          Job status
        </Heading>
        {jobStatusQuery.isLoading ? (
          <Flex align="center" gap={3}>
            <Spinner size="sm" />
            <Text color="gray.600">Loading job status…</Text>
          </Flex>
        ) : jobStatusQuery.isError ? (
          <Text color="red.500">Unable to load job status. Try again shortly.</Text>
        ) : !jobStatus ? (
          <Text color="gray.600">Job details unavailable.</Text>
        ) : (
          <VStack align="stretch" gap={3}>
            <Box>
              <Text fontSize="sm" color="gray.600">
                Job ID
              </Text>
              <Text fontWeight="semibold">{jobStatus.job_id}</Text>
            </Box>
            {jobStatus.filename && (
              <Box>
                <Text fontSize="sm" color="gray.600">
                  Filename
                </Text>
                <Text fontWeight="semibold">{jobStatus.filename}</Text>
              </Box>
            )}
            <HStack justify="space-between">
              <Text>Status</Text>
              <Text fontWeight="semibold">{statusLabel}</Text>
            </HStack>
            <Text fontSize="sm" color="gray.600">
              Updated {formatTimestamp(jobStatus.updated_at ?? jobStatus.created_at)}
            </Text>
            {jobStatus.status !== 'completed' && jobStatus.status !== 'failed' && (
              <Spinner size="sm" />
            )}
            {jobStatus.message && (
              <Text color="gray.600" fontSize="sm">
                {jobStatus.message}
              </Text>
            )}
            {downloadablePdf && (
              <Link href={downloadablePdf} target="_blank" rel="noreferrer">
                <Button variant="outline" size="sm">
                  Download original PDF
                </Button>
              </Link>
            )}
            {jobStatus.status === 'failed' && (
              <Box p={4} bg="red.50" borderWidth="1px" borderColor="red.200" rounded="md">
                <Heading size="sm" color="red.600" mb={1}>
                  Processing failed
                </Heading>
                <Text color="red.500">{jobStatus.message ?? 'Please review server logs for more details.'}</Text>
              </Box>
            )}
          </VStack>
        )}
      </Box>

      {jobStatus?.status === 'completed' && (
        <Box>
          <Heading size="md" mb={4}>
            Detected problems
          </Heading>
          {(processedVisual || downloadablePdf) && (
            <HStack mb={4} gap={3} flexWrap="wrap">
              {processedVisual && (
                <Link href={buildStorageUrl(processedVisual)} target="_blank" rel="noreferrer">
                  <Button colorPalette="blue" variant="outline">
                    View processed document
                  </Button>
                </Link>
              )}
              {downloadablePdf && (
                <Link href={downloadablePdf} target="_blank" rel="noreferrer">
                  <Button variant="ghost">
                    Download original PDF
                  </Button>
                </Link>
              )}
            </HStack>
          )}
          {jobArtifactsQuery.isLoading || jobResult === null ? (
            <Flex align="center" gap={3}>
              <Spinner size="sm" />
              <Text color="gray.600">Loading detected problems…</Text>
            </Flex>
          ) : jobArtifactsQuery.isError ? (
            <Text color="red.500">Unable to load detected problems. Try again shortly.</Text>
          ) : !hasDetectedProblems ? (
            <Text color="gray.600">No problem statements were detected for this document.</Text>
          ) : (
            <VStack gap={4} align="stretch">
                <HStack justify="center" align="center" gap={4}>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      if (currentPage > 1) {
                        setCurrentPage(currentPage - 1);
                        setPageInput(String(currentPage - 1));
                      }
                    }}
                    disabled={currentPage === 1}
                  >
                    <HiChevronLeft /> Назад
                  </Button>
                  
                  <HStack gap={2}>
                    <Text fontSize="sm">Страница</Text>
                    <Input
                      value={pageInput}
                      onChange={(e) => {
                        setPageInput(e.target.value);
                        const num = parseInt(e.target.value, 10);
                        if (!isNaN(num) && num >= 1 && num <= totalPages) {
                          setCurrentPage(num);
                        }
                      }}
                      onBlur={() => setPageInput(String(currentPage))}
                      size="sm"
                      width="60px"
                      textAlign="center"
                    />
                    <Text fontSize="sm">из {totalPages}</Text>
                  </HStack>
                  
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      if (currentPage < totalPages) {
                        setCurrentPage(currentPage + 1);
                        setPageInput(String(currentPage + 1));
                      }
                    }}
                    disabled={currentPage === totalPages}
                  >
                    Вперед <HiChevronRight />
                  </Button>
                </HStack>
              
              {(() => {
                if (!currentPageData) {
                  return (
                    <Flex align="center" justify="center" h="300px" bg="gray.100" rounded="md">
                      <Text color="gray.600">Page not found</Text>
                    </Flex>
                  );
                }
                const imageForPage = imagePaths.find((path) => path.includes(String(currentPage).padStart(3, '0')));
                if (!imageForPage) {
                  return (
                    <Flex align="center" justify="center" h="300px" bg="gray.100" rounded="md">
                      <Spinner />
                    </Flex>
                  );
                }
                return (
                  <ProblemCanvas
                    key={`page-${currentPage}-${imageForPage}`}
                    imageUrl={buildStorageUrl(imageForPage)}
                    problems={currentPageData.problems}
                    boxes={boxes[currentPage] ?? {}}
                    onBoxChange={(problemId, box) => updateBoundingBox(currentPage, problemId, box)}
                    answers={answers[currentPage] ?? {}}
                    onAnswerChange={(problemId, answer) => updateAnswer(currentPage, problemId, answer)}
                    expandedComments={expandedComments}
                    onCommentToggle={toggleComment}
                  />
                );
              })()}
            </VStack>
          )}
        </Box>
      )}
      </VStack>
    </Box>
  );
}
