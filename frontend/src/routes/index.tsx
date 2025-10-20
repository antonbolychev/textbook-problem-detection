import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Input,
  Spinner,
  Text,
  VStack
} from '@chakra-ui/react';
import { ChangeEvent, useCallback, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { createFileRoute, useNavigate } from '@tanstack/react-router';
import { ApiError, DefaultService, type JobStatusResponse, type JobCreatedResponse } from '../client';
import { formatTimestamp } from '../lib/jobs';

const COMPLETED_REFRESH_INTERVAL_MS = 30_000;

export const Route = createFileRoute('/')({
  component: HomeRoute
});

function HomeRoute(): JSX.Element {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const completedJobsQuery = useQuery<JobStatusResponse[]>({
    queryKey: ['jobs', 'completed'],
    queryFn: async () => await DefaultService.listJobStatusesApiJobsGet({ status: 'completed', limit: 100 }),
    refetchInterval: COMPLETED_REFRESH_INTERVAL_MS
  });

  const uploadMutation = useMutation<JobCreatedResponse, unknown, { file: File }>({
    mutationFn: async ({ file }) =>
      await DefaultService.submitJobApiJobsPost({
        formData: { file }
      }),
    onSuccess: (createdJob: JobCreatedResponse) => {
      if (!createdJob.job_id) {
        throw new Error('Backend response missing job identifier.');
      }
      queryClient.setQueryData<JobStatusResponse>(['job-status', createdJob.job_id], createdJob as JobStatusResponse);
      queryClient.invalidateQueries({ queryKey: ['jobs', 'completed'] }).catch(() => {
        /* ignore */
      });
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      console.log('Upload queued - navigating to job page');
      navigate({
        to: '/job/$jobId',
        params: { jobId: createdJob.job_id }
      });
    },
    onError: (error: unknown) => {
      console.error('Upload failed', error);
      const description =
        error instanceof ApiError
          ? `Upload failed with status ${error.status}.`
          : error instanceof Error
          ? error.message
          : 'Ensure the backend is reachable and try again.';
      console.error('Upload failed:', description);
    }
  });

  const handleFileChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    setSelectedFile(files && files.length > 0 ? files[0] : null);
  }, []);

  const handleUpload = useCallback(() => {
    if (!selectedFile) {
      console.info('Please select a PDF file first');
      return;
    }
    uploadMutation.mutate({ file: selectedFile });
  }, [selectedFile, uploadMutation]);

  const completedJobs = completedJobsQuery.data ?? [];

  return (
    <Box maxW="1200px" mx="auto">
      <VStack gap={10} align="stretch">
        {/* Hero Section */}
        <Box textAlign="center" py={8}>
          <Heading size="2xl" mb={3} color="blue.600">
            Textbook Problem Detection
          </Heading>
          <Text fontSize="lg" color="gray.600">
            Upload a textbook PDF to automatically detect and analyze problem statements
          </Text>
        </Box>

        {/* Upload Section */}
        <Box 
          p={8} 
          bg="white" 
          rounded="xl" 
          shadow="lg" 
          borderWidth="1px" 
          borderColor="gray.200"
        >
          <VStack align="stretch" gap={6}>
            <Box>
              <Heading size="lg" mb={2} color="blue.600">
                📤 Upload a PDF
              </Heading>
              <Text color="gray.600" fontSize="md">
                We will process the document asynchronously and notify you when it is ready.
              </Text>
            </Box>
            
            <VStack align="stretch" gap={4}>
              <Input 
                type="file" 
                accept="application/pdf" 
                onChange={handleFileChange} 
                ref={fileInputRef}
                size="lg"
                p={3}
                borderWidth="2px"
                borderStyle="dashed"
                borderColor={selectedFile ? "blue.300" : "gray.300"}
                _hover={{ borderColor: "blue.400" }}
                cursor="pointer"
              />
              
              {selectedFile && (
                <Box 
                  p={3} 
                  bg="blue.50" 
                  rounded="md" 
                  borderWidth="1px" 
                  borderColor="blue.200"
                >
                  <Text fontSize="sm" fontWeight="medium" color="blue.700">
                    ✓ Selected: {selectedFile.name}
                  </Text>
                </Box>
              )}
              
              <Button
                colorPalette="blue"
                onClick={handleUpload}
                loading={uploadMutation.isPending}
                loadingText="Uploading..."
                size="lg"
                fontSize="md"
                py={6}
              >
                Upload and Process
              </Button>
            </VStack>
          </VStack>
        </Box>

        {/* Recent Jobs Section */}
        <Box 
          p={8} 
          bg="white" 
          rounded="xl" 
          shadow="lg" 
          borderWidth="1px" 
          borderColor="gray.200"
        >
          <VStack align="stretch" gap={6}>
            <Heading size="lg" color="blue.600">
              📋 Recent Completed Jobs
            </Heading>
            
            {completedJobsQuery.isLoading ? (
              <Flex align="center" justify="center" gap={3} py={8}>
                <Spinner size="md" color="blue.500" />
                <Text color="gray.600" fontSize="md">Loading completed jobs…</Text>
              </Flex>
            ) : completedJobsQuery.isError ? (
              <Box p={6} bg="red.50" rounded="md" borderWidth="1px" borderColor="red.200">
                <Text color="red.600" fontSize="md">Unable to load completed jobs. Try again shortly.</Text>
              </Box>
            ) : completedJobs.length === 0 ? (
              <Box p={8} bg="gray.50" rounded="md" textAlign="center">
                <Text color="gray.500" fontSize="md">
                  No completed jobs yet. Upload a PDF to get started! 🚀
                </Text>
              </Box>
            ) : (
              <VStack align="stretch" gap={3}>
                {completedJobs
                  .filter((job): job is JobStatusResponse & { job_id: string } => typeof job.job_id === 'string')
                  .map((job) => {
                    const label = job.filename && job.filename.length > 0 ? job.filename : `Job ${job.job_id}`;
                    return (
                      <Button
                        key={job.job_id}
                        variant="outline"
                        colorPalette="blue"
                        onClick={() =>
                          navigate({
                            to: '/job/$jobId',
                            params: { jobId: job.job_id }
                          })
                        }
                        justifyContent="flex-start"
                        alignItems="flex-start"
                        textAlign="left"
                        w="full"
                        h="auto"
                        py={4}
                        px={5}
                        _hover={{ 
                          bg: 'blue.50', 
                          borderColor: 'blue.400',
                          transform: 'translateX(4px)',
                          transition: 'all 0.2s'
                        }}
                      >
                        <VStack align="start" gap={1.5}>
                          <Text fontWeight="bold" fontSize="md" color="blue.700">
                            📄 {label}
                          </Text>
                          <Text fontSize="sm" color="gray.500">
                            Updated {formatTimestamp(job.updated_at ?? job.created_at ?? null)}
                          </Text>
                          <Text fontSize="xs" color="gray.400" fontFamily="monospace">
                            ID: {job.job_id.slice(0, 8)}...
                          </Text>
                        </VStack>
                      </Button>
                    );
                  })}
              </VStack>
            )}
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
}
