import { Box, Button, Input, VStack, Text } from '@chakra-ui/react';
import { useEffect, useRef, useState } from 'react';
import { HiChevronDown, HiChevronUp } from 'react-icons/hi';

type ProblemCommentProps = {
  problemId: string;
  yPosition: number;
  answer: string;
  onAnswerChange: (answer: string) => void;
  isExpanded: boolean;
  onToggle: () => void;
  yOffset?: number;
};

export function ProblemComment({ 
  problemId, 
  yPosition, 
  answer, 
  onAnswerChange,
  isExpanded,
  onToggle,
  yOffset = 0
}: ProblemCommentProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <Box
      ref={containerRef}
      position="absolute"
      left="calc(100% + 16px)"
      top={`${yPosition + yOffset}px`}
      zIndex={200}
      minW="220px"
      maxW="300px"
    >
      <Box
        bg="white"
        border="2px solid"
        borderColor="blue.300"
        rounded="md"
        shadow="md"
      >
        {/* Header - always visible */}
        <Button
          onClick={onToggle}
          size="sm"
          variant="ghost"
          width="full"
          justifyContent="space-between"
          px={3}
          py={2}
        >
          <Text fontSize="sm" fontWeight="semibold">
            Задача {problemId}
          </Text>
          {isExpanded ? <HiChevronUp /> : <HiChevronDown />}
        </Button>

        {/* Expanded content */}
        {isExpanded && (
          <VStack gap={3} p={3} align="stretch" borderTop="1px" borderColor="gray.200">
            <Box>
              <Text fontSize="xs" color="gray.600" mb={1}>
                Ответ:
              </Text>
              <Input
                value={answer}
                onChange={(e) => onAnswerChange(e.target.value)}
                placeholder="Введите ответ..."
                size="sm"
                autoFocus
              />
            </Box>
          </VStack>
        )}
      </Box>
    </Box>
  );
}

