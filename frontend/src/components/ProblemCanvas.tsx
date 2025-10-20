import { Box, Flex, Spinner, Text } from '@chakra-ui/react';
import { useEffect, useMemo, useState, useRef } from 'react';
import { Rnd } from 'react-rnd';
import { ProblemComment } from './ProblemComment';

type BoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type Problem = {
  problem_id: string;
};

type Boxes = Record<string, BoundingBox>;

type ProblemCanvasProps = {
  imageUrl: string;
  problems: Problem[];
  boxes: Boxes;
  onBoxChange: (problemId: string, box: BoundingBox) => void;
  answers?: Record<string, string>;
  onAnswerChange?: (problemId: string, answer: string) => void;
  expandedComments?: Set<string>;
  onCommentToggle?: (problemId: string) => void;
};

const palette = ['#3182ce', '#38a169', '#d69e2e', '#d53f8c', '#805ad5'];

export function ProblemCanvas({ 
  imageUrl, 
  problems, 
  boxes, 
  onBoxChange, 
  answers = {}, 
  onAnswerChange,
  expandedComments = new Set(),
  onCommentToggle
}: ProblemCanvasProps) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const [displayedWidth, setDisplayedWidth] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Reset dimensions when imageUrl changes to prevent race condition
    setDimensions(null);
    setDisplayedWidth(null);
    
    const img = new window.Image();
    img.onload = () => {
      setDimensions({ width: img.width, height: img.height });
    };
    img.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const updateDisplayedWidth = () => {
      if (containerRef.current) {
        setDisplayedWidth(containerRef.current.offsetWidth);
      }
    };
    
    updateDisplayedWidth();
    window.addEventListener('resize', updateDisplayedWidth);
    return () => window.removeEventListener('resize', updateDisplayedWidth);
  }, [containerRef, dimensions]);

  const scale = useMemo(() => {
    if (!dimensions || !displayedWidth) return 1;
    return Math.min(displayedWidth / dimensions.width, 1);
  }, [dimensions, displayedWidth]);

  const preparedBoxes = useMemo(() => {
    const prepared = problems
      .map((problem, index) => {
        const box = boxes[problem.problem_id];
        if (!box) {
          return null;
        }
        return {
          problem,
          box: {
            x: box.x * scale,
            y: box.y * scale,
            width: box.width * scale,
            height: box.height * scale
          },
          color: palette[index % palette.length]
        };
      })
      .filter(Boolean) as { problem: Problem; box: BoundingBox; color: string }[];
    
    return prepared;
  }, [boxes, problems, scale]);

  if (!dimensions) {
    return (
      <Flex align="center" justify="center" h="300px" bg="gray.50" rounded="md">
        <Spinner />
      </Flex>
    );
  }

  return (
    <Box position="relative" width="100%" overflow="visible">
      <Box 
        ref={containerRef}
        position="relative" 
        width="100%"
        maxW={`${dimensions.width}px`}
        bg="gray.100" 
        rounded="md" 
        borderWidth="1px" 
        style={{ 
          userSelect: 'none',
          aspectRatio: `${dimensions.width} / ${dimensions.height}`
        }}
      >
      <img 
        src={imageUrl} 
        alt="Страница" 
        draggable={false}
        style={{ 
          width: '100%', 
          height: '100%', 
          display: 'block',
          pointerEvents: 'none',
          position: 'absolute',
          top: 0,
          left: 0,
          zIndex: 0
        }} 
      />
      {preparedBoxes.map(({ problem, box, color }) => (
        <Rnd
          key={problem.problem_id}
          bounds="parent"
          size={{ width: Math.max(box.width, 40), height: Math.max(box.height, 40) }}
          position={{ x: box.x, y: box.y }}
          onDragStop={(_, data) => {
            // Convert back to original scale when saving
            const originalBox = boxes[problem.problem_id];
            if (originalBox) {
              onBoxChange(problem.problem_id, { 
                ...originalBox, 
                x: data.x / scale, 
                y: data.y / scale 
              });
            }
          }}
          onResizeStop={(_, __, ref, delta, position) => {
            // Convert back to original scale when saving
            onBoxChange(problem.problem_id, {
              x: position.x / scale,
              y: position.y / scale,
              width: Number(ref.style.width.replace('px', '')) / scale,
              height: Number(ref.style.height.replace('px', '')) / scale
            });
          }}
          minWidth={40}
          minHeight={40}
          style={{ 
            position: 'absolute',
            border: `3px solid ${color}`, 
            boxShadow: `0 0 8px ${color}80`,
            background: `${color}20`,
            cursor: 'move',
            zIndex: 100
          }}
          resizeHandleStyles={{
            bottom: { zIndex: 101 },
            bottomLeft: { zIndex: 101 },
            bottomRight: { zIndex: 101 },
            left: { zIndex: 101 },
            right: { zIndex: 101 },
            top: { zIndex: 101 },
            topLeft: { zIndex: 101 },
            topRight: { zIndex: 101 }
          }}
        >
          <Box
            position="absolute"
            top="-24px"
            left="0"
            px={2}
            py={1}
            bg={color}
            color="white"
            rounded="md"
            shadow="md"
            fontSize="xs"
            fontWeight="bold"
            whiteSpace="nowrap"
          >
            Задача {problem.problem_id}
          </Box>
        </Rnd>
      ))}
      
      {/* Comments positioned outside the image */}
      {onAnswerChange && onCommentToggle && (() => {
        // Calculate offsets to prevent overlaps
        const COLLAPSED_HEIGHT = 40;
        const EXPANDED_HEIGHT = 120;
        const MIN_GAP = 8;
        
        const sortedBoxes = [...preparedBoxes].sort((a, b) => a.box.y - b.box.y);
        const offsets: Record<string, number> = {};
        
        sortedBoxes.forEach((item, index) => {
          if (index === 0) {
            offsets[item.problem.problem_id] = 0;
            return;
          }
          
          const prevItem = sortedBoxes[index - 1];
          const prevY = prevItem.box.y + (offsets[prevItem.problem.problem_id] || 0);
          const prevHeight = expandedComments.has(prevItem.problem.problem_id) ? EXPANDED_HEIGHT : COLLAPSED_HEIGHT;
          const prevBottom = prevY + prevHeight;
          
          const currentY = item.box.y;
          const neededOffset = Math.max(0, prevBottom + MIN_GAP - currentY);
          
          offsets[item.problem.problem_id] = neededOffset;
        });
        
        return preparedBoxes.map(({ problem, box }) => (
          <ProblemComment
            key={`comment-${problem.problem_id}`}
            problemId={problem.problem_id}
            yPosition={box.y}
            yOffset={offsets[problem.problem_id] || 0}
            answer={answers[problem.problem_id] || ''}
            onAnswerChange={(answer) => onAnswerChange(problem.problem_id, answer)}
            isExpanded={expandedComments.has(problem.problem_id)}
            onToggle={() => onCommentToggle(problem.problem_id)}
          />
        ));
      })()}
      </Box>
    </Box>
  );
}
