import { Container } from '@chakra-ui/react';
import { Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import { createRootRoute } from '@tanstack/react-router';

export const Route = createRootRoute({
  component: RootComponent
});

function RootComponent(): JSX.Element {
  return (
    <>
      <Container maxW="7xl" py={8} px={4}>
        <Outlet />
      </Container>
      {import.meta.env.DEV ? (
        <TanStackRouterDevtools position="bottom-right" />
      ) : null}
    </>
  );
}

