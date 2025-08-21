import { ApolloClient, InMemoryCache, split, HttpLink } from '@apollo/client';
import { getMainDefinition } from '@apollo/client/utilities';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';

// HTTP connection to the API
const httpLink = new HttpLink({
  uri: process.env.NEXT_PUBLIC_GRAPHQL_URL || 'http://localhost:4001/graphql',
  credentials: 'include',
});

// WebSocket connection for subscriptions
const wsLink = typeof window !== 'undefined'
  ? new GraphQLWsLink(
      createClient({
        url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:4001/graphql/ws',
        connectionParams: {
          authToken: typeof window !== 'undefined' 
            ? localStorage.getItem('authToken') 
            : null,
        },
      })
    )
  : null;

// Split based on operation type
const splitLink = typeof window !== 'undefined' && wsLink
  ? split(
      ({ query }) => {
        const definition = getMainDefinition(query);
        return (
          definition.kind === 'OperationDefinition' &&
          definition.operation === 'subscription'
        );
      },
      wsLink,
      httpLink
    )
  : httpLink;

// Apollo Client instance
export const apolloClient = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache({
    typePolicies: {
      TestRun: {
        keyFields: ['id'],
      },
      ScaffoldResult: {
        keyFields: ['id'],
      },
    },
  }),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
    },
  },
});

// Type-safe GraphQL operations
export { gql } from '@apollo/client';

// Export types
export type { ApolloClient } from '@apollo/client';