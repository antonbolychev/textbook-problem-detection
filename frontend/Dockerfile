FROM node:24-alpine AS build-stage

WORKDIR /app

COPY package*.json ./

RUN npm ci --include=dev

COPY . .

ARG VITE_API_BASE_URL
ARG VITE_WS_BASE_URL
ARG NODE_ENV=production

ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_WS_BASE_URL=${VITE_WS_BASE_URL}
ENV NODE_ENV=${NODE_ENV}

RUN npm run build

FROM nginx:1.27-alpine AS production-stage

COPY --from=build-stage /app/dist/ /usr/share/nginx/html
COPY nginx-global.conf /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found.conf
