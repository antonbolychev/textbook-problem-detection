FROM node:23-alpine

WORKDIR /app

COPY package.json ./

RUN npm install

COPY tsconfig.json ./
COPY tsconfig.node.json ./
COPY vite.config.ts ./
COPY index.html ./
COPY src ./src

CMD ["npm", "run", "dev"]
