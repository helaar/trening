# syntax=docker/dockerfile:1
FROM node:22-alpine AS base
WORKDIR /app

FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# --- dev target: used by docker-compose.override.yml with source bind-mounted ---

FROM deps AS dev
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- build target: compiles the SPA ---

FROM deps AS build
COPY . .
RUN npm run build

# --- prod target: Caddy serving the built SPA and reverse-proxying the API ---

FROM caddy:2-alpine AS prod
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
EXPOSE 80 443
