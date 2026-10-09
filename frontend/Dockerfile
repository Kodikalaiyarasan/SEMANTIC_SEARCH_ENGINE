FROM nginx:alpine

# Copy custom nginx configuration listening on port 9092
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend static files
COPY . /usr/share/nginx/html

EXPOSE 9092

CMD ["nginx", "-g", "daemon off;"]
