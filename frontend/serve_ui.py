import http.server
import socketserver
import os
import sys
import urllib.request
import urllib.error

PORT = 9092
BACKEND_URL = "http://localhost:9093"
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/standalone.html"
            return super().do_GET()
        if self.path.startswith("/api/"):
            return self._proxy("GET")
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._proxy("POST")
        return super().do_POST()

    def _proxy(self, method):
        target_path = self.path[len("/api"):]
        target_url = f"{BACKEND_URL}{target_path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req = urllib.request.Request(target_url, data=body, method=method)
        for key, val in self.headers.items():
            if key.lower() not in ["host", "content-length"]:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f'{{"error": "Failed to connect to backend at {BACKEND_URL}: {e}"}}'.encode("utf-8"))


def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"React UI server running on port {PORT}: http://localhost:{PORT}")
        print(f"Proxying /api/* requests to: {BACKEND_URL}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    run()
