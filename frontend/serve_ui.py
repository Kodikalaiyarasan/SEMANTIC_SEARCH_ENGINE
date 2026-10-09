import http.server
import socketserver
import os
import sys

PORT = 9092
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # Serve standalone.html for root path
        if self.path == '/' or self.path == '/index.html':
            self.path = '/standalone.html'
        return super().do_GET()

def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"React UI server running on port {PORT}: http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == '__main__':
    run()
