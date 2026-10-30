# tiny local server; the .wasm file needs the right MIME type
import http.server, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript"}
socketserver.TCPServer.allow_reuse_address = True
print("open http://localhost:8000")
socketserver.TCPServer(("", 8000), H).serve_forever()
