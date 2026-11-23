"""HTTPS on your local network for phone tests (standard library plus the openssl command).

Phones only allow the camera on HTTPS or localhost. This listens on the LAN with a self-signed certificate and
forwards to the app, which stays bound to 127.0.0.1:8000: only this proxy is reachable from the network.
Nothing is downloaded, nothing goes to the internet, the certificate lasts 30 days.

    python app/server.py                       # terminal 1 (default 127.0.0.1:8000)
    python tests/phone/tls_proxy.py            # terminal 2; prints the https address to open on the phone
"""
import argparse, asyncio, os, socket, ssl, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]       # no packet is sent; this only picks the Wi-Fi/LAN interface
    finally: s.close()

def make_cert(ip, cert, key):
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-keyout", key, "-out", cert, "-subj", "/CN=rehab-form-dev",
                    "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chmod(key, 0o600)

async def pipe(r, w):
    try:
        while (d := await r.read(65536)): w.write(d); await w.drain()
    except Exception: pass
    finally:
        try: w.close()
        except Exception: pass

async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443); ap.add_argument("--backend", type=int, default=8000); ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args(); ip = lan_ip(); certs = os.path.join(HERE, "certs"); os.makedirs(certs, exist_ok=True)
    cert, key = os.path.join(certs, f"{ip}.crt"), os.path.join(certs, f"{ip}.key")
    if not (os.path.exists(cert) and os.path.exists(key)): make_cert(ip, cert, key)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)
    async def handle(cr, cw):
        try: br, bw = await asyncio.open_connection("127.0.0.1", a.backend)
        except OSError: cw.close(); return
        await asyncio.gather(pipe(cr, bw), pipe(br, cw))
    srv = await asyncio.start_server(handle, a.host, a.port, ssl=ctx)
    print(f"Open on the phone (same Wi-Fi): https://{ip}:{a.port}/\nFirst time: accept the certificate warning. Add ?diag=1 for frame-rate numbers.\nForwarding to 127.0.0.1:{a.backend}. Ctrl-C to stop.", flush=True)
    async with srv: await srv.serve_forever()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
