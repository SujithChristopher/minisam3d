# Radxa Dragon Q6A: campus internet login (Sophos)

The CMC network sits behind a Sophos UTM firewall that only passes traffic for logged-in clients. The Radxa
shares the lab router (`192.168.0.1`) with a laptop running the Windows Sophos Client Authentication Agent
(CAA); when that laptop disconnects, the Radxa loses internet too. The Radxa can run the same agent itself.

**Status: installed but disabled.** Nothing starts it at boot.

## What is on the board

| path | what |
|---|---|
| `/usr/bin/qemu-x86_64-static` | `qemu-user-static` (apt). The agent is an x86-64 static binary; the Radxa is ARM64 |
| `~/bin/caa` | Sophos CAA 1.2.0 for Linux, from `https://ns1.cmcvellore.ac.in/clients/caa_x64.tar.gz` |
| `~/.caa/caa.conf` | host `1.2.3.4`, username, password (encrypted by the agent on first run), mode 600 |
| `~/.caa/ca-cert.pem` | firewall CA from the same tarball |
| `~/.caa/caa.log` | log when started with `-l` |

The agent connects to `1.2.3.4:9922` (TLS), the same endpoint the Windows agent uses. The account accepts two
simultaneous sessions, so the laptop and the Radxa can both be logged in.

## Enable for this boot only

```bash
cd ~ && qemu-x86_64-static ~/bin/caa -d -l ~/.caa/caa.log
tail -f ~/.caa/caa.log            # expect "Login was accepted."
```

Stop it:

```bash
qemu-x86_64-static ~/bin/caa -s
```

## Enable permanently (starts at boot, restarts if it exits)

```bash
sudo tee /etc/systemd/system/sophos-caa.service >/dev/null <<'EOF'
[Unit]
Description=Sophos Client Authentication Agent (x86-64 via qemu)
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
User=radxa
WorkingDirectory=/home/radxa
ExecStart=/usr/bin/qemu-x86_64-static /home/radxa/bin/caa -v
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now sophos-caa
journalctl -u sophos-caa -n 20     # expect "Login was accepted."
```

Disable again:

```bash
sudo systemctl disable --now sophos-caa
```

## Changing the password

Edit `~/.caa/caa.conf` and put the new password in plain text on the `Password:` line. The agent replaces it
with an encrypted value on its next start. Format:

```
Copernicus host: 1.2.3.4
Username: <user>
Password: <password>
```

## Log messages

- `Login was accepted.` -- logged in; the firewall then sends `PING` about every 45 s and the agent answers `PONG`.
- `Function timer_create() failed!` after each ping -- harmless; qemu-user does not implement the timer the
  agent asks for. Login, keepalive and reconnect still work.
- `Login was denied. Username/password invalid!` -- wrong credentials. Stop the service before it retries, so
  the account is not locked.
- `Too many user logins!` -- the account has hit its session limit.
