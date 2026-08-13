# Test restart recovery in an isolated Linux virtual machine

Run destructive restart tests inside a Linux virtual machine with its own Docker Engine and writable disk.

The host keeps production services running. The virtual machine receives one read-only backup and three narrow service tunnels. It receives no production volume or Docker socket.

## Prepare the virtual machine

Use a Linux virtual machine from the host's virtual machine system. The guest needs Docker Engine and enough local disk space.

Use the virtual machine system available on the host. The procedure does not depend on a specific host operating system or virtual machine product.

Start a Linux guest with the CPU, memory, disk, network, and read-only sharing described below. Then continue with the guest steps.

The macOS acceptance environment used [Tart](https://tart.run/quick-start/). The Tart commands below are one concrete example. Use equivalent controls with another virtual machine system.

Store the virtual machine disk on the external test drive. Give the guest enough space for one restored source and one complete writable case.

The harness requires 125 percent of the total archive size before restoration. It requires the same reserve before every case.

Stop if either the host or guest reports less space. Do not move or delete unrelated files to create space.

For the macOS Tart example, configure the guest from the host:

```sh
export TART_HOME="${TART_HOME:?set the Tart storage directory}"
export TART_NO_AUTO_PRUNE=1
export VM_NAME="${VM_NAME:?set the Linux virtual machine name}"
export BACKUP_ROOT="${BACKUP_ROOT:?set the verified raw backup directory}"

tart set "$VM_NAME" --cpu 2 --memory 32768 --disk-size 400
tart run \
    --no-graphics \
    --net-softnet \
    --dir "$BACKUP_ROOT:ro,tag=lms-backup" \
    "$VM_NAME"
```

Keep `TART_NO_AUTO_PRUNE=1`. Tart must not delete cached images during acceptance work.

Use a Linux guest. A macOS guest would need another virtual machine to run Linux containers.

## Mount only the backup

Expose the backup as a read-only share with the chosen virtual machine system. Verify the read-only setting before extraction.

For Tart, mount the verified backup inside the guest:

```sh
export BACKUP_MOUNT="${BACKUP_MOUNT:?set the guest backup mount directory}"
sudo mkdir -p "$BACKUP_MOUNT"
sudo mount -t virtiofs -o ro lms-backup "$BACKUP_MOUNT"
findmnt "$BACKUP_MOUNT"
```

The output must include `ro`. Stop if it reports a writable mount.

Create the writable run parent on the guest disk:

```sh
export LMS_RESTART_ACCEPTANCE_RUN_PARENT="${LMS_RESTART_ACCEPTANCE_RUN_PARENT:?set the guest run directory}"
sudo mkdir -p "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
sudo chown "$USER" "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
df -h / "$LMS_RESTART_ACCEPTANCE_RUN_PARENT"
```

This path belongs to the guest disk. Do not mount the host run directory there.

## Install Docker and the tested binaries

Install Docker Engine and the Compose plugin by following the [Docker Engine Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).

Verify the engine through a real container:

```sh
docker info
docker compose version
docker run --rm hello-world
```

Pull the etcd, MinIO, and Milvus image tags recorded by the backup. The harness checks each local image identifier before restoration.

Install LMS and Clyde from the exact merged commits under test. Both projects install their binaries under the guest user's local binary directory.

```sh
make install
lm-semantic-search version
lm-semantic-search-daemon version
clyde --version
```

Record each reported commit. Stop if an installed binary reports another commit or a dirty build.

## Bridge only required production endpoints

The harness reads production status before and after each isolated case. It also runs the final production-safe searches.

Do not share production volumes with the guest. Use one Secure Shell tunnel for the daemon socket, Milvus, and the embedding service.

Create a unique socket directory in the guest:

```sh
export BRIDGE_ROOT="$(mktemp -d /tmp/lms-production-bridge.XXXXXXXX)"
export BRIDGE_SOCKET="$BRIDGE_ROOT/daemon.sock"
printf '%s\n' "$BRIDGE_SOCKET"
```

On the host, set `PRODUCTION_SOCKET` to the socket shown by the installed daemon status. Set `GUEST_IP` to the virtual machine address.

Keep this host command running during the suite:

```sh
ssh -N \
    -o ExitOnForwardFailure=yes \
    -R "${BRIDGE_SOCKET}:${PRODUCTION_SOCKET}" \
    -R 127.0.0.1:19531:127.0.0.1:19530 \
    -R 127.0.0.1:15400:127.0.0.1:5400 \
    "admin@${GUEST_IP}"
```

The guest receives only three loopback endpoints. The isolated LMS process still uses separate clone and proxy ports.

## Run the acceptance suite

In the guest, select the production bridge and read-only backup:

```sh
export PATH="$HOME/.local/bin:$PATH"
export LMS_RESTART_ACCEPTANCE_BACKUP="${BACKUP_MOUNT:?set the guest backup mount directory}"
export LMS_RESTART_ACCEPTANCE_RUN_PARENT="${LMS_RESTART_ACCEPTANCE_RUN_PARENT:?set the guest run directory}"
export CLAUDE_CONTEXTD_SOCKET_PATH="$BRIDGE_SOCKET"
export MILVUS_ADDRESS=127.0.0.1:19531
export MILVUS_DATABASE=default
export OPENAI_BASE_URL=http://127.0.0.1:15400/v1
```

Export the same embedding provider, model, dimension, and hybrid setting used by production. Pass required credentials through environment variables only.

Run the deterministic suite first:

```sh
make restart-acceptance-unit
```

Then provide both exact confirmations and run the full suite:

```sh
export LMS_RESTART_ACCEPTANCE_CONFIRM=isolated-clone
export LMS_PRODUCTION_CONFIRM_DATABASE=default
make restart-acceptance
```

The full run must execute all eight cases without a skip. It must also pass the production census and final cold searches.

## Verify cleanup and evidence

The run succeeds only when every case removes its containers, volumes, processes, sockets, and writable case directory.

Verify Docker after the command exits:

```sh
docker ps -a --filter name=lms-restart
docker volume ls --filter name=lms-restart
```

Both commands must return no acceptance resource.

Keep the result and event artifacts under the completed run directory. Attach the exact command output and final artifact to the acceptance ticket.

Stop the tunnel after evidence capture. Then stop the virtual machine with the host's virtual machine system.

For the macOS Tart example, run:

```sh
tart stop --timeout 60 "$VM_NAME"
```

Stopping the virtual machine releases every guest Docker process and deleted-file handle. Keep the virtual machine image and raw backup.

If any case needs manual repair, record the failure and repeat the complete suite from a clean run. Do not mark a repaired run as accepted.
